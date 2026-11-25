"""Generation settings turned into the request: texts as sent, source images, Vibes, references and lint."""

from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from nai_cli import caches, enhance, images, inpaint, lint, local_models, models, sam, vibe
from nai_cli import prompt as P
from nai_cli import request as R
from nai_cli.commands.generation import (
    FROM_IMAGE,
    Blank,
    CharacterText,
    Enhanced,
    FromImage,
    GenSettings,
    Origin,
    Repaint,
    Text,
)
from nai_cli.commands.references import load_precise, load_vibes


@dataclass
class Prepared:
    spec: R.GenSpec
    built: R.Built
    job: inpaint.Job | None
    vibe_sources: list[vibe.VibeSource]
    pending_encodes: list[vibe.Encode]
    warnings: list[str]
    issues: list[lint.Issue]
    tokens: lint.TokenCount


def prepare(s: GenSettings) -> Prepared:
    """Everything up to pricing, shared by the generation commands and lint."""
    sources = [*s.stored_vibes, *load_vibes(s.vibes, s.model.id)]
    # Checked before any reference is prepared or encoded for a model that would refuse it
    outfits = sum(r.kind == R.OUTFIT_KIND for r in s.references)
    R.check_references(s.model, s.action, len(sources), len(s.references), outfits)
    spec, warnings, job = _spec(s)
    pending = attach_vibes(spec, sources)
    # A damaged image fails here, before any other Vibe is paid for
    for encode in pending:
        vibe.check_image(encode.source)
    built = R.build(spec)
    warnings += built.warnings
    issues, tokens = _lint(spec, built)
    return Prepared(spec, built, job, sources, pending, warnings, issues, tokens)


def attach_vibes(
    spec: R.GenSpec, sources: list[vibe.VibeSource], fresh: dict[int, str] | None = None
) -> list[vibe.Encode]:
    """Set the Vibes from encodings just made, stored in their files or cached; returns the encodings still to
    pay for when they are sent."""
    sent = R.vibes_sent(spec.action, len(spec.references))
    fresh = fresh or {}
    spec.vibes, pending = [], []
    for i, source in enumerate(sources):
        # Vibes that are not sent need no encoding, not even a stored one
        encoding = fresh.get(i) or (caches.vibe_encoding(source, spec.model.id) if sent else None)
        if encoding is None and sent:
            pending.append((i, source))
        spec.vibes.append(R.Vibe(encoding or "", source.strength))
    return vibe.encodes(pending, spec.model.id)


def _spec(s: GenSettings) -> tuple[R.GenSpec, list[str], inpaint.Job | None]:
    warnings = list(s.warnings)
    canvas = s.canvas
    (width, height), image_bytes, mask_bytes, job = _canvas_input(s, warnings)
    plan = canvas.plan if isinstance(canvas, Enhanced) else None
    chars = _characters(s, job, warnings)
    spec = R.GenSpec(
        model=s.model,
        prompt=_sent(s, s.prompt, s.drawn.prompt if s.drawn else None, s.fixed.prompt if s.fixed else None),
        negative=_sent(s, s.negative, s.drawn.negative if s.drawn else None, s.fixed.negative if s.fixed else None),
        characters=chars,
        action=s.action,
        width=width,
        height=height,
        steps=s.steps,
        scale=s.cfg,
        sampler=s.sampler,
        noise_schedule=s.noise_schedule,
        cfg_rescale=s.cfg_rescale,
        seed=s.seed,
        # Variations count follows the size requested; a source above the pixel limit would otherwise allow none
        n_samples=s.n if s.n is not None else enhance.variation_samples(width, height),
        quality=s.quality,
        uc_preset=s.uc_preset,
        negative_verbatim=s.negative_verbatim,
        official_nsfw=s.official_nsfw,
        use_coords=s.use_coords,
        auto_text=s.auto_text,
        normalize_vibes=s.normalize_vibes,
        transparent=s.transparent,
        straight_alpha=not s.premultiplied_alpha,
        furry=s.furry,
        variety=s.variety,
        image=image_bytes,
        mask=mask_bytes,
        strength=s.strength,
        noise=s.noise,
        inpaint_strength=canvas.area.strength if isinstance(canvas, Repaint) else 1.0,
        enhance_hint=plan is not None and not plan.upscaled,
        upscaled_enhance=plan is not None and plan.upscaled,
        legacy_uc=s.legacy_uc,
        references=load_precise(s.references, s.model),
    )
    return spec, warnings, job


def _canvas_input(
    s: GenSettings, warnings: list[str]
) -> tuple[tuple[int, int], bytes | None, bytes | None, inpaint.Job | None]:
    """The request size, the image and mask to send, and the inpainting job when there is one."""
    keep_alpha = models.for_action(s.model, s.action).transparent
    match s.canvas:
        case Repaint() as canvas:
            job = _inpaint_job(s, canvas, keep_alpha)
            warnings += job.warnings
            return (job.width, job.height), job.image, job.mask, job
        case Enhanced() as canvas:
            image = images.source_image(canvas.image, *canvas.plan.image_size, keep_alpha=keep_alpha)
            return canvas.plan.request_size, image, None, None
        case FromImage() as canvas:
            size = _import_size(canvas, s.current_size, warnings)
            return size, images.source_image(canvas.image, *size, keep_alpha=keep_alpha), None, None
        case Blank() as canvas:
            return canvas.size, None, None, None


def _import_size(canvas: FromImage, current: tuple[int, int], warnings: list[str]) -> tuple[int, int]:
    """The size given, or else the size the official client gives an imported image."""
    if canvas.size is not None:
        return canvas.size
    source_w, source_h = images.size_of(canvas.image)
    width, height = images.import_size(source_w, source_h, current)
    if width * height < source_w * source_h:
        warnings.append(
            f"source {source_w}x{source_h} is sent at {width}x{height}, as the official client imports it; "
            "use --size for a larger request (above 1 MP costs Anlas)"
        )
    return width, height


def _inpaint_job(s: GenSettings, canvas: Repaint, keep_alpha: bool) -> inpaint.Job:
    area = canvas.area
    return inpaint.prepare(
        images.open_image(canvas.image),
        mask=images.open_image(area.mask) if area.mask else None,
        regions=[inpaint.parse_region(r) for r in area.regions],
        grow_pixels=area.grow,
        expand=inpaint.parse_expand(area.expand) if area.expand else None,
        size=canvas.size,
        current=s.current_size,
        focus=area.focus,
        padding=area.focus_padding,
        keep_alpha=keep_alpha,
        segment=segmenter(area.sam_dir),
    )


def segmenter(folder: Path) -> inpaint.Segment:
    """Object regions segmented with the SAM model in folder, loaded only when one asks for it."""

    def segment(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
        return sam.segment(image, points, *local_models.sam(folder))

    return segment


def _sent(s: GenSettings, text: Text, drawn: tuple[str, str] | None, fixed: tuple[str, str] | None) -> str:
    """The text as sent: the source image's draws, then tidying, then the fixed tags for texts typed."""
    value = P.fill_drawn(text.value, *drawn) if drawn else text.value
    value = s.tidy[text.origin].apply(value)
    if fixed is None or text.origin is not Origin.TYPED:
        return value
    return P.with_fixed(value, *fixed)


def _characters(s: GenSettings, job: inpaint.Job | None, warnings: list[str]) -> list[R.Character]:
    drawn = s.drawn
    # The official client sends a character's draws again only while the image's characters are all there
    paired = drawn is not None and len(drawn.characters) == len(s.characters)
    negatives_paired = drawn is not None and len(drawn.character_negatives) == len(s.characters)
    chars = [
        _character(
            s,
            c,
            drawn.characters[i] if paired and drawn else None,
            drawn.character_negatives[i] if negatives_paired and drawn else None,
            job,
        )
        for i, c in enumerate(s.characters)
    ]
    # Enhance and variations send the image's characters as stored; only the editor rewrites them
    if s.command in FROM_IMAGE:
        return chars
    sent = [replace(c, prompt=P.official_character_prompt(c.prompt)) for c in chars]
    if sent != chars:
        warnings.append("1girl and 1boy in character prompts are sent as girl and boy, as the official client does")
    return sent


def _character(
    s: GenSettings,
    c: CharacterText,
    drawn: tuple[str, str] | None,
    drawn_negative: tuple[str, str] | None,
    job: inpaint.Job | None,
) -> R.Character:
    x, y = c.position if c.position is not None else (None, None)
    if job is not None and x is not None and y is not None:
        x, y = job.remap(x, y)
    # A model that sends no custom negatives drops one filled in for the user without a warning about it
    negative = "" if s.model.fixed and c.negative.origin is Origin.MODEL else _sent(s, c.negative, drawn_negative, None)
    return R.Character(_sent(s, c.prompt, drawn, None), negative, x, y)


def _lint(spec: R.GenSpec, built: R.Built) -> tuple[list[lint.Issue], lint.TokenCount]:
    # Counted on the texts as sent, the way the official prompt boxes count them
    tokens = lint.count_tokens(
        built.model,
        caches.load_tokenizer(built.model.tokenizer),
        built.prompt,
        built.negative,
        [(c.prompt, c.negative) for c in built.characters],
    )
    issues = lint.run(
        built.model,
        spec.prompt,
        spec.negative,
        [(c.prompt, c.negative) for c in spec.characters],
        spec.quality,
        built.prompt,
        spec.uc_preset,
        spec.transparent,
        spec.negative_verbatim,
        tokens,
    )
    return issues, tokens
