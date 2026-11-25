"""Settings of the generation commands, resolved once from options, an imported image and the config."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from nai_cli import (
    comfyui,
    config,
    enhance,
    errors,
    images,
    import_image,
    inpaint,
    meta,
    models,
    presets,
    random_prompt,
    request,
    tidy,
    vibe,
)
from nai_cli.commands import output, texts
from nai_cli.commands.context import Context
from nai_cli.commands.references import PreciseRef, VibeRef, precise_ref, vibe_ref

ACTIONS = {
    "gen": "generate",
    "lint": "generate",
    "i2i": "img2img",
    "enhance": "img2img",
    "variations": "img2img",
    "inpaint": "infill",
}
# They start from the settings of the image they work on
FROM_IMAGE = ("enhance", "variations")
_TEXT_OPTIONS = ("prompt", "neg", "char", "char_neg")


class Origin(Enum):
    TYPED = "typed"
    # Kept as stored: tidied only by switches given for this run, never by the config, and without fixed tags
    IMAGE = "image"
    # Filled in for the user, as the official client fills a new character's box; kept like IMAGE
    MODEL = "model"


@dataclass(frozen=True)
class Text:
    value: str
    origin: Origin = Origin.TYPED


@dataclass(frozen=True)
class CharacterText:
    prompt: Text
    negative: Text
    position: tuple[float, float] | None = None


@dataclass(frozen=True)
class Tidy:
    convert_sd_weights: bool = False
    format_prompt: bool = False
    underscores: bool = False
    space_numeric_close: bool = False

    def apply(self, text: str) -> str:
        if self.convert_sd_weights:
            text = tidy.convert_sd_weights(text)
        if self.format_prompt:
            text = tidy.format_prompt(text, self.underscores)
        return tidy.space_numeric_closings(text) if self.space_numeric_close else text


@dataclass(frozen=True)
class Fixed:
    """Tags joined around a text typed for this run: (prefix, suffix)."""

    prompt: tuple[str, str] = ("", "")
    negative: tuple[str, str] = ("", "")


@dataclass(frozen=True)
class InpaintArea:
    mask: Path | None
    regions: tuple[str, ...]
    grow: int
    expand: str | None
    focus: bool | None
    focus_padding: int
    strength: float
    preview: bool
    keep_raw: bool
    # Where object regions find the SAM model
    sam_dir: Path


@dataclass(frozen=True)
class Blank:
    size: tuple[int, int]


@dataclass(frozen=True)
class FromImage:
    image: Path
    # None sizes the request as the official client sizes an imported image
    size: tuple[int, int] | None


@dataclass(frozen=True)
class Enhanced:
    image: Path
    plan: enhance.Plan


@dataclass(frozen=True)
class Repaint:
    image: Path
    # Without focus; None lets the area decide
    size: tuple[int, int] | None
    area: InpaintArea


Canvas = Blank | FromImage | Enhanced | Repaint


@dataclass(frozen=True)
class GenSettings:
    command: str
    action: str
    model: models.Model
    prompt: Text
    negative: Text
    characters: tuple[CharacterText, ...]
    use_coords: bool
    tidy: dict[Origin, Tidy]
    fixed: Fixed | None
    canvas: Canvas
    # The config's size, which sizes imported images as the official client's current size does
    current_size: tuple[int, int]
    steps: int | None
    cfg: float | None
    sampler: str
    noise_schedule: str
    cfg_rescale: float
    seed: int | None
    n: int | None
    quality: str
    uc_preset: str
    negative_verbatim: bool
    official_nsfw: bool
    auto_text: bool
    normalize_vibes: bool
    transparent: bool
    premultiplied_alpha: bool
    furry: bool
    variety: bool
    legacy_uc: bool
    strength: float
    noise: float
    # Vibes the image's settings keep come first, as the official import puts them in the list
    stored_vibes: tuple[vibe.VibeSource, ...]
    vibes: tuple[VibeRef, ...]
    references: tuple[PreciseRef, ...]
    # The draws of the source image's ||a|b|| groups, which the official client sends again
    drawn: meta.Drawn | None
    output: output.Output
    dry_run: bool
    yes: bool
    show_body: bool
    warnings: tuple[str, ...]


def resolve(opts: dict[str, Any], ctx: Context, command: str) -> GenSettings:
    cfg = ctx.config
    d = cfg.defaults
    if command == "enhance" and opts.get("size"):
        raise errors.NaiError(errors.BAD_ARGS, "Enhance sizes the request with --scale; --size is not allowed")
    typed = texts.read_typed(opts, _TEXT_OPTIONS, ctx)
    chosen = models.resolve(opts["model"]) if opts.get("model") else None
    default = models.resolve(d.get("model"))

    def configured(model: models.Model) -> models.Model:
        picked = models.with_effort(model, opts.get("effort"), d.get("effort"))
        return models.with_hidden_options(picked, bool(d.get("v5_hidden_options")))

    imported = _imported(opts, command, chosen, default, configured)
    if command in FROM_IMAGE and imported is None and not typed["prompt"] and not opts.get("random_prompt"):
        raise errors.NaiError(errors.NO_METADATA, "the image has no generation settings; describe it with --prompt")
    got = imported or import_image.Imported(configured(chosen or default))
    warnings = list(got.warnings)
    references = _references(opts, warnings)
    model = models.for_references(got.model, bool(references))
    if model is not got.model:
        model = models.with_hidden_options(model, bool(d.get("v5_hidden_options")))
        warnings.append(models.HIGH_EFFORT_FOR_REFERENCES)
    written = _random_prompt(opts, typed) or _comfyui(opts, d, typed, model)
    characters, use_coords = (
        (written.characters, written.use_coords)
        if written is not None
        else _characters(typed, opts.get("char_pos"), got, model)
    )
    sampler = models.resolve_sampler(_first(opts.get("sampler"), got.sampler, d.get("sampler"), "k_euler_ancestral"))
    canvas = _canvas(opts, got, cfg, command, model, warnings)
    premultiplied = _first(
        opts.get("premultiplied_alpha"), got.premultiplied_alpha, d.get("premultiplied_alpha"), False
    )
    if opts.get("premultiplied_alpha") and not models.for_action(model, ACTIONS[command]).transparent:
        warnings.append(f"{model.name} makes no transparent images, so --premultiplied-alpha has no effect")
    strength, noise = _image_strength(opts, command, canvas)
    return GenSettings(
        command=command,
        action=ACTIONS[command],
        model=model,
        prompt=written.prompt if written is not None else _text(typed["prompt"], got.prompt),
        negative=_text(typed["neg"], got.negative),
        characters=characters,
        use_coords=use_coords,
        tidy=_tidy(opts, d),
        fixed=None if opts.get("fixed_tags") is False else _fixed(d),
        canvas=canvas,
        current_size=presets.resolve_size(d.get("size")),
        steps=_first(opts.get("steps"), got.steps, d.get("steps")),
        cfg=_first(opts.get("cfg"), got.cfg, d.get("cfg")),
        sampler=sampler,
        noise_schedule=_noise_schedule(opts, got, d, model, sampler),
        cfg_rescale=_first(opts.get("cfg_rescale"), got.cfg_rescale, d.get("cfg_rescale"), 0.0),
        seed=_first(opts.get("seed"), got.seed),
        n=_samples(opts, d, command),
        quality=_quality(opts, got, d, model),
        uc_preset=_uc_preset(opts, got, d, model),
        negative_verbatim=_first(opts.get("neg_verbatim"), got.negative_verbatim, False),
        official_nsfw=_first(opts.get("official_nsfw"), got.official_nsfw, d.get("official_nsfw"), False),
        auto_text=_first(opts.get("auto_text"), got.auto_text, True),
        normalize_vibes=_first(opts.get("normalize_vibes"), got.normalize_vibes, True),
        transparent=_first(opts.get("transparent"), got.transparent, False),
        premultiplied_alpha=premultiplied,
        furry=_first(opts.get("furry"), got.furry, False),
        variety=_first(opts.get("variety"), got.variety, False),
        legacy_uc=_first(opts.get("legacy_uc"), got.legacy_uc, False),
        strength=strength,
        noise=noise,
        stored_vibes=_stored_vibes(opts, got),
        vibes=tuple(vibe_ref(v) for v in opts.get("vibe") or []),
        references=references,
        drawn=_drawn(canvas) if command in (*FROM_IMAGE, "inpaint") else None,
        output=output.resolve(opts.get("output"), cfg),
        dry_run=bool(opts.get("dry_run")) or ctx.dry_run,
        yes=bool(opts.get("yes")),
        show_body=bool(opts.get("show_body")),
        warnings=tuple(warnings),
    )


def _references(opts: dict[str, Any], warnings: list[str]) -> tuple[PreciseRef, ...]:
    references = tuple(precise_ref(r) for r in opts.get("ref") or [])
    # As the official grid's pr_strength and pr_fidelity, which apply to every reference
    shared = {
        field: opts[option]
        for field, option in (("strength", "ref_strength"), ("fidelity", "ref_fidelity"))
        if opts.get(option) is not None
    }
    if shared and not references:
        warnings.append("--ref-strength and --ref-fidelity have no precise reference to apply to")
    return tuple(replace(r, **shared) for r in references)


def _first(*values: Any) -> Any:
    """Command line, then the imported image, then the config, then the built-in default."""
    return next((v for v in values if v is not None), None)


def _imported(
    opts: dict[str, Any],
    command: str,
    chosen: models.Model | None,
    default: models.Model,
    configured: Callable[[models.Model], models.Model],
) -> import_image.Imported | None:
    if command in FROM_IMAGE:
        # Everything but the seed: the official client starts from the image's settings with a new seed every time
        parts = tuple(p for p in import_image.PARTS if p != "seed")
        try:
            return import_image.from_image(opts["image"], parts, chosen, default, configured, rework=True)
        except errors.NaiError as e:
            if e.code != errors.NO_METADATA:
                raise
            return None
    if not opts.get("from_image"):
        return None
    parts = import_image.parse_parts(opts.get("import_parts"))
    return import_image.from_image(
        opts["from_image"],
        parts,
        chosen,
        default,
        configured,
        clean_brackets=bool(opts.get("clean_brackets")),
        drawn_prompts=bool(opts.get("actual_prompts")),
    )


def _text(typed: str | None, imported: str | None) -> Text:
    if typed is not None:
        return Text(typed)
    return Text(imported, Origin.IMAGE) if imported is not None else Text("")


def _characters(
    typed: dict[str, Any], positions: list[str] | None, got: import_image.Imported, model: models.Model
) -> tuple[tuple[CharacterText, ...], bool]:
    """Characters given replace the image's, along with their negatives and positions, as the import dialog does
    unless told to append."""
    if typed["char"] is None and typed["char_neg"] is None and positions is None:
        chars = got.characters or []
        return tuple(
            CharacterText(
                Text(c.prompt, Origin.IMAGE),
                Text(c.negative, Origin.IMAGE),
                (c.x, c.y) if c.x is not None and c.y is not None else None,
            )
            for c in chars
        ), bool(got.use_coords)
    prompts, negatives = typed["char"] or [], typed["char_neg"] or []
    if len(negatives) > len(prompts):
        raise errors.NaiError(errors.BAD_ARGS, f"{len(negatives)} --char-neg for {len(prompts)} --char")
    preset = Text(model.character_negative, Origin.MODEL)
    chars = tuple(
        CharacterText(Text(text), Text(negatives[i]) if i < len(negatives) else preset)
        for i, text in enumerate(prompts)
    )
    return _placed(chars, positions), bool(positions)


@dataclass(frozen=True)
class _Written:
    """A prompt and characters that replace the image's and --char."""

    prompt: Text
    characters: tuple[CharacterText, ...]
    use_coords: bool


def _random_prompt(opts: dict[str, Any], typed: dict[str, Any]) -> _Written | None:
    """A random prompt and characters, which replace the image's as the dice button replaces the editor's."""
    if not opts.get("random_prompt"):
        return None
    if typed["prompt"] is not None or typed["char"] is not None or typed["char_neg"] is not None:
        raise errors.NaiError(
            errors.BAD_ARGS, "--random-prompt writes the prompt and characters; leave out --prompt and --char"
        )
    if opts.get("char_pos"):
        raise errors.NaiError(errors.BAD_ARGS, "--random-prompt places its characters itself; leave out --char-pos")
    drawn = random_prompt.generate(random_prompt.Recipe(opts["random_prompt"]))
    negative = Text(random_prompt.CHARACTER_NEGATIVE, Origin.MODEL)
    return _Written(
        Text(drawn.prompt),
        tuple(CharacterText(Text(c), negative, random_prompt.CHARACTER_CENTER) for c in drawn.characters),
        False,
    )


def _comfyui(opts: dict[str, Any], d: dict[str, Any], typed: dict[str, Any], model: models.Model) -> _Written | None:
    """A ComfyUI multi-character prompt split into characters; off by default because a single | also divides
    official prompts."""
    if not _first(opts.get("comfyui_characters"), d.get("comfyui_characters"), False) or typed["prompt"] is None:
        return None
    found = comfyui.split(typed["prompt"])
    if found is None:
        return None
    if typed["char"] is not None or typed["char_neg"] is not None or opts.get("char_pos"):
        raise errors.NaiError(
            errors.BAD_ARGS,
            "the prompt gives the characters with --comfyui-characters; leave out --char, --char-neg and --char-pos",
        )
    placed = any(c.center is not None for c in found.characters)
    taken: list[tuple[float, float]] = []
    for c in found.characters:
        # A character without a region goes to the first free place, as one added in the editor does
        taken.append(c.center or request.default_center(taken, model.freeform_position))
    negative = Text(model.character_negative, Origin.MODEL)
    return _Written(
        Text(found.base),
        tuple(
            CharacterText(Text(c.prompt), negative, center if placed else None)
            for c, center in zip(found.characters, taken, strict=True)
        ),
        placed,
    )


def _placed(characters: tuple[CharacterText, ...], positions: list[str] | None) -> tuple[CharacterText, ...]:
    """--char-pos for each character, checked within 0-1."""
    if not positions:
        return characters
    if len(positions) != len(characters):
        raise errors.NaiError(errors.BAD_ARGS, f"{len(positions)} --char-pos for {len(characters)} characters")
    out = []
    for c, pos in zip(characters, positions, strict=True):
        try:
            x, y = (float(v) for v in pos.split(","))
        except ValueError:
            raise errors.NaiError(errors.BAD_ARGS, f"position must be x,y: {pos}") from None
        # Checked before focus maps it into the crop, which clamps it
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise errors.NaiError(errors.BAD_ARGS, f"character positions must be within 0-1: {pos}")
        out.append(CharacterText(c.prompt, c.negative, (x, y)))
    return tuple(out)


def _tidy(opts: dict[str, Any], d: dict[str, Any]) -> dict[Origin, Tidy]:
    def switch(name: str, from_config: bool) -> bool:
        value = opts.get(name)
        if value is not None:
            return value
        return d.get(name, False) if from_config else False

    underscores = d.get("format_underscores", False)
    return {
        origin: Tidy(
            switch("convert_sd_weights", origin is Origin.TYPED),
            switch("format_prompt", origin is Origin.TYPED),
            underscores,
            switch("space_numeric_close", origin is Origin.TYPED),
        )
        for origin in Origin
    }


def _fixed(d: dict[str, Any]) -> Fixed:
    return Fixed(
        (d.get("prompt_prefix", ""), d.get("prompt_suffix", "")),
        (d.get("negative_prefix", ""), d.get("negative_suffix", "")),
    )


def _canvas(
    opts: dict[str, Any],
    got: import_image.Imported,
    cfg: config.Config,
    command: str,
    model: models.Model,
    warnings: list[str],
) -> Canvas:
    given = presets.resolve_size(opts["size"]) if opts.get("size") else None
    if command == "enhance":
        plan = enhance.plan(
            *images.size_of(opts["image"]),
            model,
            opts.get("level"),
            opts.get("scale"),
            opts.get("strength"),
            opts.get("noise"),
        )
        return Enhanced(Path(opts["image"]), plan)
    # Inpainting sizes the request from the image being repainted
    if command == "inpaint":
        return Repaint(Path(opts["image"]), given, _inpaint_area(opts, cfg.local.sam_dir))
    size = given or _imported_size(got, warnings)
    if ACTIONS[command] == "img2img":
        return FromImage(Path(opts["image"]), size)
    return Blank(size or presets.resolve_size(cfg.defaults.get("size")))


def _imported_size(got: import_image.Imported, warnings: list[str]) -> tuple[int, int] | None:
    if got.size is not None and got.size_note:
        warnings.append(got.size_note)
    return got.size


def _samples(opts: dict[str, Any], d: dict[str, Any], command: str) -> int | None:
    """None for variations, whose count follows the size."""
    if opts.get("n") is not None:
        return opts["n"]
    if command == "enhance":
        return 1
    if command == "variations":
        return None
    return d.get("n", 1)


def _quality(opts: dict[str, Any], got: import_image.Imported, d: dict[str, Any], model: models.Model) -> str:
    chosen = _first(opts.get("quality"), got.quality)
    # The config's light applies to every model; those without the tier keep the standard one
    if chosen is None and d.get("quality") == "light" and "light" not in presets.quality_tiers(model):
        return "standard"
    return presets.resolve_quality(model, _first(chosen, d.get("quality")))


def _uc_preset(opts: dict[str, Any], got: import_image.Imported, d: dict[str, Any], model: models.Model) -> str:
    chosen = _first(opts.get("uc_preset"), got.uc_preset)
    # The config's preset applies to every model; one without it takes the closest it has
    if chosen is None and d.get("uc_preset") is not None:
        return presets.carry_uc_preset(presets.uc_preset_name(str(d["uc_preset"])), model)
    return presets.resolve_uc_preset(model, chosen)


def _noise_schedule(
    opts: dict[str, Any], got: import_image.Imported, d: dict[str, Any], model: models.Model, sampler: str
) -> str:
    chosen = _first(opts.get("noise_schedule"), got.noise_schedule, d.get("noise_schedule"))
    keeps_schedule = model.noise_schedule and not model.is_v5
    schedule = chosen or (models.default_noise_schedule(sampler) if keeps_schedule else "karras")
    offered = models.NOISE_SCHEDULES
    if schedule not in offered:
        raise errors.NaiError(
            errors.BAD_ARGS, f"{model.name} does not take the noise schedule {schedule}", choices=offered
        )
    return schedule


def _image_strength(opts: dict[str, Any], command: str, canvas: Canvas) -> tuple[float, float]:
    if isinstance(canvas, Enhanced):
        return canvas.plan.strength, canvas.plan.noise
    if command == "variations":
        default_strength, default_noise = enhance.VARIATION_STRENGTH, enhance.VARIATION_NOISE
    else:
        default_strength, default_noise = 0.7, 0.0
    return _first(opts.get("strength"), default_strength), _first(opts.get("noise"), default_noise)


def _inpaint_area(opts: dict[str, Any], sam_dir: Path) -> InpaintArea:
    if not (opts.get("mask") or opts.get("region") or opts.get("expand")):
        raise errors.NaiError(errors.BAD_ARGS, "inpaint needs --mask, --region or --expand")
    if opts.get("size") and opts.get("focus"):
        raise errors.NaiError(
            errors.BAD_ARGS, "--focus sizes the request from the crop and cannot be combined with --size"
        )
    padding = _first(opts.get("focus_padding"), inpaint.DEFAULT_FOCUS_PADDING)
    low, high = inpaint.FOCUS_PADDING_RANGE
    if not low <= padding <= high:
        raise errors.NaiError(errors.BAD_ARGS, f"--focus-padding must be within {low}-{high} pixels")
    grow = _first(opts.get("grow"), 0)
    if grow < 0:
        raise errors.NaiError(errors.BAD_ARGS, "--grow cannot be negative")
    return InpaintArea(
        mask=Path(opts["mask"]) if opts.get("mask") else None,
        regions=tuple(opts.get("region") or ()),
        grow=grow,
        expand=opts.get("expand"),
        focus=opts.get("focus"),
        focus_padding=padding,
        strength=_first(opts.get("inpaint_strength"), 1.0),
        preview=bool(opts.get("preview")),
        keep_raw=bool(opts.get("keep_raw")),
        sam_dir=sam_dir,
    )


def _stored_vibes(opts: dict[str, Any], got: import_image.Imported) -> tuple[vibe.VibeSource, ...]:
    if opts.get("vibes_from"):
        return tuple(vibe.from_settings(opts["vibes_from"]))
    return tuple(got.vibes or ())


def _drawn(canvas: Canvas) -> meta.Drawn | None:
    """What the image being enhanced, varied or repainted drew, which the official client sends again."""
    if isinstance(canvas, Blank):
        return None
    try:
        return meta.drawn(meta.read_raw(canvas.image))
    except errors.NaiError:
        # An image without settings has nothing drawn to send again
        return None
