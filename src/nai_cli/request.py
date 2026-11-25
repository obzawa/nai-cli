"""Build request bodies identical to the official client's, key order included."""

import base64
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from nai_cli import errors, models, presets
from nai_cli import prompt as P
from nai_cli.models import Model

PARAMS_VERSION = 4
# Latent volume of 832x1216; Variety+ scales sigma by it
_VARIETY_REFERENCE_LATENTS = 4 * 104 * 152

POSITION_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
# Slot order for new characters, as computed with JS floats; recomputing in Python swaps some ties
DEFAULT_SLOTS = [
    (0.5, 0.5),
    (0.3, 0.5),
    (0.7, 0.5),
    (0.1, 0.5),
    (0.9, 0.5),
    (0.5, 0.7),
    (0.5, 0.3),
    (0.7, 0.7),
    (0.3, 0.7),
    (0.7, 0.3),
    (0.3, 0.3),
    (0.5, 0.1),
    (0.5, 0.9),
    (0.1, 0.7),
    (0.9, 0.7),
    (0.1, 0.3),
    (0.9, 0.3),
    (0.3, 0.1),
    (0.3, 0.9),
    (0.7, 0.1),
    (0.7, 0.9),
    (0.1, 0.1),
    (0.1, 0.9),
    (0.9, 0.1),
    (0.9, 0.9),
]
_FREEFORM_OCCUPIED = 0.1


def snap(value: float) -> float:
    return POSITION_GRID[min(4, max(0, math.floor(5 * value)))]


def random_seed() -> int:
    """A seed for a request that names none, in the range the official client draws from."""
    return random.randint(0, 4294967294)


def default_center(taken: list[tuple[float, float]], freeform: bool) -> tuple[float, float]:
    """First free slot, by distance with free placement and by snapped grid cell otherwise."""
    for x, y in DEFAULT_SLOTS:
        if freeform:
            occupied = any(math.hypot(tx - x, ty - y) < _FREEFORM_OCCUPIED for tx, ty in taken)
        else:
            occupied = any((snap(tx), snap(ty)) == (x, y) for tx, ty in taken)
        if not occupied:
            return x, y
    return 0.5, 0.5


@dataclass
class Character:
    prompt: str
    negative: str = ""
    x: float | None = None
    y: float | None = None


@dataclass
class Vibe:
    encoding: str
    strength: float


OUTFIT_KIND = "costume"
# Outfit preferences fall back to the plain kinds on a model without outfit references
_PLAIN_KINDS = {"character_mainoutfit": "character", "character&style_anyoutfit": "character&style"}


@dataclass
class PreciseReference:
    image: bytes
    kind: str
    strength: float
    fidelity: float


@dataclass
class GenSpec:
    model: Model
    prompt: str
    negative: str = ""
    characters: list[Character] = field(default_factory=list)
    action: str = "generate"
    width: int = 832
    height: int = 1216
    steps: int | None = None
    scale: float | None = None
    sampler: str = "k_euler_ancestral"
    noise_schedule: str = "karras"
    cfg_rescale: float = 0.0
    seed: int | None = None
    n_samples: int = 1
    quality: str = "standard"
    uc_preset: str = "heavy"
    # Send the negative prompt verbatim without UC preset or nsfw, for reproducing other clients' images
    negative_verbatim: bool = False
    # Official rule: add nsfw to the negative prompt unless the prompt has it. Off: never add, and remove it when the prompt has it
    official_nsfw: bool = False
    use_coords: bool = False
    auto_text: bool = True
    # Scale several Vibe strengths down to a sum of 1, the official default
    normalize_vibes: bool = True
    transparent: bool = False
    # False sends premultiplied alpha, the other mode of the official client
    straight_alpha: bool = True
    furry: bool = False
    variety: bool = False
    image: bytes | None = None
    mask: bytes | None = None
    strength: float = 0.7
    noise: float = 0.0
    inpaint_strength: float = 1.0
    # Enhance adds down-weighting; max sends the source size for the server to upscale
    enhance_hint: bool = False
    upscaled_enhance: bool = False
    legacy_uc: bool = False
    vibes: list[Vibe] = field(default_factory=list)
    references: list[PreciseReference] = field(default_factory=list)


@dataclass
class Built:
    body: dict[str, Any]
    seed: int
    prompt: str
    negative: str
    characters: list[Character]
    warnings: list[str]
    model: Model


_PRESET_KEYS = [
    "params_version",
    "width",
    "height",
    "scale",
    "sampler",
    "steps",
    "seed",
    "n_samples",
    "strength",
    "noise",
    "ucPresetId",
    "qualityPresetId",
    "tag_hint_transparent_background",
    "autoSmea",
    "sm",
    "sm_dyn",
    "dynamic_thresholding",
    "controlnet_strength",
    "legacy",
    "add_original_image",
    "cfg_rescale",
]
_PRESET_TAIL_KEYS = [
    "legacy_v3_extend",
    "skip_cfg_above_sigma",
    "use_coords",
    "normalize_reference_strength_multiple",
    "inpaintImg2ImgStrength",
]
_STATE_KEYS = [
    "v4_prompt",
    "v4_negative_prompt",
    "uc",
    "legacy_uc",
    "characterPrompts",
    "upscale",
    "upscaled_enhance",
    "image",
    "mask",
]
_SUBMISSION_KEYS = [
    "straight_alpha",
    "tag_hint_qt",
    "tag_hint_uc_preset",
    "extra_noise_seed",
    "color_correct",
    "img2img",
]
_VIBE_KEYS = ["reference_image_multiple", "reference_strength_multiple"]
_DIRECTOR_KEYS = [
    "director_reference_images",
    "director_reference_descriptions",
    "director_reference_information_extracted",
    "director_reference_strength_values",
    "director_reference_secondary_strength_values",
]
_NORMALIZATION_KEYS = ["negative_prompt", "deliberate_euler_ancestral_bug", "prefer_brownian"]


def parameter_order(model: Model) -> list[str]:
    # V5 sets karras last, as the model had deleted the schedule before
    moved = model.is_v5
    return [
        *_PRESET_KEYS,
        *([] if moved else ["noise_schedule"]),
        *_PRESET_TAIL_KEYS,
        *_STATE_KEYS,
        *_SUBMISSION_KEYS,
        *_VIBE_KEYS,
        *_DIRECTOR_KEYS,
        *_NORMALIZATION_KEYS,
        *(["noise_schedule"] if moved else []),
        "image_format",
        "stream",
    ]


def quality_prompt(spec: GenSpec) -> str:
    model = spec.model
    quality = presets.quality_tags(model, spec.quality) if spec.quality != "none" else None
    return P.apply_suffix(spec.prompt, P.compose_suffix(quality, spec.transparent and model.transparent))


def effective_prompt(spec: GenSpec, centers: list[tuple[float, float]]) -> str:
    """Quality tags, Enhance down-weighting and the furry prefix in the official order; the teXt: block goes last."""
    text = quality_prompt(spec)
    if spec.enhance_hint and spec.model.enhance_hint:
        text = P.add_enhance_hint(text)
    if spec.furry:
        text = P.apply_furry(text)
    if spec.model.auto_text and spec.auto_text:
        chars = [P.AutoTextCharacter(c.prompt, x, y) for c, (x, y) in zip(spec.characters, centers, strict=True)]
        text = P.apply_auto_text(text, chars, spec.use_coords)
    return text


def adds_nsfw(model: Model, uc_preset: str, prompt_with_quality: str) -> bool:
    """Official rule: prepend nsfw to the negative prompt for non-Curated models with a UC preset when the
    prompt (with quality tags) has no nsfw."""
    preset_text = presets.uc_preset_text(model, uc_preset)
    return not model.curated and uc_preset != "none" and bool(preset_text) and P.NSFW not in prompt_with_quality.lower()


def effective_negative(spec: GenSpec, warnings: list[str] | None = None) -> str:
    if spec.negative_verbatim:
        return spec.negative
    model = spec.model
    preset_text = presets.uc_preset_text(model, spec.uc_preset)
    if spec.official_nsfw:
        return P.compose_negative(spec.negative, preset_text, adds_nsfw(model, spec.uc_preset, quality_prompt(spec)))
    text = P.compose_negative(spec.negative, preset_text, nsfw=False)
    if P.contains_nsfw(spec.prompt):
        damaged = P.nsfw_removal_damage(text)
        if damaged and warnings is not None:
            warnings.append(
                f"nsfw is removed from the negative prompt because the prompt has it, which breaks: {', '.join(damaged)}"
            )
        text = P.remove_nsfw(text)
    return text


def resolve_centers(spec: GenSpec) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for i, c in enumerate(spec.characters):
        valid = c.x is not None and c.y is not None and 0 <= c.x <= 1 and 0 <= c.y <= 1
        if valid:
            centers.append((c.x, c.y))
        elif spec.use_coords:
            raise errors.NaiError(errors.BAD_ARGS, f"character {i + 1} needs coordinates within 0-1")
        else:
            centers.append(default_center(centers, spec.model.freeform_position))
    return centers


def check_range(label: str, value: float, limits: tuple[float, float]) -> None:
    low, high = limits
    # NaN fails every comparison, so it is refused here as well
    if not low <= value <= high:
        bounds = "-".join(f"{b:g}" if isinstance(b, float) else str(b) for b in limits)
        raise errors.NaiError(errors.BAD_ARGS, f"{label} must be within {bounds}")


def check_references(model: models.Model, action: str, vibes: int, references: int, outfits: int = 0) -> None:
    """Against the model chosen, not the one it inpaints with: the official client offers Vibes and precise
    references, and prices them, by the model chosen."""
    if vibes and not model.vibe:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED, f"{model.name} does not support Vibe", feature="vibe", model=model.id
        )
    if not references:
        return
    support = model.precise_reference
    if support is None:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED,
            f"{model.name} does not support precise references",
            feature="precise_reference",
            model=model.id,
        )
    if action == "infill" and not support.inpainting:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED,
            f"{model.name} does not take precise references when inpainting",
            feature="precise_reference",
            model=model.id,
        )
    if outfits and not support.outfit:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED,
            f"{model.name} does not support outfit references",
            feature="outfit_reference",
            model=model.id,
        )
    if references > models.MAX_PRECISE_REFERENCES:
        raise errors.NaiError(
            errors.BAD_ARGS, f"at most {models.MAX_PRECISE_REFERENCES} precise references, got {references}"
        )


# Option on the request, what the model must offer for it, and the refusal
_FEATURES = (
    ("variety", "variety_plus", "does not offer Variety+"),
    ("furry", "furry_mode", "has no furry mode"),
    ("transparent", "transparent", "does not support transparent backgrounds"),
)


def check_support(spec: GenSpec) -> None:
    model = spec.model
    _check_legacy_uc(spec)
    # Empty characters are not sent but still take one of the official client's slots
    count = len(spec.characters)
    if count > model.max_characters:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED,
            f"{model.name} allows {model.max_characters} characters, got {count}",
            feature="characters",
            model=model.id,
        )
    for feature, offered, missing in _FEATURES:
        if getattr(spec, feature) and not getattr(model, offered):
            raise errors.NaiError(errors.MODEL_UNSUPPORTED, f"{model.name} {missing}", feature=feature, model=model.id)
    if spec.action == "img2img" and spec.image is None:
        raise errors.NaiError(errors.BAD_ARGS, "img2img needs --image")
    if spec.action == "infill" and (spec.image is None or spec.mask is None):
        raise errors.NaiError(errors.BAD_ARGS, "inpainting needs --image and --mask")
    if spec.quality not in presets.quality_tiers(model):
        raise errors.NaiError(
            errors.BAD_ARGS, f"quality for {model.name} must be one of {presets.quality_tiers(model)}"
        )
    if spec.uc_preset not in presets.uc_presets(model):
        raise errors.NaiError(errors.BAD_ARGS, f"UC presets for {model.name}: {presets.uc_presets(model)}")
    _check_values(spec)


def _check_legacy_uc(spec: GenSpec) -> None:
    model = spec.model
    if spec.legacy_uc and not model.legacy_uc:
        raise errors.NaiError(
            errors.MODEL_UNSUPPORTED,
            f"{model.name} has no legacy UC; only V4 does",
            feature="legacy_uc",
            model=model.id,
        )


def _check_values(spec: GenSpec) -> None:
    if spec.steps is not None:
        check_range("steps", spec.steps, (1, models.MAX_STEPS))
    # The official CFG box has no upper limit
    if spec.scale is not None and not (math.isfinite(spec.scale) and spec.scale >= 0):
        raise errors.NaiError(errors.BAD_ARGS, "CFG must be a number of at least 0")
    check_range("CFG rescale", spec.cfg_rescale, models.CFG_RESCALE_RANGE)
    if spec.seed is not None:
        check_range("seed", spec.seed, (0, models.MAX_SEED))
    if spec.action != "generate":
        check_range("strength", spec.strength, models.STRENGTH_RANGE)
        check_range("noise", spec.noise, models.NOISE_RANGE)
    if spec.action == "infill":
        check_range("inpaint strength", spec.inpaint_strength, models.INPAINT_STRENGTH_RANGE)
    # The official client lets these go past their sliders, so only finite values are required
    weights = [v.strength for v in spec.vibes] + [w for r in spec.references for w in (r.strength, r.fidelity)]
    if not all(math.isfinite(w) for w in weights):
        raise errors.NaiError(errors.BAD_ARGS, "Vibe and precise reference strengths must be finite numbers")
    if spec.width * spec.height > presets.MAX_PIXELS:
        raise errors.NaiError(
            errors.BAD_ARGS, f"{spec.width}x{spec.height} exceeds the NovelAI limit of {presets.MAX_PIXELS} pixels"
        )
    limit = presets.max_samples(spec.width, spec.height)
    if not 1 <= spec.n_samples <= limit:
        raise errors.NaiError(errors.BAD_ARGS, f"{spec.width}x{spec.height} allows 1 to {limit} images per request")


def build(spec: GenSpec) -> Built:
    outfits = sum(r.kind == OUTFIT_KIND for r in spec.references)
    check_references(spec.model, spec.action, len(spec.vibes), len(spec.references), outfits)
    # Each side goes on the grid just before sending, as in the official client; the image keeps its size
    spec = replace(spec, width=presets.nearest_grid(spec.width), height=presets.nearest_grid(spec.height))
    spec, warnings = _switch_to_inpainting(spec)
    check_support(spec)
    spec, fixed_warnings = _apply_fixed(spec)
    warnings += fixed_warnings
    model = spec.model
    seed = spec.seed if spec.seed is not None else random_seed()
    steps = spec.steps if spec.steps is not None else model.default_steps
    centers = resolve_centers(spec)
    prompt_text = effective_prompt(spec, centers)
    negative_text = effective_negative(spec, warnings)
    characters = _sent_characters(spec, centers, warnings)
    p = {
        **_settings(spec, seed, steps, negative_text, warnings),
        **_captions(spec, prompt_text, negative_text, characters),
        **_images(spec, seed),
        **_references(spec, warnings),
    }
    body = {
        "input": prompt_text,
        "model": model.inpainting_id if spec.action == "infill" else model.id,
        "action": spec.action,
        "parameters": _reorder(p, parameter_order(model)),
        "use_new_shared_trial": True,
    }
    return Built(
        body=body,
        seed=seed,
        prompt=prompt_text,
        negative=negative_text,
        characters=[c for c, _ in characters],
        warnings=warnings,
        model=model,
    )


def _reorder(source: dict[str, Any], order: list[str]) -> dict[str, Any]:
    out = {k: source[k] for k in order if k in source}
    out.update({k: v for k, v in source.items() if k not in out})
    return out


def _switch_to_inpainting(spec: GenSpec) -> tuple[GenSpec, list[str]]:
    model = models.for_action(spec.model, spec.action)
    if model is spec.model:
        return spec, []
    quality = spec.quality if spec.quality in presets.quality_tiers(model) else "standard"
    switched = replace(spec, model=model, quality=quality, uc_preset=presets.carry_uc_preset(spec.uc_preset, model))
    return switched, [
        f"{spec.model.name} inpaints with {spec.model.inpainting_id}; quality tags, UC presets and pricing follow {model.name}"
    ]


def _apply_fixed(spec: GenSpec) -> tuple[GenSpec, list[str]]:
    fixed = spec.model.fixed
    if fixed is None:
        return spec, []
    warnings = []
    if spec.negative or any(c.negative for c in spec.characters):
        warnings.append(f"{spec.model.name} uses no custom negative prompt; ignored")
    return (
        replace(
            spec,
            steps=fixed.steps,
            sampler=fixed.sampler,
            uc_preset=fixed.uc_preset,
            negative="",
            negative_verbatim=False,
            characters=[Character(c.prompt, "", c.x, c.y) for c in spec.characters],
        ),
        warnings,
    )


def _sent_characters(
    spec: GenSpec, centers: list[tuple[float, float]], warnings: list[str]
) -> list[tuple[Character, tuple[float, float]]]:
    """The official client drops characters without a prompt after placing them, so the others keep their
    places."""
    sent = []
    for i, (c, center) in enumerate(zip(spec.characters, centers, strict=True), start=1):
        if c.prompt:
            sent.append((c, center))
        else:
            warnings.append(f"character {i} has no prompt and is not sent, as in the official client")
    return sent


def _settings(spec: GenSpec, seed: int, steps: int, negative_text: str, warnings: list[str]) -> dict[str, Any]:
    model = spec.model
    p: dict[str, Any] = {
        "params_version": PARAMS_VERSION,
        "width": spec.width,
        "height": spec.height,
        "scale": model.default_scale if spec.scale is None else spec.scale,
        "sampler": spec.sampler,
        "steps": steps,
        "n_samples": spec.n_samples,
        "ucPresetId": presets.UC_PRESET_ID[spec.uc_preset],
        "qualityPresetId": spec.quality,
        "autoSmea": False,
        "dynamic_thresholding": False,
        "controlnet_strength": 1,
        "legacy": False,
        "add_original_image": spec.action != "infill",
        "inpaintImg2ImgStrength": spec.inpaint_strength,
        "seed": seed,
        "image_format": "png",
    }
    noise_schedule: str | None = spec.noise_schedule
    if not model.noise_schedule:
        if spec.noise_schedule != "karras":
            warnings.append(f"{model.name} always uses the karras noise schedule")
        noise_schedule = "karras"
    if noise_schedule is not None:
        p["noise_schedule"] = noise_schedule
    if model.cfg_rescale:
        p["cfg_rescale"] = spec.cfg_rescale
    if negative_text:
        p["negative_prompt"] = negative_text
    else:
        p["uc"] = ""
    # Decided on the schedule chosen, before V5 replaces it with karras, as the official client orders it
    if spec.sampler == "k_euler_ancestral":
        p["deliberate_euler_ancestral_bug"] = False
        p["prefer_brownian"] = True
    if model.variety_plus:
        p["skip_cfg_above_sigma"] = (
            model.variety_sigma * math.sqrt(4.0 * (spec.width // 8) * (spec.height // 8) / _VARIETY_REFERENCE_LATENTS)
            if spec.variety
            else None
        )
    if model.transparent:
        p["straight_alpha"] = spec.straight_alpha
        if spec.transparent:
            p["tag_hint_transparent_background"] = True
    p["tag_hint_qt"] = presets.QUALITY_HINT[spec.quality]
    # A verbatim negative prompt carries no UC preset hint, except on models with fixed settings
    if not spec.negative_verbatim or model.fixed:
        p["tag_hint_uc_preset"] = presets.UC_PRESET_HINT[spec.uc_preset]
    p["use_coords"] = spec.use_coords
    p["legacy_v3_extend"] = False
    p["legacy_uc"] = spec.legacy_uc
    p["normalize_reference_strength_multiple"] = spec.normalize_vibes
    return p


def _captions(
    spec: GenSpec,
    prompt_text: str,
    negative_text: str,
    characters: list[tuple[Character, tuple[float, float]]],
) -> dict[str, Any]:
    char_captions, neg_char_captions, character_prompts = [], [], []
    for c, (x, y) in characters:
        cx, cy = (x, y) if spec.model.freeform_position else (snap(x), snap(y))
        char_captions.append({"char_caption": c.prompt, "centers": [{"x": cx, "y": cy}]})
        neg_char_captions.append({"char_caption": c.negative, "centers": [{"x": cx, "y": cy}]})
        character_prompts.append({"prompt": c.prompt, "uc": c.negative, "center": {"x": x, "y": y}, "enabled": True})
    return {
        "v4_prompt": {
            "caption": {"base_caption": prompt_text, "char_captions": char_captions},
            "use_coords": spec.use_coords,
            "use_order": True,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": negative_text, "char_captions": neg_char_captions},
            "legacy_uc": spec.legacy_uc,
        },
        "characterPrompts": character_prompts,
    }


def _images(spec: GenSpec, seed: int) -> dict[str, Any]:
    p: dict[str, Any] = {}
    if spec.action in ("img2img", "infill"):
        p["image"] = base64.b64encode(spec.image).decode()
    if spec.action == "img2img":
        p["strength"] = spec.strength
        p["noise"] = spec.noise
        p["color_correct"] = False
        if spec.upscaled_enhance:
            p["upscaled_enhance"] = True
    if spec.action == "infill":
        p["mask"] = base64.b64encode(spec.mask).decode()
        p["strength"] = spec.strength
        p["noise"] = spec.noise
        if spec.inpaint_strength != 1.0:
            p["img2img"] = {"strength": spec.inpaint_strength, "color_correct": True}
    if "image" in p:
        # Fixed whenever a base image is sent; without it the same seed does not reproduce
        p["extra_noise_seed"] = seed - 1
    return p


def _references(spec: GenSpec, warnings: list[str]) -> dict[str, Any]:
    if spec.references:
        if spec.vibes:
            warnings.append("Vibes are not sent with precise references; ignored")
        support = spec.model.precise_reference
        assert support is not None
        kinds = [r.kind if support.outfit else _PLAIN_KINDS.get(r.kind, r.kind) for r in spec.references]
        if kinds != [r.kind for r in spec.references]:
            warnings.append(f"{spec.model.name} has no outfit preference; those references are sent without it")
        # Without fidelity the official client sends every reference at fidelity 1
        if not support.fidelity and any(r.fidelity != 1.0 for r in spec.references):
            warnings.append(f"{spec.model.name} has no precise reference fidelity; ignored")
        return {
            "director_reference_images": [base64.b64encode(r.image).decode() for r in spec.references],
            "director_reference_descriptions": [
                {"caption": {"base_caption": kind, "char_captions": []}, "legacy_uc": False} for kind in kinds
            ],
            "director_reference_information_extracted": [1 for _ in spec.references],
            "director_reference_strength_values": [r.strength for r in spec.references],
            "director_reference_secondary_strength_values": [
                1.0 - r.fidelity if support.fidelity else 0.0 for r in spec.references
            ],
        }
    if vibes_used(spec):
        strengths = [v.strength for v in spec.vibes]
        return {
            "reference_image_multiple": [v.encoding for v in spec.vibes],
            "reference_strength_multiple": normalize_strengths(strengths) if spec.normalize_vibes else strengths,
        }
    if spec.vibes:
        warnings.append("Vibes cannot be used for inpainting; ignored")
    return {}


def vibes_sent(action: str, references: int) -> bool:
    """Vibes are neither sent nor billed with precise references or inpainting."""
    return not references and action != "infill"


def vibes_used(spec: GenSpec) -> bool:
    return bool(spec.vibes) and vibes_sent(spec.action, len(spec.references))


def normalize_strengths(strengths: list[float]) -> list[float]:
    if len(strengths) < 2:
        return strengths
    total = sum(abs(s) for s in strengths)
    if total <= 1.0:
        return strengths
    return [s / total for s in strengths]
