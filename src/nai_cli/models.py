"""The models, what each supports and starts from, and the samplers, as the official client lists them."""

from dataclasses import dataclass, replace

from nai_cli import errors


@dataclass(frozen=True)
class FixedSettings:
    steps: int
    sampler: str
    uc_preset: str


@dataclass(frozen=True)
class References:
    """How a model takes precise references, as the official capability table has it."""

    # Sent when inpainting; without it they are neither sent nor priced
    inpainting: bool
    # Cropped to the closest of 97 size buckets rather than fitted into one of three canvases
    buckets: bool
    fidelity: bool
    outfit: bool


V45_REFERENCES = References(inpainting=True, buckets=False, fidelity=True, outfit=False)
V5_REFERENCES = References(inpainting=False, buckets=True, fidelity=False, outfit=True)


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    # As the official client abbreviates it
    short_name: str
    family: str
    inpainting_id: str
    default_scale: float
    default_steps: int
    token_limit: int
    # Token counts use the official tokenizer files: qwen (Qwen3.5 BPE) or t5 (T5 Unigram)
    tokenizer: str
    max_characters: int
    vibe: bool
    precise_reference: References | None
    auto_text: bool
    transparent: bool
    # Curated models never get nsfw added to the negative prompt
    curated: bool = False
    # Without free placement, character coordinates snap to a 5x5 grid
    freeform_position: bool = False
    noise_schedule: bool = True
    cfg_rescale: bool = True
    variety_plus: bool = False
    # Variety+ skips CFG above this sigma, scaled by the image size
    variety_sigma: float = 58.0
    # Offers the legacy UC switch of V4
    legacy_uc: bool = False
    # What the official client writes in a new character's negative prompt
    character_negative: str = ""
    # Inpaints with another model's inpainting model, so that id does not lead back here
    shared_inpainting: bool = False
    anlas_multiplier: float = 1.0
    medium_effort: bool = False
    opus_quota: bool = False
    # Enhance appends -2::upscaled, blurry:: to the prompt
    enhance_hint: bool = False
    # Enhance at max sends the source size with upscaled_enhance; the server upscales to the pixel limit
    max_enhance: bool = False
    # Furry mode starts the prompt with fur dataset
    furry_mode: bool = True
    # Characters of text: the official documentation says the model renders reliably
    text_limit: int | None = None
    fixed: FixedSettings | None = None
    # The presets the official client starts this model with
    default_quality: str = "standard"
    default_uc_preset: str = "heavy"

    @property
    def is_v5(self) -> bool:
        return self.family == "v5"


# Community defaults: CFG 4 and 28 steps, the Opus free maximum. The official client uses 7 and 23
_V5_COMMON = dict(
    family="v5",
    default_scale=4.0,
    default_steps=28,
    tokenizer="qwen",
    max_characters=32,
    vibe=False,
    precise_reference=V5_REFERENCES,
    auto_text=True,
    transparent=True,
    freeform_position=True,
    noise_schedule=False,
    anlas_multiplier=1.5,
    opus_quota=True,
    enhance_hint=True,
    max_enhance=True,
)

_V45_COMMON = dict(
    family="v4.5",
    default_scale=5.0,
    default_steps=23,
    token_limit=512,
    tokenizer="t5",
    max_characters=6,
    vibe=True,
    precise_reference=V45_REFERENCES,
    auto_text=False,
    transparent=False,
    variety_plus=True,
    enhance_hint=True,
    text_limit=118,
)

# Not recommended by the official client any more, but still offered
_V4_COMMON = dict(
    _V45_COMMON,
    family="v4",
    default_scale=5.5,
    precise_reference=None,
    enhance_hint=False,
    text_limit=None,
    variety_sigma=19.0,
    legacy_uc=True,
    character_negative="lowres, aliasing, ",
)

MODELS: dict[str, Model] = {
    m.id: m
    for m in [
        Model(
            id="nai-diffusion-5-full",
            name="NAI Diffusion V5 Full",
            short_name="V5 Full",
            inpainting_id="nai-diffusion-5-full-inpainting",
            token_limit=1471,
            text_limit=750,
            **_V5_COMMON,
        ),
        Model(
            id="nai-diffusion-5-curated",
            name="NAI Diffusion V5 Curated",
            short_name="V5 Curated",
            # V5 Curated has no inpainting model of its own and uses V4.5 Curated's
            inpainting_id="nai-diffusion-4-5-curated-inpainting",
            shared_inpainting=True,
            token_limit=703,
            text_limit=374,
            curated=True,
            **_V5_COMMON,
        ),
        Model(
            id="nai-diffusion-5-full-medium",
            name="NAI Diffusion V5 Full (Medium Effort)",
            short_name="V5 Full Medium",
            inpainting_id="nai-diffusion-5-full-medium-inpainting",
            token_limit=1471,
            cfg_rescale=False,
            medium_effort=True,
            fixed=FixedSettings(steps=14, sampler="k_euler_ancestral", uc_preset="heavy"),
            **_V5_COMMON,
        ),
        Model(
            id="nai-diffusion-4-5-full",
            name="NAI Diffusion V4.5 Full",
            short_name="V4.5 Full",
            inpainting_id="nai-diffusion-4-5-full-inpainting",
            **_V45_COMMON,
        ),
        Model(
            id="nai-diffusion-4-5-curated",
            name="NAI Diffusion V4.5 Curated",
            short_name="V4.5 Curated",
            inpainting_id="nai-diffusion-4-5-curated-inpainting",
            curated=True,
            **_V45_COMMON,
        ),
        Model(
            id="nai-diffusion-4-full",
            name="NAI Diffusion V4 Full",
            short_name="V4 Full",
            inpainting_id="nai-diffusion-4-full-inpainting",
            **_V4_COMMON,
        ),
        Model(
            id="nai-diffusion-4-curated-preview",
            name="NAI Diffusion V4 Curated",
            short_name="V4 Curated",
            inpainting_id="nai-diffusion-4-curated-inpainting",
            curated=True,
            **_V4_COMMON,
        ),
    ]
}
_BY_INPAINTING = {m.inpainting_id: m.id for m in MODELS.values() if not m.shared_inpainting}

ALIASES = {
    "v5": "nai-diffusion-5-full",
    "v5-full": "nai-diffusion-5-full",
    "v5-curated": "nai-diffusion-5-curated",
    "v5-medium": "nai-diffusion-5-full-medium",
    "v4.5": "nai-diffusion-4-5-full",
    "v4.5-full": "nai-diffusion-4-5-full",
    "v4.5-curated": "nai-diffusion-4-5-curated",
    "v4": "nai-diffusion-4-full",
    "v4-full": "nai-diffusion-4-full",
    "v4-curated": "nai-diffusion-4-curated-preview",
}

DEFAULT_MODEL = "nai-diffusion-5-full"
# Medium effort fixes its settings, so it is left as the official client has it
_HIDDEN_OPTIONS = ("nai-diffusion-5-full", "nai-diffusion-5-curated")
EFFORTS = ("high", "medium")
_EFFORT_MODELS = {"high": "nai-diffusion-5-full", "medium": "nai-diffusion-5-full-medium"}


MAX_STEPS = 50
MAX_PRECISE_REFERENCES = 16
# Ranges of the official sliders
CFG_RESCALE_RANGE = (0.0, 1.0)
STRENGTH_RANGE = (0.01, 0.99)
NOISE_RANGE = (0.0, 0.99)
INPAINT_STRENGTH_RANGE = (0.01, 1.0)
INFORMATION_EXTRACTED_RANGE = (0.01, 1.0)
# The official seed field takes up to 10 digits
MAX_SEED = 9_999_999_999

# The official client's list for V4 and V5; it offers DDIM only for older models, and the server fails it here
SAMPLERS = [
    "k_euler_ancestral",
    "k_euler",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_2m_sde",
    "k_dpmpp_sde",
]

SAMPLER_ALIASES = {
    "euler_a": "k_euler_ancestral",
    "euler-a": "k_euler_ancestral",
    "euler ancestral": "k_euler_ancestral",
    "euler": "k_euler",
    "dpm++ 2s ancestral": "k_dpmpp_2s_ancestral",
    "dpm++ 2m": "k_dpmpp_2m",
    "dpmpp_2m": "k_dpmpp_2m",
    "dpm++ 2m sde": "k_dpmpp_2m_sde",
    "dpm++ sde": "k_dpmpp_sde",
}

NOISE_SCHEDULES = ["karras", "exponential", "polyexponential"]
# Older models' samplers kept in image settings, which the official client sends as Euler Ancestral
_RETIRED_SAMPLERS = {"ddim", "ddim_v3"}


def resolve(name: str | None) -> Model:
    """A model by id or alias."""
    key = (name or DEFAULT_MODEL).strip().lower()
    key = ALIASES.get(key, key)
    model = MODELS.get(_BY_INPAINTING.get(key, key))
    if model is None:
        choices = sorted(ALIASES) + sorted(MODELS)
        raise errors.NaiError(errors.BAD_ARGS, f"unknown model: {name}", choices=choices)
    return model


def with_effort(model: Model, typed: str | None, configured: str | None) -> Model:
    """V5 Full at the effort given with --effort, which only V5 Full has; else at the config's, which a model
    chosen at medium effort keeps."""
    if configured is not None and configured not in EFFORTS:
        raise errors.NaiError(errors.CONFIG_MISSING, f"[defaults] effort is {configured!r}", choices=list(EFFORTS))
    effort = typed or (configured if model.id == _EFFORT_MODELS["high"] else None)
    if effort is None:
        return model
    if model.id not in _EFFORT_MODELS.values():
        raise errors.NaiError(errors.BAD_ARGS, f"{model.name} has no effort levels; only V5 Full does")
    return MODELS[_EFFORT_MODELS[effort]]


def effort_of(model_id: str) -> str | None:
    """The effort a V5 Full model id stands for."""
    return next((effort for effort, found in _EFFORT_MODELS.items() if found == model_id), None)


HIGH_EFFORT_FOR_REFERENCES = "precise references need high effort, so V5 Full runs instead of V5 Full Medium"


def for_references(model: Model, references: bool) -> Model:
    """V5 Full in place of medium effort, which the official client never sends precise references with."""
    if not references or not model.medium_effort:
        return model
    return MODELS[_EFFORT_MODELS["high"]]


def with_hidden_options(model: Model, enabled: bool = True) -> Model:
    """V5 Full and V5 Curated with the noise schedule and Variety+ the official client hides for V5."""
    if not enabled or model.id not in _HIDDEN_OPTIONS:
        return model
    return replace(model, noise_schedule=True, variety_plus=True)


def for_action(model: Model, action: str) -> Model:
    """For inpainting, every rule (quality tags, UC presets, features, pricing) follows the inpainting model."""
    if action != "infill" or not model.shared_inpainting:
        return model
    return resolve(model.inpainting_id)


def default_noise_schedule(sampler: str) -> str:
    """The schedule the official client gives a sampler picked without one chosen for it before."""
    return "exponential" if sampler == "k_dpmpp_2m" else "karras"


def imported_sampler(value: object) -> str | None:
    """A sampler read from an image's settings as V4 and later send it; one it does not know is left out."""
    if not isinstance(value, str):
        return None
    if value in _RETIRED_SAMPLERS:
        return "k_euler_ancestral"
    return value if value in SAMPLERS else None


def resolve_sampler(name: str) -> str:
    key = name.strip().lower()
    key = SAMPLER_ALIASES.get(key, key)
    if not key.startswith("k_"):
        key = "k_" + key
    if key not in SAMPLERS:
        raise errors.NaiError(errors.BAD_ARGS, f"unknown sampler: {name}", choices=SAMPLERS)
    return key
