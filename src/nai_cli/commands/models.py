"""models: the models and what each supports."""

from dataclasses import dataclass
from typing import Any

from nai_cli import models, presets
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class ModelsSettings:
    pass


def resolve(opts: dict[str, Any], ctx: Context) -> ModelsSettings:
    return ModelsSettings()


def run(s: ModelsSettings, ctx: Context) -> dict[str, Any]:
    aliases: dict[str, list[str]] = {}
    for alias, target in models.ALIASES.items():
        aliases.setdefault(target, []).append(alias)
    d = ctx.config.defaults
    default = models.with_effort(models.resolve(d.get("model")), None, d.get("effort"))
    offered = (models.with_hidden_options(m, bool(d.get("v5_hidden_options"))) for m in models.MODELS.values())
    return {
        "ok": True,
        "default": default.id,
        "models": [_model_json(m, aliases.get(m.id, [])) for m in offered],
    }


def _model_json(model: models.Model, aliases: list[str]) -> dict[str, Any]:
    fixed = model.fixed
    references = model.precise_reference
    return {
        "id": model.id,
        "name": model.name,
        "aliases": aliases,
        "family": model.family,
        "inpainting_model": model.inpainting_id,
        "token_limit": model.token_limit,
        "tokenizer": model.tokenizer,
        "text_limit": model.text_limit,
        "max_characters": model.max_characters,
        "vibe": model.vibe,
        "precise_reference": references is not None,
        "reference_inpainting": references is not None and references.inpainting,
        "reference_fidelity": references is not None and references.fidelity,
        "outfit_reference": references is not None and references.outfit,
        "transparent": model.transparent,
        "furry": model.furry_mode,
        "auto_text": model.auto_text,
        "variety": model.variety_plus,
        "noise_schedule": model.noise_schedule,
        "cfg_rescale": model.cfg_rescale,
        "legacy_uc": model.legacy_uc,
        "effort": models.effort_of(model.id),
        "quality": presets.quality_tiers(model),
        "default_quality": model.default_quality,
        "uc_presets": presets.uc_presets(model),
        "default_uc_preset": model.default_uc_preset,
        "default_steps": model.default_steps,
        "default_cfg": model.default_scale,
        "fixed": {"steps": fixed.steps, "sampler": fixed.sampler, "uc_preset": fixed.uc_preset} if fixed else None,
    }
