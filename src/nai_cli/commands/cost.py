"""cost: the Anlas a request would cost, priced against the account when a token is set."""

from dataclasses import dataclass
from typing import Any

from nai_cli import cost, errors, models, presets
from nai_cli import request as R
from nai_cli.charge import cost_json
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class CostSettings:
    model: models.Model
    action: str
    size: tuple[int, int]
    steps: int
    n: int
    strength: float
    vibes: int
    encode: int
    refs: int
    warnings: tuple[str, ...]


def resolve(opts: dict[str, Any], ctx: Context) -> CostSettings:
    d = ctx.config.defaults
    action = "infill" if opts.get("inpaint") else "generate"
    picked = models.resolve(opts.get("model") or d.get("model"))
    refs, vibes, encode = opts.get("refs") or 0, opts.get("vibes") or 0, opts.get("encode") or 0
    if vibes < 0 or refs < 0 or not 0 <= encode <= vibes:
        raise errors.NaiError(
            errors.BAD_ARGS, "--vibes and --refs cannot be negative, and --encode must be within 0 and --vibes"
        )
    effort = models.with_effort(picked, opts.get("effort"), d.get("effort"))
    chosen = models.for_references(effort, refs > 0)
    R.check_references(chosen, action, vibes, refs)
    model = models.for_action(chosen, action)
    w, h = presets.resolve_size(opts.get("size") or d.get("size"))
    n = opts["n"] if opts.get("n") is not None else d.get("n", 1)
    limit = presets.max_samples(w, h)
    if not 1 <= n <= limit:
        raise errors.NaiError(errors.BAD_ARGS, f"{w}x{h} allows 1 to {limit} images per request")
    steps = opts["steps"] if opts.get("steps") is not None else d.get("steps", model.default_steps)
    R.check_range("steps", steps, (1, models.MAX_STEPS))
    strength = opts["strength"] if opts.get("strength") is not None else 1.0
    # img2img strength or inpaint strength, whichever is the wider
    R.check_range("strength", strength, models.INPAINT_STRENGTH_RANGE)
    if not R.vibes_sent(action, refs):
        vibes = encode = 0
    warnings = (models.HIGH_EFFORT_FOR_REFERENCES,) if chosen is not effort else ()
    return CostSettings(model, action, (w, h), steps, n, strength, vibes, encode, refs, warnings)


def run(s: CostSettings, ctx: Context) -> dict[str, Any]:
    warnings = list(s.warnings)
    account = ctx.dry_account(warnings)
    est = cost.generation(
        s.model,
        *s.size,
        s.steps,
        n_samples=s.n,
        strength=s.strength,
        opus=account.opus if account else True,
        quota_exhausted=account.quota_negative if account else False,
        precise_references=s.refs,
        vibes=s.vibes,
        vibes_to_encode=s.encode,
    )
    cost.require_valid(est)
    request_model = s.model.inpainting_id if s.action == "infill" else s.model.id
    out = {"ok": True, "model": request_model, "size": "{}x{}".format(*s.size), "cost": cost_json(est, account)}
    return {**out, "warnings": warnings} if warnings else out
