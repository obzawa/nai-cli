"""FILE[:A[:B]] values of --vibe, --ref and vibe sources, and the files they name."""

import math
from dataclasses import dataclass
from pathlib import Path

from nai_cli import errors, images, models, values, vibe
from nai_cli import request as R

# Names on the command line, then the official ones; both keeps the outfit and character does not, as the
# official client starts them
PRECISE_KINDS = {
    "character": "character",
    "character+outfit": "character_mainoutfit",
    "style": "style",
    "both": "character&style",
    "both-any-outfit": "character&style_anyoutfit",
    "outfit": R.OUTFIT_KIND,
    **{kind: kind for kind in ("character&style", "character_mainoutfit", "character&style_anyoutfit", "costume")},
}


@dataclass(frozen=True)
class VibeRef:
    path: Path
    strength: float | None
    information_extracted: float | None


@dataclass(frozen=True)
class PreciseRef:
    path: Path
    kind: str
    strength: float
    fidelity: float


def vibe_ref(value: str, strength: float | None = None, info: float | None = None) -> VibeRef:
    """Fields left empty take strength and info, then what the file stores."""
    path, (s, i) = _fields(value, ["STRENGTH", "INFO"])
    return VibeRef(path, _number(s, strength, value), _number(i, info, value))


def precise_ref(value: str) -> PreciseRef:
    path, (kind_name, strength, fidelity) = _fields(value, ["KIND", "STRENGTH", "FIDELITY"])
    # Character and style by default, as the official client adds a reference
    kind = PRECISE_KINDS.get(kind_name or "both")
    if kind is None:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"precise reference kind must be both, both-any-outfit, character, character+outfit, style or outfit: "
            f"{value}",
        )
    return PreciseRef(path, kind, _number(strength, 1.0, value), _number(fidelity, 1.0, value))


def load_vibes(refs: tuple[VibeRef, ...], model: str) -> list[vibe.VibeSource]:
    return [source for ref in refs for source in vibe.load(ref.path, ref.strength, ref.information_extracted, model)]


def load_precise(refs: tuple[PreciseRef, ...], model: models.Model) -> list[R.PreciseReference]:
    return [R.PreciseReference(_reference_image(ref.path, model), ref.kind, ref.strength, ref.fidelity) for ref in refs]


def _reference_image(path: Path, model: models.Model) -> bytes:
    support = model.precise_reference
    if support is not None and support.buckets:
        return images.bucketed_reference_image(path)
    return images.precise_reference_image(path, model.transparent)


def _split(value: str) -> tuple[Path, list[str]]:
    # A drive letter's colon is not a separator
    head = ""
    if len(value) > 2 and value[1] == ":" and value[2] in "\\/":
        head, value = value[:2], value[2:]
    parts = value.split(":")
    return values.path(head + parts[0]), parts[1:]


def _fields(value: str, names: list[str]) -> tuple[Path, list[str]]:
    """FILE:A:B with at most len(names) fields after the file; missing fields are empty."""
    path, rest = _split(value)
    if len(rest) > len(names):
        usage = "".join(f"[:{name}" for name in names) + "]" * len(names)
        raise errors.NaiError(errors.BAD_ARGS, f"expected FILE{usage}: {value}")
    return path, rest + [""] * (len(names) - len(rest))


def _number(field: str, default: float | None, value: str) -> float | None:
    """An empty field takes the default."""
    if not field:
        return default
    try:
        number = float(field)
    except ValueError:
        raise errors.NaiError(errors.BAD_ARGS, f"not a number: {field!r}: {value}") from None
    if not math.isfinite(number):
        raise errors.NaiError(errors.BAD_ARGS, f"not a finite number: {field!r}: {value}")
    return number
