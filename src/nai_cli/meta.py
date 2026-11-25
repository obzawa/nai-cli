"""Generation settings read from NovelAI images: PNG text, EXIF UserComment or stealth data in the alpha."""

import io
import json
import math
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image

from nai_cli import decode, errors, presets, values
from nai_cli import prompt as P
from nai_cli import request as R
from nai_cli.models import CFG_RESCALE_RANGE, MAX_SEED, MAX_STEPS, MODELS, Model, resolve

_STEALTH_MAGIC = {b"stealth_pnginfo": False, decode.STEALTH_COMPRESSED: True}
_EXIF_IFD = 0x8769
T = TypeVar("T")
_USER_COMMENT = 0x9286

_QUALITY_FROM_HINT = {0: "none", 1: "standard", 3: "light"}
_UC_FROM_HINT = {2: "heavy", 3: "light", 4: "human", 5: "furry", 0: "none"}
_UC_FROM_ID = {v: k for k, v in presets.UC_PRESET_ID.items()}
# Models are identified by the hash in Source; unlisted V5, V4.5 and V4 hashes are the Curated model of each
_V5_SOURCES = {
    "NovelAI Diffusion V5 657484A5": "nai-diffusion-5-full",
    "NovelAI Diffusion V5 0ADF9AB7": "nai-diffusion-5-full",
    "NovelAI Diffusion V5 DB276663": "nai-diffusion-5-curated",
    "NovelAI Diffusion V5 93F4BD30": "nai-diffusion-5-full-medium",
    "NovelAI Diffusion V5 70AB5786": "nai-diffusion-5-full-medium",
}
_V4_SOURCES = {
    "DiffusionModelMetaName.NAIv4next 4BDE2A90": "nai-diffusion-4-5-full",
    "NovelAI Diffusion V4.5 4BDE2A90": "nai-diffusion-4-5-full",
    "NovelAI Diffusion V4.5 1229B44F": "nai-diffusion-4-5-full",
    "NovelAI Diffusion V4.5 B9F340FD": "nai-diffusion-4-5-full",
    "NovelAI Diffusion V4.5 F3D95188": "nai-diffusion-4-5-full",
    "DiffusionModelMetaName.NAIv4next C02D4F98": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4.5 C02D4F98": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4.5 5AB81C7C": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4.5 B5A2A797": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4 5AB81C7C": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4 B5A2A797": "nai-diffusion-4-5-curated",
    "NovelAI Diffusion V4 37442FCA": "nai-diffusion-4-full",
    "NovelAI Diffusion V4 4F49EC75": "nai-diffusion-4-full",
    "NovelAI Diffusion V4 CA4B7203": "nai-diffusion-4-full",
    "NovelAI Diffusion V4 79F47848": "nai-diffusion-4-full",
    "NovelAI Diffusion V4 F6302A9D": "nai-diffusion-4-full",
    "NovelAI Diffusion V4 7ABFFA2A": "nai-diffusion-4-curated-preview",
    "NovelAI Diffusion V4 C1CCBA86": "nai-diffusion-4-curated-preview",
    "NovelAI Diffusion V4 770A9E12": "nai-diffusion-4-curated-preview",
}


def png_text_entries(data: bytes) -> list[tuple[str, str]]:
    """Every tEXt, iTXt and zTXt chunk as (keyword, text), in file order."""
    out: list[tuple[str, str]] = []
    for kind, body, _, _ in decode.png_chunks(data):
        try:
            if kind == b"tEXt":
                key, value = body.split(b"\0", 1)
                out.append((key.decode("latin1"), decode.latin1_text(value)))
            elif kind == b"iTXt":
                key, rest = body.split(b"\0", 1)
                compressed = rest[0]
                rest = rest[2:]
                _, rest = rest.split(b"\0", 1)
                _, text = rest.split(b"\0", 1)
                text = decode.inflate(text) if compressed else text
                out.append((key.decode("latin1"), text.decode("utf-8", "replace")))
            elif kind == b"zTXt":
                key, rest = body.split(b"\0", 1)
                out.append((key.decode("latin1"), decode.latin1_text(decode.inflate(rest[1:]))))
        except (ValueError, IndexError):
            continue
    return out


def png_text_chunks(data: bytes) -> dict[str, str]:
    """By keyword; a keyword repeated keeps its last text."""
    return dict(png_text_entries(data))


def stealth_text(img: Image.Image) -> dict[str, str]:
    """Read alpha LSBs column-major: magic, 32-bit length in bits, payload."""
    if "A" not in img.getbands() or img.width * img.height > decode.STEALTH_MAX_PIXELS:
        return {}
    alpha = img.getchannel("A")
    width, height = alpha.size
    pixels = alpha.load()

    def bits():
        for x in range(width):
            for y in range(height):
                yield pixels[x, y] & 1

    stream = bits()

    def read_bytes(n: int) -> bytes | None:
        out = bytearray()
        for _ in range(n):
            byte = 0
            for _ in range(8):
                bit = next(stream, None)
                if bit is None:
                    return None
                byte = (byte << 1) | bit
            out.append(byte)
        return bytes(out)

    magic = read_bytes(len(b"stealth_pnginfo"))
    if magic not in _STEALTH_MAGIC:
        return {}
    length_bytes = read_bytes(4)
    if length_bytes is None:
        return {}
    (bit_length,) = struct.unpack(">I", length_bytes)
    if bit_length <= 0 or bit_length > width * height - (len(magic) + 4) * 8:
        return {}
    payload = read_bytes(-(-bit_length // 8))
    if payload is None:
        return {}
    try:
        text = decode.inflate(payload, decode.GZIP) if _STEALTH_MAGIC[magic] else payload
        parsed = values.parse_json(text.decode("utf-8"))
    except (OSError, ValueError, EOFError, zlib.error):
        return {}
    if isinstance(parsed, dict):
        return {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in parsed.items()}
    return {}


def exif_text(img: Image.Image) -> dict[str, str]:
    """Settings in the EXIF UserComment, where NovelAI's WebP images carry them, decoded as the official
    importer does: an 8-byte charset header, UTF-16 for UNICODE, and trailing NULs dropped."""
    try:
        raw = img.getexif().get_ifd(_EXIF_IFD).get(_USER_COMMENT)
    except (OSError, ValueError, SyntaxError, struct.error, KeyError, TypeError):
        return {}
    if not isinstance(raw, bytes) or len(raw) < 8:
        return {}
    head, body = raw[:8], raw[8:]
    if head == b"UNICODE\0":
        # The official decoder follows a byte order mark and otherwise reads big-endian
        text = body.decode("utf-16" if body[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-16-be", "replace")
    else:
        text = (body if head in (b"ASCII\0\0\0", b"\0" * 8) else raw).decode("utf-8", "replace")
    try:
        parsed = values.parse_json(text.rstrip("\0"))
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        k: v.replace("\0", "") if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in parsed.items()
    }


def read_raw(path: str | Path) -> dict[str, str]:
    path = Path(path)
    return raw_of(_bytes(path), path.name)


def raw_of(data: bytes, name: str) -> dict[str, str]:
    """The settings an image's bytes keep, from PNG text, EXIF or the alpha channel; bytes that are no image are
    refused, not taken for an image without settings."""
    chunks = png_text_chunks(data)
    if "Comment" in chunks:
        return chunks
    try:
        with Image.open(io.BytesIO(data)) as img:
            found = exif_text(img)
            if "Comment" not in found:
                img.load()
                found = stealth_text(img)
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {name}: {e}") from e
    if "Comment" in found:
        return found
    raise errors.NaiError(errors.NO_METADATA, f"{name} has no NovelAI generation settings")


def _stored_vibes(encodings: Any) -> int:
    if not isinstance(encodings, list):
        return 0
    return sum(1 for e in encodings if isinstance(e, str) and e)


def _normalized_vibes(strengths: Any) -> bool | None:
    """Guessed as the official import does: several strengths summing to 1 were normalized, above 1 they were
    not; otherwise unknown."""
    if not isinstance(strengths, list) or len(strengths) < 2:
        return None
    try:
        total = sum(abs(float(s)) for s in strengths)
    except (TypeError, ValueError):
        return None
    if 0.99 <= total <= 1.01:
        return True
    return False if total > 1 else None


def model_from_source(source: str | None) -> str | None:
    """Model id supported by nai-cli, or None for V3 and earlier."""
    if not source:
        return None
    if "NovelAI Diffusion V5" in source:
        return _V5_SOURCES.get(source, "nai-diffusion-5-curated")
    if "NovelAI Diffusion V4" in source or "DiffusionModelMetaName.NAIv4next" in source:
        curated = "nai-diffusion-4-5-curated" if "V4.5" in source else "nai-diffusion-4-curated-preview"
        return _V4_SOURCES.get(source, curated)
    return None


# The key V5 had while in testing, which the test site ran with the Curated weights
_STAGING_MODELS = {"custom": "nai-diffusion-5-curated"}


def model_of(cm: dict[str, Any], source: str | None) -> str | None:
    """The model named in the settings, else the one the Source line points to."""
    named = _STAGING_MODELS.get(str(cm.get("model_name") or ""), str(cm.get("model_name") or ""))
    try:
        return resolve(named).id if named else model_from_source(source)
    except errors.NaiError:
        return model_from_source(source)


def parse(path: str | Path) -> dict[str, Any]:
    return parse_raw(read_raw(path))


def parse_checked(path: str | Path) -> tuple[dict[str, Any], bool | None]:
    """The settings, and whether every pixel decodes, None for an image too large to decode in full; settings kept
    before a file was cut off are still read."""
    path = Path(path)
    data = _bytes(path)
    record = parse_raw(raw_of(data, path.name))
    try:
        decode.verify(data)
    except Image.DecompressionBombError:
        return record, None
    except decode.BAD_IMAGE:
        return record, False
    return record, True


def _bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read {path}: {e}") from e


@dataclass(frozen=True)
class Drawn:
    """An image's prompts as stored, with ||a|b|| groups, and as actually sent once each group was drawn."""

    prompt: tuple[str, str]
    negative: tuple[str, str]
    characters: tuple[tuple[str, str], ...]
    character_negatives: tuple[tuple[str, str], ...]


def drawn(raw: dict[str, str]) -> Drawn | None:
    """None when the image keeps no actual_prompts, which only random prompts leave."""
    return _readable(lambda: _drawn(_comment(raw)))


def _drawn(cm: dict[str, Any]) -> Drawn | None:
    actual = cm.get("actual_prompts")
    if not isinstance(actual, dict):
        return None
    sent, sent_neg = actual.get("prompt") or {}, actual.get("negative_prompt") or {}
    stored = (cm.get("v4_prompt") or {}).get("caption") or {}
    stored_neg = (cm.get("v4_negative_prompt") or {}).get("caption") or {}

    def pairs(stored_caption: dict, sent_caption: dict) -> tuple[tuple[str, str], ...]:
        before = [_text(c.get("char_caption", "")) for c in stored_caption.get("char_captions") or []]
        after = [_text(c.get("char_caption", "")) for c in sent_caption.get("char_captions") or []]
        return tuple(zip(before, after, strict=False)) if len(before) == len(after) else ()

    return Drawn(
        (_text(cm.get("prompt", "")), _text(sent.get("base_caption", ""))),
        (_text(cm.get("uc", "")), _text(sent_neg.get("base_caption", ""))),
        pairs(stored, sent),
        pairs(stored_neg, sent_neg),
    )


def with_drawn_prompts(raw: dict[str, str]) -> dict[str, str]:
    """The settings as if the prompts actually sent had been typed, as the import dialog's Actual Prompt reads
    them: both captions replaced whole. The switches beside them are settings, not prompts, and stay."""
    return _readable(lambda: _with_drawn_prompts(raw, _comment(raw)))


def _with_drawn_prompts(raw: dict[str, str], cm: dict[str, Any]) -> dict[str, str]:
    actual = cm.get("actual_prompts")
    if not isinstance(actual, dict):
        return raw
    switches = {k: v for k, v in (cm.get("v4_negative_prompt") or {}).items() if k != "caption"}
    use_coords = (cm.get("v4_prompt") or {}).get("use_coords", False)
    cm["v4_prompt"] = {"caption": dict(actual.get("prompt") or {}), "use_coords": use_coords}
    cm["v4_negative_prompt"] = {"caption": dict(actual.get("negative_prompt") or {}), **switches}
    return {**raw, "Comment": json.dumps(cm)}


def parse_raw(raw: dict[str, str]) -> dict[str, Any]:
    """raw holds the key-value pairs from text chunks or stealth data and must contain Comment; settings of a
    shape nai-cli cannot read are no_metadata."""
    return _readable(lambda: _record(raw, _comment(raw)))


def _comment(raw: dict[str, str]) -> dict[str, Any]:
    try:
        cm = values.parse_json(raw["Comment"])
    except ValueError as e:
        raise errors.NaiError(errors.NO_METADATA, f"Comment is not JSON: {e}") from e
    if not isinstance(cm, dict):
        raise errors.NaiError(errors.NO_METADATA, "Comment is not a JSON object")
    return cm


def _readable(read: Callable[[], T]) -> T:
    try:
        return read()
    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as e:
        # Other tools write Comment too, with any shape inside
        raise errors.NaiError(errors.NO_METADATA, f"the settings cannot be read: {type(e).__name__}: {e}") from e


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"a prompt is {type(value).__name__}, not text")
    return value


def _whole(cm: dict[str, Any], key: str) -> int | None:
    value = cm.get(key)
    if value is None or (values.is_whole(value) and values.is_number(value)):
        return value
    raise TypeError(f"{key} is {value!r}, not a whole number")


def _string(cm: dict[str, Any], key: str) -> str | None:
    value = cm.get(key)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} is {value!r}, not text")


def _premultiplied(straight_alpha: Any) -> bool | None:
    if straight_alpha is None or isinstance(straight_alpha, bool):
        return None if straight_alpha is None else not straight_alpha
    raise TypeError(f"straight_alpha is {straight_alpha!r}, not true or false")


def _number(value: Any, name: str) -> float | None:
    # true counts as 1 in Python, and would come back as --cfg=True
    if value is None or values.is_number(value):
        return value
    raise TypeError(f"{name} is {value!r}, not a number")


def _record(raw: dict[str, str], cm: dict[str, Any]) -> dict[str, Any]:
    v4p = (cm.get("v4_prompt") or {}).get("caption") or {}
    v4n = (cm.get("v4_negative_prompt") or {}).get("caption") or {}
    use_coords = bool((cm.get("v4_prompt") or {}).get("use_coords", False))
    source = raw.get("Source")
    model_id = model_of(cm, source)
    record: dict[str, Any] = {
        "source": source,
        "software": raw.get("Software"),
        "model": model_id,
        "prompt": _text(v4p.get("base_caption", cm.get("prompt", ""))),
        "negative": _text(v4n.get("base_caption", cm.get("uc", ""))),
        "characters": _characters_of(v4p, v4n),
        "seed": _whole(cm, "seed"),
        "width": _whole(cm, "width"),
        "height": _whole(cm, "height"),
        "steps": _whole(cm, "steps"),
        "scale": _number(cm.get("scale"), "scale"),
        "cfg_rescale": _number(cm.get("cfg_rescale"), "cfg_rescale"),
        "sampler": _string(cm, "sampler"),
        "noise_schedule": _string(cm, "noise_schedule"),
        "use_coords": use_coords,
        "strength": _number(cm.get("strength"), "strength"),
        "noise": _number(cm.get("noise"), "noise"),
        "request_type": _string(cm, "request_type"),
        "vibes": len(cm.get("reference_strength_multiple") or []),
        # Encodings the official import restores; gen_args reads them back with --vibes-from
        "stored_vibes": _stored_vibes(cm.get("reference_image_multiple")),
        "normalize_vibe_strength": _normalized_vibes(cm.get("reference_strength_multiple")),
        "precise_references": len(cm.get("director_reference_strengths") or []),
        "transparent": bool(cm.get("tag_hint_transparent_background")),
        "premultiplied_alpha": _premultiplied(cm.get("straight_alpha")),
        # Variety+ is on whenever a sigma is sent; its value follows from the size
        "variety": cm.get("skip_cfg_above_sigma") is not None,
        "legacy_uc": bool((cm.get("v4_negative_prompt") or {}).get("legacy_uc") or cm.get("legacy_uc")),
        # What a random prompt actually sent, which --actual-prompts imports
        "actual_prompts": cm.get("actual_prompts") if isinstance(cm.get("actual_prompts"), dict) else None,
        "raw": cm,
    }
    if model_id:
        record.update(_recover_user_input(cm, record, MODELS[model_id]))
    return record


def _characters_of(v4p: dict[str, Any], v4n: dict[str, Any]) -> list[dict[str, Any]]:
    """Characters with the negative at the same position and their first centre."""
    negatives = v4n.get("char_captions") or []
    chars = []
    for i, c in enumerate(v4p.get("char_captions") or []):
        center = (c.get("centers") or [{}])[0]
        chars.append(
            {
                "prompt": _text(c.get("char_caption", "")),
                "negative": _text(negatives[i].get("char_caption", "")) if i < len(negatives) else "",
                "x": _number(center.get("x"), "a character's x"),
                "y": _number(center.get("y"), "a character's y"),
            }
        )
    return chars


def _hint(cm: dict, key: str, table: dict[int, str]) -> str | None:
    value = cm.get(key)
    return table.get(value) if values.is_whole(value) else None


def _strip_negative(negative: str, model: Model, hinted: str | None, strip_nsfw: bool) -> tuple[str, str]:
    """Strip a leading nsfw added by the official rule, then the hinted UC preset or the longest matching one.
    Returns (preset, remaining negative prompt)."""
    body = P.strip_nsfw_prefix(negative) if strip_nsfw else negative
    if hinted == "none":
        return "none", body
    by_length = sorted(
        (name for name in presets.uc_presets(model) if name != "none"),
        key=lambda name: len(presets.uc_preset_text(model, name)),
        reverse=True,
    )
    for name in ([hinted] if hinted in by_length else []) + by_length:
        ok, rest = P.strip_uc_preset(body, presets.uc_preset_text(model, name))
        if ok:
            return name, rest
    return "none", body


def _recover_user_input(cm: dict, record: dict, model: Model) -> dict[str, Any]:
    """Remove the automatic teXt block, quality tags, UC preset and nsfw to recover rerunnable settings.
    Anything that would not rebuild exactly is kept verbatim."""
    quality, uc_hint = _hints(cm, model)
    typed = _user_prompt(record, model, quality)
    spec = R.GenSpec(
        model=model,
        prompt=typed.prompt,
        characters=[R.Character(c["prompt"], c["negative"], c["x"], c["y"]) for c in record["characters"]],
        quality=typed.quality,
        use_coords=record["use_coords"],
        auto_text=typed.auto_text,
        transparent=typed.transparent,
        furry=typed.furry,
    )
    uc, user_negative, official_nsfw, matched = _user_negative(record["negative"], spec, uc_hint)
    out: dict[str, Any] = {
        "user_prompt": typed.prompt,
        "user_negative": user_negative,
        "quality": typed.quality,
        "uc_preset": uc,
        "transparent": typed.transparent,
        "auto_text": typed.auto_text,
        "furry": typed.furry,
        "official_nsfw": official_nsfw,
        # Neither rule rebuilds it (usually edited by another client), so send the negative prompt verbatim
        "negative_verbatim": not matched,
        "quality_from": "tag_hint" if quality else "detected",
        "uc_preset_from": "tag_hint" if uc_hint else "detected",
    }
    spec = replace(spec, negative=user_negative, uc_preset=uc, official_nsfw=official_nsfw)
    try:
        rebuilt = R.effective_prompt(spec, R.resolve_centers(spec))
    except errors.NaiError:
        rebuilt = None
    prompt_text = record["prompt"]
    if rebuilt is not None and rebuilt != prompt_text:
        # Send verbatim when it would not rebuild; a trailing transparency tag is re-added, so drop it
        trailing, raw_prompt = (
            P.strip_suffix(prompt_text, P.compose_suffix(None, True)) if typed.transparent else (False, prompt_text)
        )
        out.update(user_prompt=raw_prompt, quality="none", transparent=trailing, auto_text=False, furry=False)
    return out


def _hints(cm: dict, model: Model) -> tuple[str | None, str | None]:
    """The quality tier and UC preset the image names, when the model has them."""
    quality = _hint(cm, "tag_hint_qt", _QUALITY_FROM_HINT)
    uc = _hint(cm, "tag_hint_uc_preset", _UC_FROM_HINT)
    if uc is None and isinstance(cm.get("ucPresetId"), str):
        uc = _UC_FROM_ID.get(cm["ucPresetId"])
    return (
        quality if quality in presets.quality_tiers(model) else None,
        uc if uc in presets.uc_presets(model) else None,
    )


@dataclass(frozen=True)
class _Typed:
    prompt: str
    quality: str
    transparent: bool
    auto_text: bool
    furry: bool


def _user_prompt(record: dict, model: Model, quality: str | None) -> _Typed:
    prompt = record["prompt"]
    # Like the official import, a leading fur dataset means furry mode
    furry = model.furry_mode and prompt.startswith(P.FURRY_PREFIX)
    if furry:
        prompt = prompt.removeprefix(P.FURRY_PREFIX)
    auto_text = model.auto_text
    if model.auto_text:
        chars = [
            P.AutoTextCharacter(
                c["prompt"], c["x"] if c["x"] is not None else 0.5, c["y"] if c["y"] is not None else 0.5
            )
            for c in record["characters"]
        ]
        stripped = P.strip_auto_text(prompt, chars, record["use_coords"])
        if stripped == prompt and P.auto_text_block(prompt, chars, record["use_coords"]) is not None:
            # Quotes without a teXt block mean automatic text was off
            auto_text = False
        prompt = stripped
    # Transparency counts only when its tag is actually removed, as in the official import
    for tier, with_transparency in _quality_candidates(model, quality, record["transparent"] and model.transparent):
        tags = presets.quality_tags(model, tier) if tier != "none" else None
        ok, stripped = P.strip_suffix(prompt, P.compose_suffix(tags, with_transparency))
        if ok:
            return _Typed(stripped, tier, with_transparency, auto_text, furry)
    return _Typed(prompt, "none", False, auto_text, furry)


def _user_negative(negative: str, spec: R.GenSpec, uc_hint: str | None) -> tuple[str, str, bool, bool]:
    """UC preset, typed negative, nsfw rule, and whether they rebuild the stored text. Both nsfw rules are tried,
    preferring a recognized UC preset, then the default rule."""
    matches = []
    for official in (False, True):
        uc, body = _strip_negative(negative, spec.model, uc_hint, strip_nsfw=official)
        if R.effective_negative(replace(spec, negative=body, uc_preset=uc, official_nsfw=official)) == negative:
            matches.append((uc == "none", official, uc, body))
    if not matches:
        uc, body = _strip_negative(negative, spec.model, uc_hint, strip_nsfw=False)
        return uc, body, False, False
    _, official, uc, body = min(matches)
    return uc, body, official, True


def _quality_candidates(model: Model, hinted: str | None, transparent: bool) -> list[tuple[str, bool]]:
    """(tier, transparency) in the order the official import tries them: the hinted tier first, then
    every tier with transparency, transparency alone and every tier without; a hint of none tries only
    transparency alone."""
    tiers = [t for t in presets.quality_tiers(model) if t != "none"]
    every = ([(t, True) for t in [*tiers, "none"]] if transparent else []) + [(t, False) for t in tiers]
    if hinted is None:
        return every
    if hinted == "none":
        return [("none", True)] if transparent else []
    return ([(hinted, True)] if transparent else []) + [(hinted, False), *every]


@dataclass(frozen=True)
class ImageNumber:
    """A number of the image's settings and where it goes; the model feature it needs, if any."""

    key: str
    attr: str
    low: float
    high: float
    whole: bool = False
    feature: str | None = None
    feature_name: str = ""

    def read(self, record: dict[str, Any]) -> int | float | None:
        value = record.get(self.key)
        if not values.is_number(value) or not self.low <= value <= self.high:
            return None
        if self.whole:
            return value if values.is_whole(value) else None
        return float(value)


SEED = ImageNumber("seed", "seed", 0, MAX_SEED, whole=True)
NUMERIC_SETTINGS = (
    ImageNumber("steps", "steps", 1, MAX_STEPS, whole=True),
    ImageNumber("scale", "cfg", 0, math.inf),
    ImageNumber(
        "cfg_rescale",
        "cfg_rescale",
        *CFG_RESCALE_RANGE,
        feature="cfg_rescale",
        feature_name="prompt guidance rescale",
    ),
)
