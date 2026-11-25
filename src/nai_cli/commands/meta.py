"""meta: the generation settings an image keeps, with the nai gen arguments that rerun it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, fs, ledger, meta, rerun
from nai_cli.commands import sources
from nai_cli.commands.context import LINES_IN_JOB, Context

# What a folder is searched for: the formats the official client writes settings into
IMAGE_SUFFIXES = (".png", ".webp")


@dataclass(frozen=True)
class MetaSettings:
    images: tuple[Path, ...]
    raw: bool
    # One JSON line per image and a summary, even for a single image
    lines: bool
    hidden_options: bool


def resolve(opts: dict[str, Any], ctx: Context) -> MetaSettings:
    given = tuple(map(Path, opts["image"]))
    images = tuple(s.path for s in sources.expand(given, IMAGE_SUFFIXES, bool(opts.get("recursive"))))
    # A folder prints lines even when it holds one image or none
    lines = bool(opts.get("jsonl")) or any(fs.is_folder(p) for p in given) or len(images) > 1
    return MetaSettings(images, bool(opts.get("raw")), lines, bool(ctx.config.defaults.get("v5_hidden_options")))


def run(s: MetaSettings, ctx: Context) -> dict[str, Any]:
    if s.lines and ctx.in_job:
        raise errors.NaiError(errors.BAD_ARGS, LINES_IN_JOB)
    warnings = []
    try:
        files = ledger.by_file()
    except OSError as e:
        files = {}
        warnings.append(f"cannot read the ledger: {e}")
    if not s.lines:
        (image,) = s.images
        return {"ok": True, **_intact(_read(image, s, files, warnings))}
    counts = {"images": len(s.images), "without_settings": 0, "damaged": 0, "failed": 0}
    for image in s.images:
        try:
            line = {"ok": True, "type": "image", **_intact(_read(image, s, files, warnings))}
        except errors.NaiError as e:
            # An image without settings is an answer about it, not a failure to read it
            if e.code == errors.NO_METADATA:
                counts["without_settings"] += 1
            else:
                counts["damaged" if e.detail.get("image_ok") is False else "failed"] += 1
            line = {**e.to_json(), "type": "image", "image": str(image)}
        ctx.emit(line)
    if counts["failed"] or counts["damaged"]:
        unread = [f"{counts['failed']} could not be read"] if counts["failed"] else []
        damaged = [f"{counts['damaged']} are damaged"] if counts["damaged"] else []
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"of {len(s.images)} images, " + " and ".join(unread + damaged),
            type="summary",
            meta=counts,
        )
    return {"ok": True, "type": "summary", "meta": counts}


def _intact(out: dict[str, Any]) -> dict[str, Any]:
    """The settings of a damaged image are still given, but the image does not pass as a good one."""
    if out["image_ok"] is not False:
        return out
    raise errors.NaiError(
        errors.BAD_ARGS,
        f"{Path(out['image']).name} is damaged; its settings were read from the text kept before the damage",
        **out,
    )


def _damage(intact: bool | None) -> list[str]:
    if intact is None:
        return ["the image is past the pixel limit, too large to check that every pixel decodes"]
    return [] if intact else ["the image is damaged: not every pixel decodes"]


def _read(image: Path, s: MetaSettings, files: dict[Path, dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    record, intact = meta.parse_checked(image)
    if not s.raw:
        record.pop("raw", None)
    out = {"image": str(image), "image_ok": intact, **record}
    out["gen_args"] = rerun.to_gen_args(record, image, hidden_options=s.hidden_options)
    out["warnings"] = [*warnings, *_damage(intact), *rerun.rerun_gaps(record)]
    entry = files.get(image.resolve())
    if entry:
        out["ledger"] = entry
    return out
