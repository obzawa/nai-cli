"""Sharing copies named after their source with a label, written all or none."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from nai_cli import decode, errors, fs, images, meta, naming, values
from nai_cli.commands import output, sources
from nai_cli.commands.context import Context
from nai_cli.commands.output import IMAGE_SUFFIXES
from nai_cli.commands.sources import Source

# What a folder is searched for: the formats images are shared in
FOLDER_SUFFIXES = (".png", ".webp", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Targets:
    images: tuple[Source, ...]
    # A .png name for one image; taken, it gets the next number, as every name does
    exact: Path | None
    folder: Path | None
    # Without -o: the output folder and its folder rule, as generated images are filed
    filed: naming.Destination
    # Named by the day and a number, not after the source
    rename: bool
    when: datetime
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context) -> Targets:
    given = tuple(map(Path, opts["images"]))
    found = sources.expand(given, FOLDER_SUFFIXES, bool(opts.get("recursive")))
    if not found:
        raise errors.NaiError(errors.BAD_ARGS, "no images in the folders given")
    out = values.path(opts["output"]) if opts.get("output") else None
    rename = bool(opts.get("rename"))
    exact = folder = None
    if out is not None and out.suffix.lower() == ".png":
        # A folder stands for as many images as it holds, even when it holds one
        if len(found) > 1 or any(fs.is_folder(p) for p in given):
            raise errors.NaiError(errors.BAD_ARGS, "-o names one file; give a folder for several images")
        if rename:
            raise errors.NaiError(errors.BAD_ARGS, "--rename names the copies itself; give -o a folder, or no -o")
        exact = out
    elif out is not None:
        if out.suffix.lower() in IMAGE_SUFFIXES and not fs.is_folder(out):
            raise errors.NaiError(errors.BAD_ARGS, f"copies are saved as PNG; give -o a .png name or a folder: {out}")
        folder = out
    return Targets(found, exact, folder, output.filed(ctx.config), rename, ctx.now(), ctx.dry_run)


def write(t: Targets, label: str, make: Callable[[Path], bytes], warnings: list[str] | None = None) -> dict[str, Any]:
    """NAME_label.png in the output folder, or NAME.png in the folder given, under the place its source has in the
    folder it came from; with rename, DATE-NUMBER.png there. Warnings are those make adds and one about names that
    give the image away."""
    # Every image is made before anything is written, so a bad one leaves no partial set
    copies = [make(source.path) for source in t.images]
    days = fs.DayNumbers(t.when)
    targets = [_target(t, source, label, data, days) for source, data in zip(t.images, copies, strict=True)]
    notes = [*(warnings or []), *_telling_names(t)]
    noted = {"warnings": notes} if notes else {}
    if t.dry_run:
        for target in targets:
            output.check_folder(target.parent, target if target == t.exact else None)
        return {"ok": True, "dry_run": True, "files": _listing(t.images, _free_names(targets)), **noted}
    written: list[Path] = []
    try:
        for target, data in zip(targets, copies, strict=True):
            written.append(fs.create_numbered(target, data))
    except BaseException as e:
        for path in written:
            fs.remove(path)
        if isinstance(e, OSError | ValueError):
            raise errors.NaiError(errors.BAD_ARGS, f"cannot write {target}: {e}") from e
        raise
    return {"ok": True, "files": _listing(t.images, written), **noted}


def _free_names(targets: list[Path]) -> list[Path]:
    """The names the copies would be written under, each after those before it."""
    taken: set[Path] = set()
    out = []
    for target in targets:
        path = fs.free_name(target, taken)
        taken.add(path)
        out.append(path)
    return out


def _target(t: Targets, source: Source, label: str, data: bytes, days: fs.DayNumbers) -> Path:
    if t.exact is not None:
        return t.exact
    if t.folder is not None:
        folder, kept = t.folder / source.relative.parent, ""
    else:
        folder, kept = t.filed.folder_for(_kept(data, t.when)), "_" + label
    if t.rename:
        return days.next_in(folder)
    return folder / f"{naming.fit(source.path.stem, kept)}.png"


def _telling_names(t: Targets) -> list[str]:
    # A copy named after its source keeps what the official download name holds, however clean the copy is
    if t.rename or t.exact is not None:
        return []
    count = sum(naming.is_download_name(source.path.stem) for source in t.images)
    if not count:
        return []
    which = "a copy is" if count == 1 else f"{count} copies are"
    return [
        f"{which} named after the official download name, which holds the start of the prompt and the seed; "
        "--rename names copies by date and number instead"
    ]


def _kept(data: bytes, when: datetime) -> naming.Meta:
    # Filed by the settings the copy keeps, so one without them is filed by date and size alone
    width, height = decode.size_of_bytes(data)
    try:
        record = meta.parse_raw(meta.raw_of(data, "the copy"))
    except errors.NaiError:
        return naming.Meta(date=when, width=width, height=height)
    return replace(output.recorded(record, ""), date=when, width=width, height=height)


def _listing(images: tuple[Source, ...], files: list[Path]) -> list[dict[str, str]]:
    return [{"source": str(a.path), "file": str(b)} for a, b in zip(images, files, strict=True)]


def color(text: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if text is None:
        return default
    value = text.strip().removeprefix("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise errors.NaiError(errors.BAD_ARGS, f"a colour is #RRGGBB: {text}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def finish(source: Path, edited: Image.Image, keep_metadata: bool, warnings: list[str]) -> bytes:
    """The edited pixels as a bare PNG; with keep_metadata, the source's PNG text is carried over as it is. Settings
    it keeps in EXIF or in the alpha channel, which the edit loses, are written as text in place of PNG text of the
    same keywords, so the copy reads back as the source did."""
    out = images.strip_metadata(edited)
    if not keep_metadata:
        return out
    data = source.read_bytes()
    try:
        settings = meta.raw_of(data, source.name)
    except errors.NaiError:
        settings = {}
    if not settings or "Comment" in meta.png_text_chunks(data):
        # The settings are the PNG text itself, or there are none
        return images.copy_text_chunks(data, out)
    out = images.copy_text_chunks(data, out, leaving=set(settings))
    return images.add_text(out, _writable(settings, source.name, warnings))


def _writable(settings: dict[str, str], name: str, warnings: list[str]) -> dict[str, str]:
    # EXIF and alpha-channel settings may hold any key, which PNG text cannot always carry
    out = {}
    for key, text in settings.items():
        try:
            decode.itxt(key, text)
        except ValueError as e:
            warnings.append(f"{name}: {e}, so the copy leaves it out")
            continue
        out[key] = text
    return out
