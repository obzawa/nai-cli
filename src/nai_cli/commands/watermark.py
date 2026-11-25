"""watermark: copies of images with visible text or a logo, for sharing."""

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nai_cli import errors, fonts, images, share
from nai_cli.commands import copies
from nai_cli.commands.context import Context

# Drawn when neither --text nor --logo is given
DEFAULT_TEXT = "NovelAI"


@dataclass(frozen=True)
class WatermarkSettings:
    targets: copies.Targets
    mark: share.Watermark
    keep_metadata: bool


def resolve(opts: dict[str, Any], ctx: Context) -> WatermarkSettings:
    logo = images.open_still(opts["logo"]) if opts.get("logo") else None
    # A logo alone has no text unless asked for
    text = opts.get("text") if opts.get("text") is not None else (None if logo else DEFAULT_TEXT)
    defaults = share.Watermark(None)
    # Composed, so a font draws an accented letter as one glyph and the check sees what is drawn
    text = unicodedata.normalize("NFC", text.replace("\\n", "\n")) if text else None
    font = Path(opts["font"]) if opts.get("font") else None
    if text:
        font = share.text_font(text, font, fonts.offered(text))
    mark = share.Watermark(
        text=text,
        text_layer=_layer(opts, "", defaults.text_layer),
        font=font,
        color=copies.color(opts.get("color"), defaults.color),
        logo=logo,
        logo_layer=_layer(opts, "logo_", defaults.logo_layer),
        auto_place=bool(opts.get("auto_place")),
        auto_tone=bool(opts.get("auto_tone")),
    )
    return WatermarkSettings(copies.resolve(opts, ctx), mark, bool(opts.get("keep_metadata")))


def _layer(opts: dict[str, Any], prefix: str, default: share.Layer) -> share.Layer:
    def value(key: str, valid: Callable[[float], bool]) -> float:
        given = opts.get(prefix + key)
        if given is None:
            return float(getattr(default, key))
        if not valid(given):
            raise errors.NaiError(errors.BAD_ARGS, f"--{(prefix + key).replace('_', '-')} is out of range: {given}")
        return float(given)

    return share.Layer(
        anchor=opts.get(prefix + "position") or default.anchor,
        size=value("size", lambda v: 0 < v <= 1),
        margin=value("margin", lambda v: 0 <= v < 0.5),
        opacity=value("opacity", lambda v: 0 < v <= 1),
    )


def run(s: WatermarkSettings, ctx: Context) -> dict[str, Any]:
    warnings: list[str] = []

    def make(source: Path) -> bytes:
        return copies.finish(source, share.watermark(images.open_still(source), s.mark), s.keep_metadata, warnings)

    return copies.write(s.targets, "watermarked", make, warnings)
