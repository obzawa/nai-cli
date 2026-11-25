"""Edits made before sharing an image: regions redacted, or a visible watermark of text and a logo."""

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from nai_cli import errors, inpaint

EFFECTS = ("pixelate", "blur", "solid")
# Block and blur sizes as fractions of the short edge
PIXEL_SIZE = 0.025
BLUR = 0.018
ANCHORS = (
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
)
# Another corner must be this much calmer than the chosen one before the marks move there
_SWITCH_MARGIN = 0.15
# Placement and tone are judged on a copy this long
_ANALYSIS_EDGE = 256
_TONE_CONTRAST = 0.7
_TRANSPARENT_LIGHTNESS = 50.0
# Outline and shadow of text, as fractions of the short edge, and their alpha
_STROKE = 0.002
_STROKE_ALPHA = 0x99
_SHADOW_BLUR = 0.008
_SHADOW_OFFSET = (0.002, 0.003)
_SHADOW_ALPHA = 0x80
_NO_MIRROR = (False, False)

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont
# Noncharacters, which no font may draw, so each shows how this font draws a character it lacks; a private use
# character would not do, since some fonts draw those
_NO_GLYPHS = ("\ufdd0", "\uffff", "\U0010ffff")
_GLYPH_CHECK_SIZE = 32
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Redaction:
    regions: tuple[inpaint.Region, ...]
    effect: str = "pixelate"
    # Block or blur size as a fraction of the short edge
    amount: float | None = None
    color: tuple[int, int, int] = (0, 0, 0)
    opacity: float = 1.0
    # Redact everything outside the regions
    invert: bool = False


def redact(img: Image.Image, r: Redaction, segment: inpaint.Segment | None = None) -> Image.Image:
    source = img.convert("RGBA")
    width, height = source.size
    short = min(width, height)
    mask = inpaint.draw_regions(Image.new("L", source.size, 0), list(r.regions), source, segment)
    if r.invert:
        mask = mask.point(lambda v: 255 - v)
    if not mask.getbbox():
        raise errors.NaiError(errors.BAD_ARGS, "the regions cover nothing to redact")
    if r.effect == "pixelate":
        block = max(2.0, short * (r.amount or PIXEL_SIZE))
        small = source.resize((max(1, round(width / block)), max(1, round(height / block))), Image.Resampling.BOX)
        cover = small.resize(source.size, Image.Resampling.NEAREST)
    elif r.effect == "blur":
        cover = source.filter(ImageFilter.GaussianBlur(max(0.5, short * (r.amount or BLUR))))
    else:
        cover = Image.new("RGBA", source.size, (*r.color, 255))
    if r.opacity < 1:
        mask = mask.point(lambda v: round(v * r.opacity))
    return Image.composite(cover, source, mask)


@dataclass(frozen=True)
class Layer:
    """Where one part of a watermark goes, placed by an anchor inside a margin; size and margin are fractions of
    the short edge."""

    anchor: str
    size: float
    margin: float
    opacity: float


@dataclass(frozen=True)
class Watermark:
    text: str | None
    text_layer: Layer = Layer("bottom-right", 0.1, 0.035, 0.82)
    font: Path | None = None
    color: tuple[int, int, int] = (255, 255, 255)
    logo: Image.Image | None = None
    logo_layer: Layer = Layer("bottom-left", 0.16, 0.035, 0.85)
    # Mirror the marks to the calmest corners
    auto_place: bool = False
    # Grey the marks to stand out from what lies under them
    auto_tone: bool = False


def watermark(img: Image.Image, w: Watermark) -> Image.Image:
    out = img.convert("RGBA")
    marks = _marks(w, out.size)
    if not marks:
        raise errors.NaiError(errors.BAD_ARGS, "a watermark needs text or a logo")
    analysis = _Analysis(out) if w.auto_place or w.auto_tone else None
    mirror = _choose_mirror(marks, out.size, analysis) if w.auto_place and analysis else _NO_MIRROR
    for mark in marks:
        box = _box(mark.layer, mark.size, out.size, mirror)
        lightness = analysis.region(box, out.size)[0] if w.auto_tone and analysis else None
        picture, (dx, dy) = mark.render(lightness, mirror)
        _overlay(out, picture, round(box[0]) + dx, round(box[1]) + dy)
    return out


@dataclass(frozen=True)
class _Logo:
    layer: Layer
    picture: Image.Image

    @property
    def size(self) -> tuple[int, int]:
        return self.picture.size

    def render(self, lightness: float | None, mirror: tuple[bool, bool]) -> tuple[Image.Image, tuple[int, int]]:
        picture = self.picture if lightness is None else _toned_logo(self.picture, lightness)
        return _faded(picture, self.layer.opacity), (0, 0)


@dataclass(frozen=True)
class _Text:
    layer: Layer
    text: str
    font: Font
    color: tuple[int, int, int]
    # In pixels
    stroke: int
    shadow_blur: float
    shadow_offset: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        left, top, right, bottom = self._bounds()
        return right - left, bottom - top

    def render(self, lightness: float | None, mirror: tuple[bool, bool]) -> tuple[Image.Image, tuple[int, int]]:
        """The text with its outline and shadow, and where it starts against the box measured, since the shadow
        reaches past it."""
        fill, halo = (self.color, (0, 0, 0)) if lightness is None else _tones(lightness)
        # Right-aligned lines, flipped with the marks
        align = "left" if mirror[0] else "right"
        left, top, right, bottom = self._bounds()
        pad = math.ceil(self.shadow_blur * 3) + max(map(abs, self.shadow_offset))
        canvas = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
        origin = (pad - left, pad - top)
        shadow_at = (origin[0] + self.shadow_offset[0], origin[1] + self.shadow_offset[1])
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(shadow_at, self.text, font=self.font, fill=(*halo, _SHADOW_ALPHA), align=align)
        canvas = canvas.filter(ImageFilter.GaussianBlur(self.shadow_blur))
        ImageDraw.Draw(canvas).multiline_text(
            origin,
            self.text,
            font=self.font,
            fill=(*fill, 255),
            stroke_width=self.stroke,
            stroke_fill=(*halo, _STROKE_ALPHA),
            align=align,
        )
        return _faded(canvas, self.layer.opacity), (-pad, -pad)

    def _bounds(self) -> tuple[int, int, int, int]:
        draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        left, top, right, bottom = draw.multiline_textbbox((0, 0), self.text, font=self.font, stroke_width=self.stroke)
        return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)


def anchor_cell(anchor: str) -> tuple[int, int]:
    """Column and row of an anchor, 0 to 2 from the top left."""
    index = ANCHORS.index(anchor)
    return index % 3, index // 3


def _box(layer: Layer, size: tuple[int, int], canvas: tuple[int, int], mirror: tuple[bool, bool]) -> Box:
    column, row = anchor_cell(layer.anchor)
    column, row = (2 - column if mirror[0] else column), (2 - row if mirror[1] else row)
    margin = layer.margin * min(canvas)
    width, height = size
    left = min(max(0.0, margin + column / 2 * (canvas[0] - 2 * margin - width)), max(0.0, canvas[0] - width))
    top = min(max(0.0, margin + row / 2 * (canvas[1] - 2 * margin - height)), max(0.0, canvas[1] - height))
    return left, top, left + width, top + height


def _marks(w: Watermark, canvas: tuple[int, int]) -> list[_Logo | _Text]:
    short = min(canvas)
    marks: list[_Logo | _Text] = []
    if w.logo is not None:
        scale = w.logo_layer.size * short / max(w.logo.size)
        size = (max(1, round(w.logo.width * scale)), max(1, round(w.logo.height * scale)))
        marks.append(_Logo(w.logo_layer, w.logo.convert("RGBA").resize(size, Image.Resampling.LANCZOS)))
    if w.text:
        marks.append(_text_mark(w, canvas))
    return marks


def _text_mark(w: Watermark, canvas: tuple[int, int]) -> _Text:
    short = min(canvas)
    room = (
        max(1.0, canvas[0] - 2 * w.text_layer.margin * short),
        max(1.0, canvas[1] - 2 * w.text_layer.margin * short),
    )
    font_size = max(1.0, w.text_layer.size * short)
    offset = (round(short * _SHADOW_OFFSET[0]), round(short * _SHADOW_OFFSET[1]))

    def mark(size: float) -> _Text:
        scale = size / font_size
        return _Text(
            w.text_layer,
            w.text or "",
            load_font(w.font, size),
            w.color,
            max(1, round(short * _STROKE * scale)),
            short * _SHADOW_BLUR * scale,
            (round(offset[0] * scale), round(offset[1] * scale)),
        )

    full = mark(font_size)
    width, height = full.size
    # Text too long for the canvas shrinks to fit, outline and shadow with it
    scale = min(1.0, room[0] / max(1, width), room[1] / max(1, height))
    return full if scale >= 1 else mark(max(1.0, font_size * scale))


def text_font(text: str, font: Path | None, offered: Iterable[Path]) -> Path | None:
    """The font to draw text in: the one given, else the built-in one, else the first font offered that draws all
    of it, the installed fonts in practice. A font given is never replaced; text no font draws is refused."""
    if font is not None:
        require_glyphs(text, font)
        return font
    missing, face = _missing(text, None)
    # Without FreeType no font file can be used, and require_glyphs says so
    if not missing or not isinstance(face, ImageFont.FreeTypeFont):
        require_glyphs(text, None)
        return None
    chosen = next((path for path in offered if _draws(text, path)), None)
    if chosen is None:
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"neither the built-in font nor an installed one has glyphs for {missing}; give --font a font that has them",
        )
    return chosen


def _draws(text: str, font: Path) -> bool:
    try:
        return not _missing(text, font)[0]
    except errors.NaiError:
        # A damaged or unreadable installed font is passed over like one without the glyphs
        return False


def require_glyphs(text: str, font: Path | None) -> None:
    """Text the font cannot draw is refused, not drawn as boxes; the built-in font has little beyond ASCII."""
    missing, face = _missing(text, font)
    if not missing:
        return
    if not isinstance(face, ImageFont.FreeTypeFont):
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"this Pillow has no FreeType, so its bitmap font draws Latin-1 only, not {missing}; "
            "install a Pillow built with FreeType",
        )
    name = "the built-in font" if font is None else font.name
    raise errors.NaiError(errors.BAD_ARGS, f"{name} has no glyphs for {missing}; give --font a font that has them")


def _missing(text: str, font: Path | None) -> tuple[str, Font]:
    """The characters of text the font cannot draw, and the font. Each character is checked in the composed form
    the mark is drawn in, a combining mark left over on its own; joiners and variation selectors only change the
    character beside them and are not checked."""
    shown = [ch for ch in unicodedata.normalize("NFC", text) if not ch.isspace() and not _invisible(ch)]
    if not shown:
        raise errors.NaiError(errors.BAD_ARGS, "the text has nothing to draw")
    face = load_font(font, _GLYPH_CHECK_SIZE)
    boxes = {_drawn(face, ch) for ch in _NO_GLYPHS}
    return "".join(dict.fromkeys(ch for ch in shown if _drawn(face, ch) in boxes | {None})), face


def _invisible(ch: str) -> bool:
    # Format characters such as the zero width joiner, and variation selectors
    return unicodedata.category(ch) == "Cf" or "\ufe00" <= ch <= "\ufe0f" or "\U000e0100" <= ch <= "\U000e01ef"


def _drawn(face: Font, ch: str) -> tuple[tuple[int, int], bytes] | None:
    try:
        left, top, right, bottom = face.getbbox(ch)
        img = Image.new("L", (max(1, int(right - left)), max(1, int(bottom - top))))
        ImageDraw.Draw(img).text((-left, -top), ch, font=face, fill=255)
    except UnicodeEncodeError:
        # A bitmap font, without FreeType, has Latin-1 only
        return None
    return img.size, img.tobytes()


def load_font(path: Path | None, size: float) -> Font:
    """The font at path, or the built-in one, at size; without FreeType only the built-in bitmap font."""
    try:
        # The truetype factory searches system fonts by basename when the given file cannot be opened
        return ImageFont.FreeTypeFont(str(path), size) if path is not None else ImageFont.load_default(size)
    except ImportError as e:
        # Pillow built without FreeType has only its small bitmap font, which takes no size and no font file
        if path is None:
            return ImageFont.load_default()
        raise errors.NaiError(
            errors.BAD_ARGS, f"this Pillow has no FreeType, so it cannot use the font {path}; leave out --font"
        ) from e
    except OSError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot load the font {path}: {e}") from e


def _faded(img: Image.Image, opacity: float) -> Image.Image:
    out = img.convert("RGBA")
    out.putalpha(out.getchannel("A").point(lambda v: round(v * opacity)))
    return out


def _overlay(base: Image.Image, picture: Image.Image, x: int, y: int) -> None:
    """alpha_composite that clips what falls outside the base, which Pillow refuses."""
    left, top = max(0, x), max(0, y)
    right, bottom = min(base.width, x + picture.width), min(base.height, y + picture.height)
    if right > left and bottom > top:
        base.alpha_composite(picture.crop((left - x, top - y, right - x, bottom - y)), (left, top))


class _Analysis:
    """Lightness and detail of the image, judged on a copy at most _ANALYSIS_EDGE long."""

    def __init__(self, img: Image.Image) -> None:
        scale = min(1.0, _ANALYSIS_EDGE / max(img.size))
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.BOX)
        rgba = np.asarray(small, dtype=np.float64) / 255
        self.alpha = rgba[..., 3]
        self.lightness = _lightness(rgba[..., :3])
        # Transparent areas count as flat mid grey, so a cut-out's outline counts as detail
        composite = self.alpha * self.lightness + (1 - self.alpha) * _TRANSPARENT_LIGHTNESS
        gy, gx = np.gradient(composite) if min(composite.shape) > 1 else (np.zeros_like(composite),) * 2
        self.detail = gx**2 + gy**2

    def region(self, box: Box, canvas: tuple[int, int]) -> tuple[float | None, float]:
        """Mean lightness, None where fully transparent, and the root mean square of the detail."""
        rows, columns = self.alpha.shape
        x0 = min(columns - 1, max(0, math.floor(box[0] / canvas[0] * columns)))
        y0 = min(rows - 1, max(0, math.floor(box[1] / canvas[1] * rows)))
        x1 = max(x0 + 1, min(columns, math.ceil(box[2] / canvas[0] * columns)))
        y1 = max(y0 + 1, min(rows, math.ceil(box[3] / canvas[1] * rows)))
        alpha = self.alpha[y0:y1, x0:x1]
        coverage = float(alpha.sum())
        lightness = float((alpha * self.lightness[y0:y1, x0:x1]).sum() / coverage) if coverage > 1e-9 else None
        return lightness, float(np.sqrt(self.detail[y0:y1, x0:x1].mean()))


def _choose_mirror(marks: list[_Logo | _Text], canvas: tuple[int, int], analysis: _Analysis) -> tuple[bool, bool]:
    costs = {
        mirror: sum(analysis.region(_box(m.layer, m.size, canvas, mirror), canvas)[1] for m in marks)
        for mirror in (_NO_MIRROR, (True, False), (False, True), (True, True))
    }
    best = min(costs, key=costs.__getitem__)
    # Close calls stay where the user put the marks
    return best if costs[best] < costs[_NO_MIRROR] * (1 - _SWITCH_MARGIN) else _NO_MIRROR


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """CIE L*, 0 to 100, where equal steps look equal."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(luminance > 216 / 24389, 116 * np.cbrt(luminance) - 16, luminance * 24389 / 27)


def _grey(lightness: float) -> int:
    """The sRGB value of the grey at this L*."""
    lightness = min(100.0, max(0.0, lightness))
    linear = ((lightness + 16) / 116) ** 3 if lightness > 8 else lightness / (24389 / 27)
    srgb = linear * 12.92 if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
    return round(srgb * 255)


def _targets(background: float) -> tuple[float, float]:
    """The mark's lightness and its outline's, pushed away from the background: lighter on dark, darker on
    light."""
    if background <= 50:
        return background + _TONE_CONTRAST * (100 - background), background - _TONE_CONTRAST * background
    return background - _TONE_CONTRAST * background, background + _TONE_CONTRAST * (100 - background)


def _tones(background: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    fill, halo = (_grey(v) for v in _targets(background))
    return (fill, fill, fill), (halo, halo, halo)


def _toned_logo(logo: Image.Image, background: float) -> Image.Image:
    """The logo in grey with its mean moved to the target, its range squeezed to fit; alpha is kept."""
    rgba = np.asarray(logo.convert("RGBA"), dtype=np.float64)
    alpha = rgba[..., 3]
    if not alpha.any():
        return logo
    luma = (rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722])) / 255
    mean = float((luma * alpha).sum() / alpha.sum())
    # Near-transparent edge pixels carry unreliable colour
    solid = luma[alpha >= 8]
    darkest, brightest = (float(solid.min()), float(solid.max())) if solid.size else (mean, mean)
    target = _grey(_targets(background)[0]) / 255
    gain = 1.0
    if mean > darkest:
        gain = min(gain, target / (mean - darkest))
    if brightest > mean:
        gain = min(gain, (1 - target) / (brightest - mean))
    grey = np.clip((gain * luma + (target - gain * mean)) * 255, 0, 255)
    return Image.fromarray(np.dstack([grey, grey, grey, alpha]).round().astype(np.uint8), "RGBA")
