"""Images as the requests send them: opened, converted, resized for import, masked and stripped."""

import io
import math
from collections.abc import Collection
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from nai_cli import decode, errors, presets, resample
from nai_cli.presets import GRID, MAX_PIXELS, nearest_grid

LATENT_SCALE = 8
PRECISE_REFERENCE_CANVASES = [(1024, 1536), (1536, 1024), (1472, 1472)]
_PORTRAIT_BUCKETS = [
    (width, height)
    for width, tallest, shortest in [
        (256, 4096, 3328),
        (320, 3264, 2752),
        (384, 2688, 2368),
        (448, 2304, 2112),
        (512, 2048, 1856),
        (576, 1792, 1664),
        (640, 1600, 1536),
        (704, 1472, 1408),
        (768, 1344, 1280),
        (832, 1216, 1216),
        (896, 1152, 1152),
        (960, 1088, 1088),
    ]
    for height in range(tallest, shortest - 1, -64)
]
# In the official order, which settles ties: tallest first, square, then their mirrors
REFERENCE_BUCKETS = [*_PORTRAIT_BUCKETS, (1024, 1024), *((h, w) for w, h in reversed(_PORTRAIT_BUCKETS))]
# Masks are binarized on alpha before sending; values above this are repainted
MASK_THRESHOLD = 155


def open_image(path: str | Path) -> Image.Image:
    try:
        img = _open(path)
        img.load()
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return ImageOps.exif_transpose(img)


# Modes PNG stores as they are; others, such as CMYK from a JPEG, are converted first
_PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I;16"})


def to_png(img: Image.Image) -> bytes:
    if img.mode == "I":
        # Pillow stores I as 16 bits but no longer takes it as is; I;16 writes the same file
        img = img.convert("I;16")
    elif img.mode not in _PNG_MODES:
        img = _rgb_or_rgba(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def copy_text_chunks(source: bytes, target: bytes, leaving: Collection[str] = ()) -> bytes:
    """Copy the text chunks of source verbatim after the IHDR of target, so the composite keeps its metadata;
    chunks of the keywords in leaving are not copied."""
    texts = [
        source[start:end]
        for kind, body, start, end in decode.png_chunks(source)
        if kind in decode.TEXT_CHUNKS and body.split(b"\0", 1)[0].decode("latin1") not in leaving
    ]
    return _after_header(target, b"".join(texts))


def add_text(target: bytes, texts: dict[str, str]) -> bytes:
    """texts written as iTXt chunks after the IHDR of target."""
    return _after_header(target, b"".join(decode.itxt(key, text) for key, text in texts.items()))


def _after_header(target: bytes, chunks: bytes) -> bytes:
    for kind, _, _, end in decode.png_chunks(target):
        if kind == b"IHDR":
            return target[:end] + chunks + target[end:]
    return target


# Reference sides for imported images: 1216 long and 896 short for portrait
_IMPORT_LONG_SIDE = 1216
_IMPORT_SHORT_SIDE = 896


def import_size(width: int, height: int, current: tuple[int, int]) -> tuple[int, int]:
    """Generation size for an imported image: the current size if the aspect ratio matches and it fits;
    the source size if it is a multiple of 64 within the limit; otherwise scaled from a 1216 long or 896 short side."""
    ratio = width / height
    short, long = (height, width) if ratio > 1 else (width, height)
    oriented = short / long
    # Compared against the portrait-oriented sides while current stays as is
    if current[0] / current[1] == ratio and short <= current[0] and long <= current[1]:
        return current
    if not (short % GRID == 0 and long % GRID == 0 and short * long <= MAX_PIXELS):
        from_long = nearest_grid(_IMPORT_LONG_SIDE * oriented)
        from_short = nearest_grid(_IMPORT_SHORT_SIDE / oriented)
        long_error = abs(from_long / _IMPORT_LONG_SIDE - oriented)
        short_error = abs(_IMPORT_SHORT_SIDE / from_short - oriented)
        if long_error < short_error and from_long * _IMPORT_LONG_SIDE <= MAX_PIXELS:
            short, long = from_long, _IMPORT_LONG_SIDE
        elif _IMPORT_SHORT_SIDE * from_short <= MAX_PIXELS:
            short, long = _IMPORT_SHORT_SIDE, from_short
        else:
            short, long = 512, 512
    return (long, short) if ratio > 1 else (short, long)


def clear_stealth_alpha(img: Image.Image) -> Image.Image:
    """For images carrying stealth data, alpha 254 becomes 255 and 1 becomes 0 so the bits do not show as noise."""
    if img.width * img.height > decode.STEALTH_MAX_PIXELS or not _has_compressed_stealth(img):
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: 255 if v == 254 else 0 if v == 1 else v))
    return out


def _has_compressed_stealth(img: Image.Image) -> bool:
    """Detect stealth_pngcomp: alpha LSBs read column-major start with the magic."""
    if "A" not in img.getbands():
        return False
    alpha = img.getchannel("A")
    width, height = alpha.size
    if width * height < 8 * len(decode.STEALTH_COMPRESSED):
        return False
    pixels = alpha.load()
    for i, expected in enumerate(decode.STEALTH_COMPRESSED):
        value = 0
        for bit in range(8):
            index = 8 * i + bit
            value |= (pixels[index // height, index % height] & 1) << (7 - bit)
        if value != expected:
            return False
    return True


def source_image(path: str | Path, width: int, height: int, keep_alpha: bool) -> bytes:
    """Base image for img2img and inpainting, stretched to size, on white when the model has no transparency."""
    return to_png(source_canvas(open_image(path), width, height, keep_alpha))


def clean_source(img: Image.Image) -> Image.Image:
    return clear_stealth_alpha(_rgb_or_rgba(img))


def source_canvas(img: Image.Image, width: int, height: int, keep_alpha: bool) -> Image.Image:
    img = resample.lanczos(clean_source(img), width, height)
    if not keep_alpha and img.mode == "RGBA":
        img = resample.fill_background(img, (255, 255, 255))
    return img


def mask_coverage(img: Image.Image) -> Image.Image:
    """White or opaque areas are repainted: alpha decides for transparent masks, luminance for opaque ones."""
    if "A" in img.getbands() and img.getextrema()[-1][0] < 255:
        return img.getchannel("A")
    return img.convert("L")


def latent_mask(coverage: Image.Image, width: int, height: int) -> Image.Image:
    """Nearest-neighbour downscale to latent size (1/8) and threshold at 155; returns a 0/255 grayscale image."""
    latent = resample.nearest(coverage, width // LATENT_SCALE, height // LATENT_SCALE)
    binary = latent.point(lambda v: 255 if v > MASK_THRESHOLD else 0)
    if binary.getextrema()[1] == 0:
        raise errors.NaiError(errors.EMPTY_MASK, "the mask selects nothing to repaint")
    return binary


def mask_request(latent: Image.Image, width: int, height: int) -> bytes:
    """Nearest-neighbour upscale back to the generation size on black: white is repainted, black is kept."""
    full = resample.nearest(latent, width, height)
    return to_png(Image.merge("RGBA", (full, full, full, Image.new("L", full.size, 255))))


def tool_image(img: Image.Image) -> tuple[bytes, int, int]:
    """Director tool input, scaled to roughly 1 to 3 megapixels."""
    img = _rgb_or_rgba(img)
    width, height = presets.tool_request_size(img.width, img.height)
    return to_png(resample.lanczos(img, width, height)), width, height


def _rgb_or_rgba(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img
    return img.convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")


def precise_reference_image(path: str | Path, transparent: bool) -> bytes:
    return to_png(precise_reference_canvas(open_image(path), transparent))


def precise_reference_canvas(img: Image.Image, transparent: bool = False) -> Image.Image:
    """Fit into the closest of three canvases and centre on transparency, which a model without transparency gets
    filled black."""
    img = clean_source(img)
    ratio = img.width / img.height
    cw, ch = min(PRECISE_REFERENCE_CANVASES, key=lambda c: abs(c[0] / c[1] - ratio))
    if ratio > cw / ch:
        width, height = cw, resample.js_round(cw / ratio)
    else:
        width, height = resample.js_round(ch * ratio), ch
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    left, top = resample.js_round((cw - width) / 2), resample.js_round((ch - height) / 2)
    canvas.paste(resample.lanczos(img, width, height).convert("RGBA"), (left, top))
    return canvas if transparent else resample.fill_background(canvas, (0, 0, 0))


def bucketed_reference_image(path: str | Path) -> bytes:
    return to_png(bucketed_reference(open_image(path)))


def bucketed_reference(img: Image.Image) -> Image.Image:
    """Scaled to cover the size bucket closest in aspect ratio and cropped to it from the centre; transparency
    stays."""
    img = clean_source(img)
    width, height = reference_bucket(img.width, img.height)
    if img.size == (width, height):
        return img
    scale = max(width / img.width, height / img.height)
    scaled_w = max(math.floor(img.width * scale + 0.5), width)
    scaled_h = max(math.floor(img.height * scale + 0.5), height)
    left, top = (scaled_w - width) // 2, (scaled_h - height) // 2
    return resample.lanczos(img, scaled_w, scaled_h).crop((left, top, left + width, top + height))


def reference_bucket(width: int, height: int) -> tuple[int, int]:
    aspect = math.log(max(width, 1) / max(height, 1))
    return min(REFERENCE_BUCKETS, key=lambda b: abs(aspect - math.log(b[0] / b[1])))


def size_of(path: str | Path) -> tuple[int, int]:
    return open_image(path).size


def strip_metadata(img: Image.Image) -> bytes:
    """A PNG of the pixels alone: no text chunks, EXIF or colour profile, and no low alpha bit, where stealth
    settings hide."""
    return to_png(_bare(img))


def strip_file(path: str | Path) -> bytes:
    """strip_metadata for an image file; an animated one stays animated, each frame cleaned, as an APNG."""
    try:
        with _open(path) as img:
            if getattr(img, "n_frames", 1) > 1:
                return _strip_animation(img)
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    return strip_metadata(open_image(path))


def open_still(path: str | Path) -> Image.Image:
    """open_image for edits that draw on one picture, which an animation is not."""
    try:
        with _open(path) as img:
            animated = getattr(img, "n_frames", 1) > 1
    except decode.BAD_IMAGE as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read image {path}: {e}") from e
    if animated:
        raise errors.NaiError(errors.BAD_ARGS, f"{path} is animated; only still images can be edited")
    return open_image(path)


def _open(path: str | Path) -> Image.Image:
    return Image.open(io.BytesIO(Path(path).read_bytes()))


def _strip_animation(img: Image.Image) -> bytes:
    frames, durations = [], []
    for frame in ImageSequence.Iterator(img):
        # Pillow composites each frame whole, so every frame replaces the last when played
        frames.append(_bare(frame.convert("RGBA")))
        durations.append(frame.info.get("duration", 0))
    buf = io.BytesIO()
    frames[0].save(
        buf, format="PNG", save_all=True, append_images=frames[1:], duration=durations, loop=img.info.get("loop", 1)
    )
    return buf.getvalue()


def _bare(img: Image.Image) -> Image.Image:
    # Alpha keeps its look: 254 becomes 255 and 1 becomes 0, as cleaning stealth data does
    if "A" in img.getbands() or "transparency" in img.info:
        img = img.convert("RGBA")
        img.putalpha(img.getchannel("A").point(lambda v: 255 if v >= 254 else 0 if v <= 1 else v & 0xFE))
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # A new image carries none of the source's info, which Pillow would otherwise write back
    return Image.frombytes(img.mode, img.size, img.tobytes())
