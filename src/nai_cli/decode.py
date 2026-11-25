"""Image bytes from files and replies: whether they decode, their size, PNG chunks and bounded inflating."""

import io
import struct
import zlib
from collections.abc import Iterator

from PIL import Image

# What Pillow raises for a damaged, truncated or oversized image
BAD_IMAGE = (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STEALTH_COMPRESSED = b"stealth_pngcomp"
# The official reader skips stealth data in larger images
STEALTH_MAX_PIXELS = 0x1000000
TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt")
# Settings and tokenizer files are a few megabytes; anything inflating past this is not one
MAX_INFLATED = 64 * 1024 * 1024
# zlib's wbits for each container
ZLIB, GZIP, RAW = zlib.MAX_WBITS, 16 + zlib.MAX_WBITS, -zlib.MAX_WBITS
# Browsers read Latin-1 as windows-1252, and so does the official client for tEXt; its unassigned bytes stay as they are
_WINDOWS_1252 = {b: bytes([b]).decode("cp1252") for b in range(0x80, 0xA0) if b not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)}


def latin1_text(data: bytes) -> str:
    """The text of a tEXt or zTXt chunk, which PNG keeps in Latin-1."""
    return data.decode("latin-1").translate(_WINDOWS_1252)


def inflate(data: bytes, wbits: int = ZLIB, limit: int = MAX_INFLATED) -> bytes:
    """ValueError for data that is damaged or inflates past limit, so a small file cannot fill the memory."""
    inflater = zlib.decompressobj(wbits)
    try:
        out = inflater.decompress(data, limit)
    except zlib.error as e:
        raise ValueError(f"damaged compressed data: {e}") from e
    if inflater.unconsumed_tail:
        raise ValueError(f"compressed data inflates past {limit} bytes")
    return out


def png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes, int, int]]:
    """Yield (type, data, start, end) per chunk, spans including length and CRC; nothing for non-PNG data."""
    if data[:8] != PNG_SIGNATURE:
        return
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        end = i + 12 + length
        # A chunk cut short ends the file there
        if end > len(data):
            return
        yield kind, data[i + 8 : i + 8 + length], i, end
        if kind == b"IEND":
            return
        i = end


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def itxt(key: str, text: str) -> bytes:
    """An uncompressed iTXt chunk, which holds UTF-8 where tEXt holds only Latin-1. ValueError for a keyword PNG
    does not allow or text that is not Unicode."""
    if not valid_keyword(key):
        raise ValueError(f"{key!r} is not a PNG keyword")
    try:
        body = text.encode()
    except UnicodeEncodeError as e:
        raise ValueError(f"the text of {key} is not valid Unicode") from e
    return png_chunk(b"iTXt", key.encode("latin1") + b"\0" * 5 + body)


def valid_keyword(key: str) -> bool:
    """1 to 79 printable Latin-1 characters, without spaces at either end or two together, as PNG requires."""
    return (
        0 < len(key) <= 79
        and all(" " <= c <= "~" or "\xa1" <= c <= "\xff" for c in key)
        and key == key.strip(" ")
        and "  " not in key
    )


def size_of_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def verify(data: bytes) -> None:
    """Decode every pixel; opening an image reads only its header, which a truncated file still has."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
