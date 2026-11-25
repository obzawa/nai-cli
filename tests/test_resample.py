"""Resize and precise reference digests come from the official client's bundled code on the same inputs."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from nai_cli import images, resample


def _pattern(w: int, h: int, alpha: bool) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 0] = (x * 37 + y * 11) % 256
    img[..., 1] = (x * 5 + y * 23) % 256
    img[..., 2] = ((x ^ y) * 7) % 256
    img[..., 3] = ((x * 13 + y * 3) % 256) if alpha else 255
    return img


@pytest.mark.parametrize(
    ("size", "target", "alpha", "digest"),
    [
        ((40, 60), (100, 90), False, "98e294447a9e21344ee502631bb09f3f976967509ae610e2f617fc043a1f80e2"),
        ((90, 70), (33, 21), True, "dd16b115154427a76eff1460f44899186f5d9c5ac29cdd3d662ff9bb28dc1fd5"),
    ],
)
def test_lanczos_matches_official_pica(size, target, alpha, digest):
    out = resample._resize_rgba(_pattern(*size, alpha), *target)
    assert hashlib.sha256(out.tobytes()).hexdigest() == digest


def test_nearest_samples_pixel_centers():
    img = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4), "L")
    assert np.asarray(resample.nearest(img, 2, 2)).tolist() == [[5, 7], [13, 15]]
    assert np.asarray(resample.nearest(img, 8, 1)).tolist() == [[8, 8, 9, 9, 10, 10, 11, 11]]


def test_fill_background_blends_by_alpha():
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (0, 0, 0, 0))
    img.putpixel((1, 0), (100, 0, 200, 128))
    out = resample.fill_background(img, (255, 255, 255))
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((1, 0)) == (177, 127, 227)


def _stealth(w: int, h: int) -> np.ndarray:
    img = _pattern(w, h, False)
    bits = np.unpackbits(np.frombuffer(b"stealth_pngcomp", dtype=np.uint8))
    for i, bit in enumerate(bits):
        img[i % h, i // h, 3] = 254 + bit
    img[h - 1, w - 1, 3] = 1
    return img


@pytest.mark.parametrize(
    ("source", "digest"),
    [
        # Centred with an odd margin, which the official client rounds up
        (_pattern(37, 53, True), "75f1824c7675ba1a160ab3f6999bffbff464e5b4fc1a4a5588c187917ceceae6"),
        (_stealth(300, 300), "9f70176fa1677bc625a782ffcae1c31aa6ae48a2c723b377f139e70fba213a6a"),
        (_pattern(120, 70, False)[..., :3], "9c362a90f8bc7090e18b477292a05ff904562165c1bd56514d94fa12b421f4de"),
    ],
)
def test_precise_reference_matches_official(source, digest):
    img = Image.fromarray(source, "RGBA" if source.shape[2] == 4 else "RGB")
    out = np.asarray(images.precise_reference_canvas(img).convert("RGBA"))
    assert hashlib.sha256(out.tobytes()).hexdigest() == digest


@pytest.mark.parametrize(
    ("source", "size", "digest"),
    [
        (_pattern(37, 53, True), (832, 1216), "c0b49080db925dbf65ce13c6ef760b6728aebfced2befec87f05dfd28046dd81"),
        (_stealth(300, 300), (1024, 1024), "332d8812d58def8288e5a0b05d371879d5784b553e4ae40bc882a0c19e1ef7ff"),
        (
            _pattern(120, 70, False)[..., :3],
            (1344, 768),
            "efdeddf6cdeb101c3db026fbf115d7343b06ccecdee64c9ef9b203c433529766",
        ),
        # Already the bucket's width, so only cropped
        (_pattern(1024, 1050, True), (1024, 1024), "84fd03a37d22d34ece915f2bcf308051169af349c0bd4c46ca4551f8112939e2"),
        (
            _pattern(50, 900, False)[..., :3],
            (256, 4096),
            "15fd8014a92ee2865fae998081d52e6f714825921055b84cb78477b702ec318a",
        ),
        # A bucket's exact size is sent as it is
        (_pattern(832, 1216, True), (832, 1216), "7d694e45760eb84a528abbae3d0b39e54861fbf88b158449d7e3f1d3daa80460"),
    ],
)
def test_bucketed_reference_matches_official(source, size, digest):
    img = Image.fromarray(source, "RGBA" if source.shape[2] == 4 else "RGB")
    out = np.asarray(images.bucketed_reference(img).convert("RGBA"))
    assert (out.shape[1], out.shape[0]) == size
    assert hashlib.sha256(out.tobytes()).hexdigest() == digest
