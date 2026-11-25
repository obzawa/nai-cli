"""Copies for sharing: animated strip, redaction and visible watermarks."""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import run
from PIL import Image, ImageDraw, PngImagePlugin
from samples import nai_image

from nai_cli import cli, config, errors, fonts, images, inpaint, share, sheet


def _animation(path, frames=3):
    pictures = [Image.new("RGBA", (16, 16), (40 * i, 0, 0, 255)) for i in range(frames)]
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", '{"prompt": "secret"}')
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=[50, 60, 70], loop=2, pnginfo=info)


def test_strip_keeps_an_animation_and_cleans_each_frame(capsys, tmp_path):
    source = tmp_path / "anim.png"
    _animation(source)
    code, out = run(capsys, "strip", str(source))
    copy = Path(out["files"][0]["file"])
    with Image.open(copy) as img:
        assert code == 0 and img.n_frames == 3 and img.info.get("loop") == 2 and "Comment" not in img.info
        durations = []
        for i in range(img.n_frames):
            img.seek(i)
            durations.append(img.info["duration"])
            # The low alpha bit, where stealth settings hide, is cleared
            assert np.asarray(img.convert("RGBA"))[..., 3].min() == 255
        assert durations == [50, 60, 70]
    assert b"secret" not in copy.read_bytes()


def test_still_edits_refuse_animations(capsys, tmp_path):
    source = tmp_path / "anim.png"
    _animation(source)
    code, out = run(capsys, "redact", str(source), "--region", "rect:0,0,1,1")
    assert code == 1 and out["code"] == "bad_args" and "animated" in out["message"]


def _pattern(size=(64, 48)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 2):
        draw.line([(x, 0), (x, size[1])], fill="black")
    return img


@pytest.mark.parametrize("effect", share.EFFECTS)
def test_redaction_changes_only_the_regions(effect):
    img = _pattern()
    region = inpaint.parse_region("rect:0,0,0.5,1")
    out = np.asarray(share.redact(img, share.Redaction((region,), effect=effect)).convert("RGB"))
    source = np.asarray(img)
    assert (out[:, 40:] == source[:, 40:]).all()
    assert (out[:, :30] != source[:, :30]).any()
    inverted = np.asarray(share.redact(img, share.Redaction((region,), effect=effect, invert=True)).convert("RGB"))
    assert (inverted[:, :30] == source[:, :30]).all() and (inverted[:, 40:] != source[:, 40:]).any()


def test_solid_redaction_uses_the_colour_and_opacity():
    region = inpaint.parse_region("rect:0,0,1,1")
    out = share.redact(Image.new("RGB", (8, 8), "white"), share.Redaction((region,), "solid", color=(255, 0, 0)))
    assert out.convert("RGB").getpixel((3, 3)) == (255, 0, 0)
    half = share.redact(Image.new("RGB", (8, 8), "white"), share.Redaction((region,), "solid", opacity=0.5))
    assert half.convert("RGB").getpixel((3, 3)) == (127, 127, 127)
    empty = inpaint.parse_region("-rect:0,0,1,1")
    with pytest.raises(errors.NaiError):
        share.redact(Image.new("RGB", (8, 8)), share.Redaction((empty,), "solid"))


def test_redact_command_names_the_copy_and_drops_settings(capsys, tmp_path):
    source = tmp_path / "pic.png"
    nai_image(source, w=64, h=64)
    code, out = run(capsys, "redact", str(source), "--region", "ellipse:0.2,0.2,0.8,0.8", "--effect", "blur")
    copy = Path(out["files"][0]["file"])
    assert code == 0 and copy.name == "pic_redacted.png" and "Comment" not in Image.open(copy).info
    code, out = run(capsys, "redact", str(source), "--region", "rect:0,0,1,1", "--keep-metadata")
    copy = Path(out["files"][0]["file"])
    assert copy.name == "pic_redacted-1.png" and "Comment" in Image.open(copy).info
    code, out = run(capsys, "redact", str(source), "--region", "rect:0,0,1,1", "--color", "red")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "redact", str(source), "--region", "rect:0,0,1,1", "--amount", "0")
    assert code == 1 and out["code"] == "bad_args"


def _changed(before, after):
    diff = np.abs(np.asarray(after.convert("RGB"), dtype=int) - np.asarray(before.convert("RGB"), dtype=int)).sum(2)
    ys, xs = np.nonzero(diff)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_text_watermark_sits_at_its_anchor():
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    out = share.watermark(img, share.Watermark("NovelAI"))
    left, top, right, bottom = _changed(img, out)
    # Bottom right inside the 3.5% margin, give or take the shadow
    assert right > 340 and bottom > 250 and left > 150 and top > 200
    moved = share.watermark(img, share.Watermark("NovelAI", text_layer=share.Layer("top-left", 0.1, 0.035, 0.82)))
    left, top, right, bottom = _changed(img, moved)
    assert left < 40 and top < 40


def test_long_text_shrinks_to_fit():
    img = Image.new("RGB", (200, 200), "black")
    out = share.watermark(img, share.Watermark("a very long watermark text " * 4))
    left, _, right, _ = _changed(img, out)
    assert left >= 0 and right < 200


def test_auto_place_moves_off_busy_areas():
    img = Image.new("RGB", (300, 300), "white")
    busy = _pattern((150, 150))
    img.paste(busy, (150, 150))
    mark = share.Watermark("mark", auto_place=True)
    _, _, right, _ = _changed(img, share.watermark(img, mark))
    # The bottom right is busy, so the text mirrors to the calm bottom left
    assert right < 150


def test_auto_tone_darkens_on_light_backgrounds():
    img = Image.new("RGB", (300, 200), "white")
    out = np.asarray(share.watermark(img, share.Watermark("M", auto_tone=True)).convert("L"))
    # A dark grey at 82% opacity over white
    assert out.min() < 128


def test_logo_and_text_together(capsys, tmp_path):
    source, logo = tmp_path / "pic.png", tmp_path / "logo.png"
    Image.new("RGB", (200, 200), "gray").save(source)
    Image.new("RGBA", (20, 10), (255, 0, 0, 255)).save(logo)
    code, out = run(
        capsys, "watermark", str(source), "--logo", str(logo), "--text", "hi", "-o", str(tmp_path / "w.png")
    )
    marked = Image.open(out["files"][0]["file"]).convert("RGB")
    assert code == 0 and marked.getpixel((15, 182))[0] > 150
    # A logo alone writes no text
    code, out = run(capsys, "watermark", str(source), "--logo", str(logo), "--position", "top-left")
    alone = Image.open(out["files"][0]["file"])
    assert code == 0 and Path(out["files"][0]["file"]).name == "pic_watermarked.png"
    assert np.asarray(alone.convert("RGB"))[:40, :40].std() == 0
    code, out = run(capsys, "watermark", str(source), "--opacity", "2")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "watermark", str(source), "--font", str(tmp_path / "none.ttf"))
    assert code == 1 and out["code"] == "bad_args"


def test_marks_past_the_edge_are_clipped():
    base = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
    share._overlay(base, Image.new("RGBA", (6, 6), (255, 255, 255, 255)), -3, 7)
    pixels = np.asarray(base)[..., 0]
    assert pixels[7:, :3].min() == 255 and pixels[:7].max() == 0


def test_strip_file_keeps_the_pixels_of_a_still_image(tmp_path):
    path = tmp_path / "s.png"
    Image.new("RGB", (4, 4), "blue").save(path)
    data = images.strip_file(path)
    assert Image.open(io.BytesIO(data)).getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("magic", [b"stealth_rgbinfo", b"stealth_rgbcomp", b"stealth_rgbnone"])
def test_strip_clears_stealth_data_in_the_colours(magic):
    width, height = 8, 32
    pixels = np.full((height, width, 3), 200, np.uint8)
    bits = np.unpackbits(np.frombuffer(magic + b"\0\0\0\x08\x01", np.uint8))
    for i, bit in enumerate(bits):
        pixel = i // 3
        pixels[pixel % height, pixel // height, i % 3] |= bit
    data = images.strip_metadata(Image.fromarray(pixels, "RGB"))
    copy = np.asarray(Image.open(io.BytesIO(data)))
    if magic.startswith(b"stealth_rgb") and magic != b"stealth_rgbnone":
        assert not (copy & 1).any() and (copy >= 200).all()
    else:
        assert np.array_equal(copy, pixels)


def test_text_the_font_cannot_draw_is_refused_before_any_copy(capsys, tmp_path, monkeypatch):
    source = tmp_path / "pic.png"
    nai_image(source, w=64, h=64)
    damaged = tmp_path / "damaged.ttf"
    damaged.write_bytes(b"not a font")
    # No installed font draws it either: one is missing, one is damaged
    monkeypatch.setattr(fonts, "offered", lambda text: iter([tmp_path / "none.ttf", damaged]))
    code, out = run(capsys, "watermark", str(source), "--text", "作者 me")
    assert (code, out["code"]) == (1, "bad_args") and "作者" in out["message"] and "--font" in out["message"]
    assert "files" not in out and not any(config.load(config.default_path()).output_dir.rglob("*.png"))
    code, out = run(capsys, "watermark", str(source), "--text", "by me\nline two")
    assert code == 0, out


def test_text_the_built_in_font_draws_never_looks_for_installed_fonts(capsys, tmp_path, monkeypatch):
    source = tmp_path / "pic.png"
    nai_image(source, w=64, h=64)

    def offered(text):
        raise AssertionError("installed fonts searched")
        yield

    monkeypatch.setattr(fonts, "offered", offered)
    code, out = run(capsys, "watermark", str(source), "--text", "by me")
    assert code == 0, out


def test_an_installed_font_draws_what_the_built_in_one_cannot(capsys, tmp_path, monkeypatch):
    font = _system_font("msyh.ttc")
    source = tmp_path / "pic.png"
    nai_image(source, w=256, h=256)
    damaged = tmp_path / "damaged.ttf"
    damaged.write_bytes(b"not a font")
    latin = _system_font("segoeui.ttf")
    tried = []

    def offered(text):
        for path in (damaged, latin, font, _system_font("simsun.ttc")):
            tried.append(path)
            yield path

    assert share.text_font("作者 me", None, offered("作者 me")) == font
    # The first that draws the whole text is taken and the rest are not opened
    assert tried == [damaged, latin, font]
    monkeypatch.setattr(fonts, "offered", offered)
    code, out = run(capsys, "watermark", str(source), "--text", "作者 me", "-o", str(tmp_path / "out.png"))
    assert code == 0, out
    drawn = np.asarray(Image.open(out["files"][0]["file"]).convert("RGB")).astype(int)
    assert np.abs(drawn - np.asarray(Image.open(source).convert("RGB")).astype(int)).max() > 0
    # A font given is used as it is, even when an installed one would draw the text too
    assert share.text_font("by me", latin, offered("by me")) == latin


_FONTS = Path("C:/Windows/Fonts")


def _system_font(name):
    path = _FONTS / name
    if not path.is_file():
        pytest.skip(f"{name} is a Windows font")
    return path


@pytest.mark.parametrize(
    ("font", "text", "drawn"),
    [
        (None, "Hello", True),
        # The built-in font is a small Latin subset that draws accented letters as boxes
        (None, "caf\u00e9", False),
        ("arial.ttf", "cafe\u0301", True),
        ("arial.ttf", "caf\u00e9", True),
        ("arial.ttf", "\u732b", False),
        ("arial.ttf", "\ue000", False),
        ("msyh.ttc", "\u732b\u306e\u7d75", True),
        # A symbol font that draws private use characters still has no CJK
        ("seguisym.ttf", "\u732b", False),
        ("seguiemj.ttf", "\U0001f468\u200d\U0001f469\u200d\U0001f467", True),
        ("seguiemj.ttf", "\u2764\ufe0f", True),
        ("seguiemj.ttf", "\U0001f44d\U0001f3fd", True),
        # A mark with no composed form is checked on its own
        ("arial.ttf", "x\u0301", True),
        ("arial.ttf", "a\u20dd", False),
        (None, "x\u0301", False),
        # Nothing left to draw once joiners and spaces are set aside
        (None, " \u200d ", False),
        (None, "\ufe0f", False),
    ],
)
def test_the_glyph_check_judges_what_the_font_draws(font, text, drawn):
    path = _system_font(font) if font else None
    if drawn:
        share.require_glyphs(text, path)
    else:
        with pytest.raises(errors.NaiError):
            share.require_glyphs(text, path)


def test_without_freetype_the_bitmap_font_marks_ascii(capsys, tmp_path, monkeypatch):
    def bitmap_only(size=None):
        if size is not None:
            raise ImportError("The _imagingft C module is not installed")
        return share.ImageFont.load_default_imagefont()

    def no_truetype(*args, **kwargs):
        raise ImportError("The _imagingft C module is not installed")

    monkeypatch.setattr(share.ImageFont, "load_default", bitmap_only)
    monkeypatch.setattr(share.ImageFont.FreeTypeFont, "__init__", no_truetype)
    source = tmp_path / "pic.png"
    Image.new("RGB", (256, 256), "white").save(source)
    code, out = run(capsys, "watermark", str(source), "--text", "me")
    assert code == 0, out
    with Image.open(out["files"][0]["file"]) as img:
        assert img.convert("L").getextrema()[0] < 255
    code, out = run(capsys, "watermark", str(source), "--text", "\u732b")
    # --font cannot help here, so the message must not send the user to it
    assert (code, out["code"]) == (1, "bad_args") and "--font" not in out["message"], out
    assert "FreeType" in out["message"]
    code, out = run(capsys, "watermark", str(source), "--text", "me", "--font", str(tmp_path / "any.ttf"))
    assert (code, out["code"]) == (1, "bad_args") and "FreeType" in out["message"]


def test_without_freetype_a_grid_sheet_still_labels_every_cell(monkeypatch):
    def bitmap_only(size=None):
        if size is not None:
            raise ImportError("The _imagingft C module is not installed")
        return share.ImageFont.load_default_imagefont()

    monkeypatch.setattr(share.ImageFont, "load_default", bitmap_only)
    picture = sheet.render(sheet.Sheet([[Image.new("RGB", (32, 32)), None]], ["猫", "犬"], [], "prompt", ""))
    assert picture.width > 64


@pytest.mark.parametrize("damaged", [False, True])
def test_explicit_font_never_searches_system_folders(tmp_path, monkeypatch, damaged):
    path = tmp_path / "custom.ttf"
    if damaged:
        path.write_bytes(b"not a font")
    monkeypatch.setattr(os, "walk", lambda *args, **kwargs: pytest.fail("searched outside the supplied path"))
    with pytest.raises(errors.NaiError) as err:
        share.load_font(path, 12)
    assert err.value.code == errors.BAD_ARGS


def test_strip_removes_settings_exif_and_stealth(capsys, tmp_path):
    img = tmp_path / "pic.png"
    nai_image(img, w=64, h=64)
    rgba = Image.open(img).convert("RGBA")
    rgba.putalpha(Image.new("L", (64, 64), 254))
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "{}")
    exif = Image.Exif()
    exif[0x010E] = "settings"
    rgba.save(img, pnginfo=info, exif=exif.tobytes())
    code, out = run(capsys, "strip", str(img))
    clean = Path(out["files"][0]["file"])
    assert code == 0 and clean.name == "pic_clean.png"
    with Image.open(clean) as copy:
        assert not copy.text and "exif" not in copy.info and "icc_profile" not in copy.info
        assert copy.getchannel("A").getextrema() == (255, 255)
    code, out = run(capsys, "meta", str(clean))
    assert out["code"] == "no_metadata"
    # A name nai-cli chose is numbered when taken; the copy before stays as it was
    before = clean.read_bytes()
    code, out = run(capsys, "strip", str(img))
    assert code == 0 and Path(out["files"][0]["file"]).name == "pic_clean-1.png" and clean.read_bytes() == before
    code, out = run(capsys, "strip", str(img), "-o", str(clean))
    assert code == 0 and Path(out["files"][0]["file"]).name == "pic_clean-2.png" and clean.read_bytes() == before
    code, out = run(capsys, "strip", str(img), str(clean), "-o", str(tmp_path / "one.png"))
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "strip", str(img), "-o", str(tmp_path / "shared"))
    assert code == 0 and Path(out["files"][0]["file"]) == tmp_path / "shared" / "pic.png"


@pytest.mark.parametrize("command", ["strip", "redact", "watermark"])
def test_copy_help_names_where_copies_go(capsys, command):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert "in the output folder" in shown and "beside each image" not in shown
