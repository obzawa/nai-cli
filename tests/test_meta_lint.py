import gzip
import json
import struct
import zlib
from pathlib import Path

import pytest
from helpers import run, write_defaults
from PIL import Image, PngImagePlugin
from samples import nai_image, v45_image, with_comment

from nai_cli import cli, config, decode, errors, ledger, lint, meta, models, presets, request, rerun


def _comment(prompt="1girl, very aesthetic, masterpiece, no text", uc="lowres, x"):
    return {
        "prompt": prompt,
        "uc": uc,
        "seed": 5,
        "width": 64,
        "height": 64,
        "steps": 28,
        "scale": 4.0,
        "sampler": "k_euler_ancestral",
        "v4_prompt": {
            "caption": {
                "base_caption": prompt,
                "char_captions": [{"char_caption": "girl", "centers": [{"x": 0.5, "y": 0.5}]}],
            },
            "use_coords": False,
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": uc,
                "char_captions": [{"char_caption": "bad", "centers": [{"x": 0.5, "y": 0.5}]}],
            }
        },
        "tag_hint_qt": 1,
        "tag_hint_uc_preset": 0,
    }


def test_reads_text_chunks(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", "NovelAI Diffusion V5 0ADF9AB7")
    info.add_text("Comment", json.dumps(_comment()))
    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64)).save(path, pnginfo=info)
    record = meta.parse(path)
    assert record["user_prompt"] == "1girl"
    assert record["quality"] == "standard" and record["uc_preset"] == "none"
    assert record["characters"] == [{"prompt": "girl", "negative": "bad", "x": 0.5, "y": 0.5}]
    assert record["model"] == "nai-diffusion-5-full"
    args = rerun.to_gen_args(record)
    assert args[0] == "--prompt=1girl" and "--char-neg=bad" in args and "--model=nai-diffusion-5-full" in args


def _raw(
    negative: str, uc_hint: int = 2, prompt: str = "1girl, very aesthetic, masterpiece, no text"
) -> dict[str, str]:
    comment = _comment(prompt=prompt, uc=negative)
    comment["tag_hint_uc_preset"] = uc_hint
    return {"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V5 0ADF9AB7"}


def test_negative_without_nsfw_uses_default_rule():
    heavy = presets.uc_preset_text(models.resolve("v5"), "heavy")
    record = meta.parse_raw(_raw(f"{heavy}, x"))
    assert record["uc_preset"] == "heavy" and record["user_negative"] == "x"
    assert not record["negative_verbatim"] and not record["official_nsfw"]
    args = rerun.to_gen_args(record)
    assert "--neg=x" in args and "--no-official-nsfw" in args


def test_negative_composed_by_official_rules_is_split():
    heavy = presets.uc_preset_text(models.resolve("v5"), "heavy")
    record = meta.parse_raw(_raw(f"nsfw, {heavy}, x"))
    assert record["uc_preset"] == "heavy" and record["user_negative"] == "x" and record["official_nsfw"]
    args = rerun.to_gen_args(record)
    assert "--neg=x" in args and "--official-nsfw" in args and "--neg-verbatim" not in args


def test_negative_neither_rule_rebuilds_is_kept_verbatim():
    # nsfw in both prompts: the default rule removes it, the official rule treats it as automatic
    record = meta.parse_raw(_raw("nsfw, x", uc_hint=0, prompt="1girl, nsfw, very aesthetic, masterpiece, no text"))
    assert record["negative_verbatim"]
    args = rerun.to_gen_args(record)
    assert "--neg=nsfw, x" in args and "--neg-verbatim" in args


def test_unsplittable_prompt_is_kept_whole():
    # Quality tags recorded but missing from the prompt: send everything verbatim
    record = meta.parse_raw(_raw("", uc_hint=0, prompt="1girl, best quality"))
    assert record["user_prompt"] == "1girl, best quality" and record["quality"] == "none"


def test_model_from_source_follows_official_table():
    assert meta.model_from_source("NovelAI Diffusion V5 93F4BD30") == "nai-diffusion-5-full-medium"
    assert meta.model_from_source("NovelAI Diffusion V5 FFFFFFFF") == "nai-diffusion-5-curated"
    assert meta.model_from_source("NovelAI Diffusion V4.5 1229B44F") == "nai-diffusion-4-5-full"
    assert meta.model_from_source("NovelAI Diffusion V4 37442FCA") == "nai-diffusion-4-full"
    assert meta.model_from_source("NovelAI Diffusion V4 FFFFFFFF") == "nai-diffusion-4-curated-preview"
    # Some V4.5 Curated images name themselves V4
    assert meta.model_from_source("NovelAI Diffusion V4 5AB81C7C") == "nai-diffusion-4-5-curated"
    assert meta.model_from_source("Stable Diffusion XL C1E1DE52") is None


def _embed_stealth(img: Image.Image, payload: dict, declared_bits: int | None = None) -> Image.Image:
    data = gzip.compress(json.dumps(payload).encode())
    length = len(data) * 8 if declared_bits is None else declared_bits
    bits = []
    for byte in b"stealth_pngcomp" + struct.pack(">I", length) + data:
        bits.extend((byte >> (7 - i)) & 1 for i in range(8))
    img = img.convert("RGBA")
    px = img.load()
    w, h = img.size
    assert len(bits) <= w * h
    for n, bit in enumerate(bits):
        x, y = divmod(n, h)
        r, g, b, a = px[x, y]
        px[x, y] = (r, g, b, (a & ~1) | bit)
    return img


def test_reads_stealth_alpha(tmp_path):
    payload = {"Source": "NovelAI Diffusion V5 0ADF9AB7", "Comment": json.dumps(_comment())}
    img = _embed_stealth(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), payload)
    path = tmp_path / "s.png"
    img.save(path)
    record = meta.parse(path)
    assert record["seed"] == 5 and record["user_prompt"] == "1girl"


def test_no_metadata(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(errors.NaiError) as e:
        meta.parse(path)
    assert e.value.code == errors.NO_METADATA


def test_itxt_and_ztxt_chunks():
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"iTXt", b"Comment\x00\x01\x00\x00\x00" + zlib.compress("内容".encode()))
        + chunk(b"zTXt", b"K\x00\x00" + zlib.compress(b"v"))
        + chunk(b"IEND", b"")
    )
    assert meta.png_text_chunks(png) == {"Comment": "内容", "K": "v"}


def test_text_chunks_are_latin1():
    # NovelAI writes text that fits Latin-1 as tEXt in single bytes, a no-break space as 0xA0
    png = (
        decode.PNG_SIGNATURE
        + decode.png_chunk(b"tEXt", b"Description\0caf\xe9\xa0x")
        + decode.png_chunk(b"zTXt", b"K\0\0" + zlib.compress(b"\x80\x81"))
        + decode.png_chunk(b"IEND", b"")
    )
    assert meta.png_text_chunks(png) == {"Description": "caf\xe9\xa0x", "K": "€\x81"}


@pytest.mark.parametrize(
    "text,code",
    [
        ("(blue hair:1.2), smile", "sd_weight"),
        ("[ugly:0.8], smile", "sd_weight"),
        ("1.2::tag, 0.8::b::", "weight_unclosed"),
        ("a，b", "fullwidth_punct"),
        ("long hair（smile）", "fullwidth_punct"),
        ("雨中，rain", "fullwidth_punct"),
        ("{{a}, b", "bracket_unbalanced"),
        ("smile, smile", "duplicate_tag"),
    ],
)
def test_lint_finds(text, code):
    assert code in [i.code for i in lint.check_text(text, "prompt")]


@pytest.mark.parametrize(
    "text",
    [
        "1.2::tag::, 0.8::b::",
        "0.9::artist:example : year 2024, ::",
        "-2::x::, 0::ai-generated::",
        "sign, text:: hello",
        "<artist>a</artist>, <style>b</style>, <artist>c</artist>",
        "一个女孩站在雨中，手里拿着伞。",
        "1girl, 雨中，撑伞",
        '1girl, "你好，世界"',
        "1girl, “（笑）”",
        "sign, text:你好，世界",
        '1girl, "(hello:1.2)"',
    ],
)
def test_lint_accepts(text):
    assert [i.code for i in lint.check_text(text, "prompt") if i.level != "info"] == []


def test_lint_text_marker_prefix_and_preset_overlap():
    v5 = models.resolve("v5")
    heavy = presets.uc_preset_text(v5, "heavy")
    codes = [i.code for i in lint.run(v5, "Text: hi", negative=f"{heavy}, x", uc_preset="heavy")]
    assert "text_marker_prefix" in codes and "uc_preset_duplicated" in codes
    assert lint.check_text_marker_prefix("sign, text: hi", "standard", False) == []
    assert lint.check_text_marker_prefix("text: hi", "none", False) == []
    assert lint.check_text_marker_prefix("text: hi", "none", True) != []
    assert lint.check_uc_preset_overlap(f"{heavy}, x", v5, "heavy", verbatim=True) == []


def test_lint_quality_overlap_and_tokens():
    v5c = models.resolve("v5-curated")
    tokens = lint.TokenCount(prompt=800, negative=10, limit=703, exact=False)
    issues = lint.run(v5c, "masterpiece, 1girl", quality="standard", tokens=tokens)
    codes = [i.code for i in issues]
    assert "quality_duplicated" in codes and "token_limit" in codes


@pytest.mark.parametrize(("name", "limit"), [("v5", 750), ("v5-curated", 374), ("v4.5", 118)])
def test_lint_text_longer_than_documented(name, limit):
    model = models.resolve(name)
    text = f"sign, text: {'a' * limit}"
    assert lint.check_text_length(text, model) == []
    assert [i.code for i in lint.check_text_length(text + "b", model)] == ["text_too_long"]
    # Only the text after text: counts, in any mix segment
    assert lint.check_text_length(f"{'a' * 900}|x, Text: {'b' * (limit + 1)}", model) != []


def test_lint_text_length_counts_trailing_whitespace_and_characters():
    v45 = models.resolve("v4.5")
    # Only the separator after text: is left out
    assert lint.check_text_length("text:  " + "a" * 118, v45) == []
    assert [i.code for i in lint.check_text_length("text: " + "a" * 118 + chr(10), v45)] == ["text_too_long"]
    issues = lint.run(v45, "1girl", characters=[("sign, text: " + "b" * 119, "")])
    assert [(i.code, i.field) for i in issues if i.code == "text_too_long"] == [("text_too_long", "character1")]


def test_lint_text_length_not_documented_for_medium():
    assert lint.check_text_length("text: " + "a" * 2000, models.resolve("v5-medium")) == []


def test_furry_prefix_read_as_furry_mode_like_official():
    prompt = "fur dataset, 1girl, very aesthetic, masterpiece, no text"
    record = meta.parse_raw(_raw("lowres", uc_hint=0, prompt=prompt))
    assert record["furry"] is True and record["user_prompt"] == "1girl"
    args = rerun.to_gen_args(record)
    assert "--furry" in args and "--prompt=1girl" in args
    # V5 Medium has furry mode too, taking every V5 Full feature
    raw = _raw("lowres", uc_hint=0, prompt=prompt)
    raw["Source"] = "NovelAI Diffusion V5 93F4BD30"
    record = meta.parse_raw(raw)
    assert record["furry"] is True and record["user_prompt"] == "1girl"


def test_furry_with_empty_prompt_rebuilds():
    # The official client adds the prefix after the quality tags, so there is no stray comma
    record = meta.parse_raw(_raw("lowres", uc_hint=0, prompt="fur dataset, very aesthetic, masterpiece, no text"))
    assert record["furry"] is True and record["user_prompt"] == "" and record["quality"] == "standard"


def _comment_of(built: request.Built) -> dict:
    p = built.body["parameters"]
    return {**p, "prompt": built.prompt, "uc": built.negative}


def test_variety_read_back():
    v45 = models.resolve("v4.5")
    for variety in (True, False):
        built = request.build(request.GenSpec(model=v45, prompt="1girl", seed=3, variety=variety))
        raw = {"Comment": json.dumps(_comment_of(built)), "Source": "NovelAI Diffusion V4.5 4BDE2A90"}
        record = meta.parse_raw(raw)
        assert record["variety"] is variety and ("--variety" in rerun.to_gen_args(record)) is variety


def test_lint_weight_mistakes():
    codes = lambda t: [i.code for i in lint.check_weights(t, "prompt")]  # noqa: E731
    assert codes("1.2::year_2025::") == ["weight_numeric_close"]
    assert codes("1.5 ::tag::") == ["weight_spaced"]
    assert codes("tag::1.5, x") == ["weight_number_after"]
    assert codes("1.2::a, 0.8::b::") == ["weight_unclosed"]
    assert codes("1.2::year 2025 ::, 2::x::") == []


def test_stealth_length_beyond_the_image_is_ignored(tmp_path):
    payload = {"Source": "NovelAI Diffusion V5 0ADF9AB7", "Comment": json.dumps(_comment())}
    img = _embed_stealth(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), payload, declared_bits=96 * 96)
    assert meta.stealth_text(img) == {}


@pytest.mark.parametrize(
    "header, encoding",
    [(b"ASCII\0\0\0", "utf-8"), (b"\0" * 8, "utf-8"), (b"UNICODE\0\xff\xfe", "utf-16-le"), (b"UNICODE\0", "utf-16-be")],
)
@pytest.mark.parametrize("fmt", ["WEBP", "PNG"])
def test_reads_exif_user_comment_like_official(tmp_path, header, encoding, fmt):
    settings = {"Software": "NovelAI", "Source": "NovelAI Diffusion V5 0ADF9AB7", "Comment": json.dumps(_comment())}
    exif = Image.Exif()
    exif[0x8769] = {0x9286: header + json.dumps(settings).encode(encoding) + "\0".encode(encoding)}
    path = tmp_path / f"s.{fmt.lower()}"
    Image.new("RGB", (64, 64)).save(path, fmt, exif=exif.tobytes())
    record = meta.parse(path)
    assert record["seed"] == 5 and record["user_prompt"] == "1girl"


@pytest.mark.parametrize(
    "size, sent, clamped",
    [
        # On the grid, as the official client sends an imported size
        ((1500, 1000), "1472x1024", False),
        ((1000, 1500), "1024x1472", False),
        # The official client would send 1472x2176 with no images; the longer side is clamped as its size box does
        ((1467, 2144), "1472x2112", True),
        ((2144, 1467), "2112x1472", True),
        ((1800, 1800), "1728x1792", True),
    ],
)
def test_gen_args_align_the_size_like_official(size, sent, clamped):
    comment = _comment()
    comment.update(width=size[0], height=size[1])
    record = meta.parse_raw({"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V5 0ADF9AB7"})
    assert f"--size={sent}" in rerun.to_gen_args(record)
    w, h = map(int, sent.split("x"))
    assert w * h <= 3145728 and w % 64 == 0 and h % 64 == 0
    assert any("pixel limit" in gap for gap in rerun.rerun_gaps(record)) is clamped


@pytest.mark.parametrize("strengths, normalized", [([0.5, 0.5], True), ([0.8, 0.8], False), ([0.2, 0.2], None)])
def test_vibe_normalization_guessed_like_official_import(strengths, normalized):
    comment = _comment()
    comment["reference_strength_multiple"] = strengths
    record = meta.parse_raw({"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V4.5 B5A2A797"})
    assert record["normalize_vibe_strength"] is normalized
    assert ("--no-normalize-vibe-strength" in rerun.to_gen_args(record)) is (normalized is False)


@pytest.mark.parametrize(
    "straight, flag", [(True, "--no-premultiplied-alpha"), (False, "--premultiplied-alpha"), (None, None)]
)
def test_alpha_mode_rerun(straight, flag):
    comment = _comment()
    if straight is not None:
        comment["straight_alpha"] = straight
    record = meta.parse_raw({"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V5 0ADF9AB7"})
    assert [a for a in rerun.to_gen_args(record) if "premultiplied" in a] == ([flag] if flag else [])


def test_vibes_only_carried_for_models_that_have_them():
    comment = _comment()
    comment.update(reference_image_multiple=["E"], reference_strength_multiple=[0.6])
    v45 = meta.parse_raw({"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V4.5 4BDE2A90"})
    assert "--vibes-from=" in " ".join(rerun.to_gen_args(v45, "made.png")) and rerun.rerun_gaps(v45) == []
    v5 = meta.parse_raw({"Comment": json.dumps(comment), "Source": "NovelAI Diffusion V5 0ADF9AB7"})
    assert not any(a.startswith("--vibes-from") for a in rerun.to_gen_args(v5, "made.png"))
    assert any("1 Vibes" in gap for gap in rerun.rerun_gaps(v5))


def test_sd_weight_message_quotes_what_was_typed():
    [issue] = [i for i in lint.check_text("[ugly:0.8]", "prompt") if i.code == "sd_weight"]
    assert "[ugly:0.8]" in issue.message


def _weird_image(path):
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", json.dumps({"v4_prompt": ["not", "a", "caption"], "seed": 1}))
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)


def test_settings_of_another_shape_are_no_metadata(capsys, tmp_path):
    weird = tmp_path / "weird.png"
    _weird_image(weird)
    code, out = run(capsys, "meta", str(weird))
    assert code == 1 and out["code"] == "no_metadata"


def test_meta_of_several_images_prints_lines(capsys, tmp_path):
    folder = tmp_path / "pics"
    (folder / "day").mkdir(parents=True)
    nai_image(folder / "a.png", seed=1)
    nai_image(folder / "day" / "b.png", seed=2)
    _weird_image(folder / "c.png")
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    code = cli.main(["meta", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and [line["type"] for line in lines] == ["image", "image", "summary"]
    assert lines[0]["seed"] == 1 and lines[0]["image"].endswith("a.png") and lines[1]["code"] == "no_metadata"
    assert lines[-1]["meta"] == {"images": 2, "without_settings": 1, "damaged": 0, "failed": 0}
    code = cli.main(["meta", str(folder / "a.png"), str(folder / "gone.png")])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1 and lines[1]["code"] == "bad_args" and lines[-1]["meta"]["failed"] == 1
    code = cli.main(["meta", "-r", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(line.get("seed") for line in lines[:-1] if line["ok"]) == [1, 2]
    # One image prints one object unless --jsonl asks for lines
    code = cli.main(["meta", "--jsonl", str(folder / "a.png")])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and len(lines) == 2 and lines[-1]["meta"]["images"] == 1


def test_meta_lines_cannot_run_inside_a_batch(capsys, tmp_path):
    nai_image(tmp_path / "a.png")
    nai_image(tmp_path / "b.png")
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(["meta", str(tmp_path / "a.png"), str(tmp_path / "b.png")]), encoding="utf-8")
    cli.main(["batch", str(jobs)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["type"] == "job" and lines[0]["code"] == "bad_args" and lines[-1]["type"] == "summary"


def test_text_that_inflates_past_the_limit_is_skipped():
    bomb = zlib.compress(b"\0" * (decode.MAX_INFLATED + 1), 9)
    data = (
        decode.PNG_SIGNATURE
        + decode.png_chunk(b"zTXt", b"Comment\0\0" + bomb)
        + decode.png_chunk(b"tEXt", b"Title\0ok")
    )
    assert meta.png_text_chunks(data) == {"Title": "ok"}
    with pytest.raises(ValueError):
        decode.inflate(gzip.compress(b"x" * 100), decode.GZIP, limit=10)


def test_a_chunk_cut_short_ends_the_chunks():
    whole = decode.png_chunk(b"tEXt", b"Title\0ok")
    data = decode.PNG_SIGNATURE + whole + decode.png_chunk(b"tEXt", b"Comment\0" + b"x" * 50)[:-20]
    assert [kind for kind, *_ in decode.png_chunks(data)] == [b"tEXt"]


@pytest.mark.parametrize("kept_in", ["exif", "stealth"])
@pytest.mark.parametrize("command", [["redact", "--region", "rect:0,0,1,1"], ["watermark", "--text", "me"]])
def test_kept_settings_survive_an_edit_wherever_the_source_held_them(capsys, tmp_path, kept_in, command):
    settings = {"Source": "NovelAI Diffusion V5 0ADF9AB7", "Comment": json.dumps(_comment())}
    if kept_in == "exif":
        exif = Image.Exif()
        exif[0x8769] = {0x9286: b"UNICODE\0" + json.dumps(settings).encode("utf-16-be")}
        source = tmp_path / "pic.webp"
        Image.new("RGB", (96, 96)).save(source, "WEBP", exif=exif.tobytes())
    else:
        source = tmp_path / "pic.png"
        _embed_stealth(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), settings).save(source)
    name, *options = command
    code, out = run(capsys, name, str(source), *options, "--keep-metadata")
    assert code == 0, out
    copy = out["files"][0]["file"]
    assert meta.png_text_chunks(Path(copy).read_bytes())["Source"] == settings["Source"]
    assert meta.parse(copy)["seed"] == meta.parse(source)["seed"]


@pytest.mark.parametrize("command", [["redact", "--region", "rect:0,0,1,1"], ["watermark", "--text", "me"]])
def test_a_copy_reads_back_as_its_source_did(capsys, tmp_path, command):
    # PNG text naming another model, beside settings in the alpha channel that win as they do on the source
    settings = {"Source": "NovelAI Diffusion V5 0ADF9AB7", "Comment": json.dumps(_comment())}
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", "NovelAI Diffusion V4.5 4BDE2A90")
    info.add_text("Title", "kept as it is")
    source = tmp_path / "pic.png"
    _embed_stealth(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), settings).save(source, pnginfo=info)
    name, *options = command
    code, out = run(capsys, name, str(source), *options, "--keep-metadata")
    assert code == 0, out
    copy = out["files"][0]["file"]
    for read in (
        lambda p: meta.parse(p)["model"],
        lambda p: run(capsys, "gen", "--from-image", str(p), "--dry-run")[1]["model"],
    ):
        assert read(copy) == read(source) == "nai-diffusion-5-full"
    assert meta.png_text_chunks(Path(copy).read_bytes())["Title"] == "kept as it is"


@pytest.mark.parametrize("command", [["redact", "--region", "rect:0,0,1,1"], ["watermark", "--text", "me"]])
def test_keys_png_text_cannot_hold_are_left_out_with_a_warning(capsys, tmp_path, command):
    settings = {"Comment": json.dumps(_comment()), "中文": "a", "": "b", " padded": "c", "x" * 80: "d"}
    source = tmp_path / "pic.png"
    _embed_stealth(Image.new("RGBA", (96, 96), (10, 20, 30, 255)), settings).save(source)
    name, *options = command
    code, out = run(capsys, name, str(source), *options, "--keep-metadata")
    assert code == 0 and len(out["warnings"]) == 4, out
    kept = meta.png_text_chunks(Path(out["files"][0]["file"]).read_bytes())
    assert set(kept) == {"Comment"} and meta.parse(out["files"][0]["file"])["seed"] == meta.parse(source)["seed"]


def test_a_copy_is_filed_by_its_size(capsys, tmp_path):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + 'folder = "{resolution}"\n', encoding="utf-8")
    source = tmp_path / "pic.png"
    Image.new("RGB", (96, 64)).save(source)
    for argv in (["watermark", str(source), "--text", "me"], ["redact", str(source), "--region", "rect:0,0,1,1"]):
        code, out = run(capsys, *argv, "--keep-metadata")
        assert code == 0 and Path(out["files"][0]["file"]).parent.name == "96x64", out


@pytest.mark.parametrize("streaming", [False, True])
def test_reading_image_settings_does_not_require_a_readable_ledger(capsys, tmp_path, monkeypatch, streaming):
    img = tmp_path / "image.png"
    nai_image(img, seed=77)

    def unreadable():
        raise PermissionError("ledger unavailable")

    monkeypatch.setattr(ledger, "by_file", unreadable)
    code = cli.main(["meta", str(img), *(["--jsonl"] if streaming else [])])
    raw = capsys.readouterr().out
    assert code == 0, raw
    record = json.loads(raw.splitlines()[0] if streaming else raw)
    assert record["seed"] == 77 and record["gen_args"] and "ledger" not in record
    assert any("ledger" in warning for warning in record["warnings"])
    if streaming:
        assert json.loads(raw.splitlines()[-1])["type"] == "summary"


def test_lint_uses_character_positions(capsys):
    code, out = run(
        capsys,
        "lint",
        "--prompt",
        "x",
        "--char",
        'says "B"',
        "--char",
        'says "A"',
        "--char-pos",
        "0.8,0.2",
        "--char-pos",
        "0.2,0.2",
        "--quality",
        "none",
        "--uc-preset",
        "none",
    )
    assert code == 0 and out["prompt"] == "x, teXt: A\n\nB"


def test_meta_gen_args_parse_back(capsys, tmp_path):
    comment = {
        "prompt": "-1::x::, girl",
        "uc": "",
        "seed": 2,
        "width": 832,
        "height": 1216,
        "steps": 28,
        "scale": 4.0,
        "sampler": "k_euler_ancestral",
        "cfg_rescale": 0.1,
        "noise_schedule": "karras",
        "v4_prompt": {
            "caption": {
                "base_caption": "-1::x::, girl",
                "char_captions": [
                    {"char_caption": "a", "centers": [{"x": 0.3, "y": 0.5}]},
                    {"char_caption": "b", "centers": [{"x": 0.7, "y": 0.5}]},
                ],
            },
            "use_coords": True,
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": "",
                "char_captions": [
                    {"char_caption": "", "centers": [{"x": 0.3, "y": 0.5}]},
                    {"char_caption": "bad b", "centers": [{"x": 0.7, "y": 0.5}]},
                ],
            }
        },
        "tag_hint_qt": 0,
        "tag_hint_uc_preset": 0,
    }
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", "NovelAI Diffusion V5 0ADF9AB7")
    info.add_text("Comment", json.dumps(comment))
    img = tmp_path / "m.png"
    Image.new("RGB", (8, 8)).save(img, pnginfo=info)
    code, out = run(capsys, "meta", str(img))
    code, again = run(capsys, "gen", *out["gen_args"], "--dry-run")
    assert code == 0
    assert again["prompt"] == "-1::x::, girl"
    assert again["characters"] == [{"prompt": "a", "negative": ""}, {"prompt": "b", "negative": "bad b"}]
    params = again["body"]["parameters"]
    assert params["seed"] == 2 and params["cfg_rescale"] == 0.1 and params["use_coords"] is True
    assert params["v4_prompt"]["caption"]["char_captions"][1]["centers"] == [{"x": 0.7, "y": 0.5}]


def test_lint_reports_token_counts(capsys):
    code, out = run(capsys, "lint", "--prompt", "1girl", "--char", "boy")
    assert code == 0 and "tokens_estimate" not in out
    assert out["tokens"]["limit"] == 1471 and out["tokens"]["exact"] is False and out["tokens"]["prompt"] > 0


def test_lint_checks_like_gen(capsys, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "lint", "--vibe", str(img))
    assert code == 1 and out["code"] == "model_unsupported"
    code, out = run(capsys, "lint", "--prompt", "x", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_rerun_reports_what_the_image_does_not_keep(capsys, tmp_path):
    img = tmp_path / "made.png"
    v45_image(img, reference_strength_multiple=[0.5], director_reference_strengths=[1.0])
    code, record = run(capsys, "meta", str(img))
    assert any("1 precise references" in w for w in record["warnings"])
    assert any("1 Vibes" in w for w in record["warnings"])
    assert not any(a.startswith("--vibes-from") for a in record["gen_args"])
    code, out = run(capsys, "variations", "--image", str(img), "--dry-run")
    assert code == 0 and any("1 precise references" in w for w in out["warnings"])


def test_gen_args_hold_against_the_config_defaults(capsys, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, prompt="cat\n| dog")
    _, out = run(capsys, "meta", str(img))
    # A default of several images would rerun it at several times the price
    write_defaults('effort = "medium"\ncomfyui_characters = true\nn = 3\n')
    code, again = run(capsys, "gen", *out["gen_args"], "--dry-run")
    assert code == 0 and again["model"] == out["model"] == "nai-diffusion-5-full" and again["characters"] == []
    _, one = run(capsys, "gen", *out["gen_args"], "--n", "1", "--dry-run")
    assert again["body"]["parameters"]["n_samples"] == 1 and again["cost"] == one["cost"]


@pytest.mark.parametrize(
    "field",
    [
        '"steps": true',
        '"steps": 28.5',
        '"scale": "5"',
        '"scale": false',
        '"seed": 1' + "0" * 400,
        '"cfg_rescale": [0]',
        '"v4_prompt": {"caption": {"base_caption": "x", "char_captions": [{"char_caption": "a", "centers": [{"x": true, "y": 0.5}]}]}}',
        '"v4_prompt": {"caption": {"base_caption": "x", "char_captions": [{"char_caption": "a", "centers": [{"x": "0.5", "y": 0.5}]}]}}',
        '"sampler": []',
        '"sampler": {}',
        '"noise_schedule": 1',
        '"request_type": []',
        '"straight_alpha": []',
        '"straight_alpha": "no"',
    ],
)
def test_settings_of_another_type_are_unreadable_in_every_reader(capsys, tmp_path, field):
    folder = tmp_path / "pics"
    folder.mkdir()
    img = with_comment(folder / "odd.png", '{"prompt": "1girl", "uc": "", ' + field + "}")
    code, out = run(capsys, "meta", str(img))
    assert (code, out["code"]) == (1, "no_metadata"), out
    code, out = run(capsys, "gen", "--from-image", str(img), "--dry-run")
    assert code == 1 and out["code"] != "internal", out
    for command in ("enhance", "variations"):
        code, out = run(capsys, command, "--image", str(img), "--dry-run")
        assert out.get("code") != "internal", (command, out)
    code = cli.main(["meta", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and lines[-1]["meta"]["without_settings"] == 1


def test_a_folder_scan_counts_unreadable_images_as_failed(capsys, tmp_path):
    folder = tmp_path / "pics"
    (folder / "nested.png").mkdir(parents=True)
    (folder / "bad.png").write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(folder / "plain.png")
    code = cli.main(["meta", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = lines[-1]
    assert code == 1 and summary["meta"] == {"images": 2, "without_settings": 1, "damaged": 0, "failed": 1}


def test_settings_of_another_shape_are_unreadable_settings_in_every_reader(capsys, tmp_path):
    folder = tmp_path / "dir"
    folder.mkdir()
    shapes = {
        "nan.png": '{"prompt": "1girl", "uc": "", "scale": NaN}',
        "overflow.png": '{"prompt": "1girl", "uc": "", "scale": 1e309}',
        "huge.png": '{"prompt": "1girl", "uc": "", "seed": 1' + "0" * 400 + "}",
        "number.png": '{"prompt": 123, "uc": ""}',
        "caption.png": '{"prompt": "a", "uc": "", "v4_prompt": "x"}',
    }
    for name, comment in shapes.items():
        with_comment(folder / name, comment)
    code = cli.main(["meta", str(folder)])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and lines[-1]["meta"] == {"images": 5, "without_settings": 5, "damaged": 0, "failed": 0}
    assert all(line["type"] == "image" and line["code"] == "no_metadata" for line in lines[:-1])
    for name in shapes:
        code, out = run(capsys, "gen", "--from-image", str(folder / name), "--dry-run")
        assert out.get("code") == "no_metadata", (name, out)
    drawn = with_comment(tmp_path / "drawn.png", '{"prompt": "a", "uc": "", "actual_prompts": {"prompt": "x"}}')
    code, out = run(capsys, "gen", "--from-image", str(drawn), "--actual-prompts", "--dry-run")
    assert out.get("code") == "no_metadata", out


def _cut_off(path):
    """The text and a little of the pixels, as an export stopped part way leaves a PNG."""
    data = path.read_bytes()
    at = data.index(b"IDAT")
    path.write_bytes(data[: at + 200])


def test_a_damaged_image_gives_its_settings_but_never_passes_as_good(capsys, tmp_path):
    good, damaged = tmp_path / "good.png", tmp_path / "damaged.png"
    nai_image(good, w=256, h=256)
    nai_image(damaged, w=256, h=256)
    _cut_off(damaged)
    code, out = run(capsys, "meta", str(good))
    assert code == 0 and out["image_ok"] is True
    code, out = run(capsys, "meta", str(damaged))
    assert (code, out["ok"], out["image_ok"]) == (1, False, False) and "damaged" in out["message"]
    assert out["seed"] == 5 and out["gen_args"] and any("damaged" in w for w in out["warnings"]), out
    for argv in (["meta", str(good), str(damaged)], ["meta", str(tmp_path)], ["meta", "--jsonl", str(damaged)]):
        code = cli.main(argv)
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        flagged = [line for line in lines if line.get("image_ok") is False]
        assert code == 1 and len(flagged) == 1 and flagged[0]["gen_args"], argv
        assert lines[-1]["meta"]["damaged"] == 1 and lines[-1]["meta"]["failed"] == 0, argv


def test_an_image_past_the_pixel_limit_is_unchecked_not_damaged(capsys, tmp_path, monkeypatch):
    big = tmp_path / "big.png"
    nai_image(big, w=256, h=256)
    # Pillow refuses outright past twice its limit
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 256 * 256 // 3)
    code, out = run(capsys, "meta", str(big))
    assert code == 0 and out["image_ok"] is None and any("pixel limit" in w for w in out["warnings"]), out
    code = cli.main(["meta", str(tmp_path)])
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])["meta"]
    assert code == 0 and (summary["damaged"], summary["failed"]) == (0, 0), summary
