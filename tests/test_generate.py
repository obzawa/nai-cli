"""Generation options turned into the request: models, samplers, tidying, fixed tags, prices and image modes."""

import io
import json

import pytest
from helpers import run, write_defaults
from PIL import Image
from samples import bomb_png, nai_image

from nai_cli import cli, images, presets


def test_cost_counts_references(capsys):
    _, base = run(capsys, "cost", "--model", "v4.5", "--size", "1024x1024")
    assert base["cost"]["estimate"] == 0
    _, out = run(capsys, "cost", "--model", "v4.5", "--size", "1024x1024", "--vibes", "5", "--encode", "1")
    assert out["cost"]["estimate"] == 2 + 2
    # Precise references drop the Vibe and are charged on top of the free image
    _, out = run(
        capsys, "cost", "--model", "v4.5", "--size", "1024x1024", "--refs", "1", "--vibes", "5", "--encode", "1"
    )
    assert out["cost"]["estimate"] == 5
    _, out = run(capsys, "cost", "--model", "v5-curated", "--inpaint", "--size", "1024x1024", "--steps", "28")
    assert out["model"] == "nai-diffusion-4-5-curated-inpainting" and out["cost"]["estimate"] == 0
    _, out = run(capsys, "cost", "--size", "1024x1024", "--n", "5")
    assert out["code"] == "bad_args"


def test_tidy_options(capsys):
    argv = ["gen", "--prompt", "1girl，  1.2::year 2025::", "--quality", "none", "--dry-run"]
    code, out = run(capsys, *argv)
    assert code == 0 and out["prompt"] == "1girl，  1.2::year 2025::"
    assert "weight_numeric_close" in [i["code"] for i in out["lint"]]
    code, out = run(capsys, *argv, "--format-prompt", "--space-numeric-close")
    assert out["prompt"] == "1girl, 1.2::year 2025 ::"
    write_defaults("format_prompt = true\n")
    code, out = run(capsys, *argv, "--no-format-prompt")
    assert out["prompt"] == "1girl，  1.2::year 2025::"


def test_gen_furry_mode(capsys):
    code, out = run(capsys, "gen", "--prompt", "wolf", "--furry", "--dry-run")
    assert code == 0 and out["prompt"].startswith("fur dataset, wolf, ")
    # V5 Medium takes every V5 Full feature, furry mode included
    code, out = run(capsys, "gen", "--prompt", "wolf", "--furry", "--model", "v5-medium", "--dry-run")
    assert code == 0 and out["prompt"].startswith("fur dataset, wolf")


def test_ddim_is_refused_wherever_a_sampler_is_chosen(capsys):
    for sampler in ("ddim", "ddim_v3"):
        code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--sampler", sampler, "--dry-run")
        assert (code, out["code"]) == (1, "bad_args") and "k_euler" in out["choices"], out
    code = cli.main(["grid", "--dry-run", "--x", "sampler=k_euler,ddim", "gen", "--model", "v4.5", "--prompt", "x"])
    assert code == 1 and json.loads(capsys.readouterr().out.splitlines()[0])["code"] == "bad_args"
    write_defaults('sampler = "ddim"\n')
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--dry-run")
    assert (code, out["code"]) == (1, "bad_args")


@pytest.mark.parametrize(
    "argv",
    [
        ["--refs", "-1"],
        ["--model", "v4.5", "--vibes", "2", "--encode", "3"],
        ["--steps", "0"],
        ["--strength", "0"],
        ["--strength", "nan"],
    ],
)
def test_cost_refuses_impossible_counts(capsys, argv):
    code, out = run(capsys, "cost", *argv)
    assert code == 1 and out["code"] == "bad_args"


def test_cfg_above_ten_allowed(capsys):
    code, out = run(capsys, "gen", "--cfg", "12", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["scale"] == 12


def test_cost_refuses_unsupported_references(capsys):
    code, out = run(capsys, "cost", "--model", "v4", "--refs", "2")
    assert code == 1 and out["code"] == "model_unsupported" and out["feature"] == "precise_reference"
    code, out = run(capsys, "cost", "--model", "v5", "--vibes", "1")
    assert code == 1 and out["feature"] == "vibe"
    code, out = run(capsys, "cost", "--model", "v5", "--inpaint", "--refs", "1")
    assert code == 1 and out["feature"] == "precise_reference"


def test_v5_references_are_priced_and_raise_medium_effort(capsys):
    code, plain = run(capsys, "cost", "--model", "v5", "--size", "832x1216")
    code, out = run(capsys, "cost", "--model", "v5", "--size", "832x1216", "--refs", "2")
    assert code == 0 and out["cost"]["estimate"] == plain["cost"]["estimate"] + 10
    code, out = run(capsys, "cost", "--model", "v5-medium", "--refs", "1")
    assert code == 0 and out["model"] == "nai-diffusion-5-full"
    assert any("high effort" in w for w in out["warnings"])


def test_oversized_image_is_bad_args(capsys, tmp_path):
    img = tmp_path / "huge.png"
    img.write_bytes(bomb_png())
    code, out = run(capsys, "i2i", "--image", str(img), "--prompt", "x", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "meta", str(img))
    assert code == 1 and out["code"] == "bad_args"


def test_premultiplied_alpha_switch(capsys):
    argv = ["gen", "--prompt", "x", "--dry-run"]
    code, out = run(capsys, *argv)
    assert code == 0 and out["body"]["parameters"]["straight_alpha"] is True
    code, out = run(capsys, *argv, "--premultiplied-alpha")
    assert out["body"]["parameters"]["straight_alpha"] is False
    assert not any("premultiplied" in w for w in out["warnings"])
    write_defaults("premultiplied_alpha = true\n")
    code, out = run(capsys, *argv)
    assert out["body"]["parameters"]["straight_alpha"] is False
    code, out = run(capsys, *argv, "--no-premultiplied-alpha")
    assert out["body"]["parameters"]["straight_alpha"] is True
    # V4.5 has no transparency, so nothing is sent
    code, out = run(capsys, *argv, "--model", "v4.5", "--premultiplied-alpha")
    assert code == 0 and "straight_alpha" not in out["body"]["parameters"]
    assert any("premultiplied" in w for w in out["warnings"])


def test_fixed_tags_join_typed_prompts_only(capsys, tmp_path):
    write_defaults('prompt_prefix = "artist:x"\nprompt_suffix = "best"\nnegative_prefix = "blurry"\nquality = "none"\n')
    _, out = run(capsys, "gen", "--prompt", "cat", "--neg", "", "--dry-run")
    assert out["prompt"] == "artist:x, cat, best" and out["negative"].endswith("blurry")
    _, out = run(capsys, "gen", "--prompt", "cat", "--no-fixed-tags", "--dry-run")
    assert out["prompt"] == "cat"
    img = tmp_path / "pic.png"
    nai_image(img, prompt="artist:x, cat, best")
    _, record = run(capsys, "meta", str(img))
    assert "--no-fixed-tags" in record["gen_args"]
    _, out = run(capsys, "gen", *record["gen_args"], "--dry-run")
    assert out["prompt"].startswith("artist:x, cat, best") and "artist:x, artist:x" not in out["prompt"]
    _, out = run(capsys, "variations", "--image", str(img), "--dry-run")
    assert "artist:x, artist:x" not in out["prompt"]


def test_sd_weights_switch(capsys, tmp_path):
    argv = ["gen", "--prompt", "(long hair:1.2), \\(cat\\)", "--quality", "none", "--dry-run"]
    _, out = run(capsys, *argv)
    assert out["prompt"] == "(long hair:1.2), \\(cat\\)"
    _, out = run(capsys, *argv, "--convert-sd-weights")
    assert out["prompt"] == "1.2::long hair::, (cat)"
    write_defaults("convert_sd_weights = true\n")
    _, out = run(capsys, *argv)
    assert out["prompt"] == "1.2::long hair::, (cat)"
    img = tmp_path / "pic.png"
    nai_image(img, prompt="(a:1.5)")
    _, record = run(capsys, "meta", str(img))
    assert "--no-convert-sd-weights" in record["gen_args"]


def test_a_preset_typed_is_kept_or_refused_and_the_configs_carries_over(capsys):
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--quality", "light", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "gen", "--model", "v4.5-curated", "--prompt", "x", "--uc-preset", "furry", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"
    # The config's presets apply to every model, the closest one where a model lacks it
    write_defaults('quality = "light"\nuc_preset = "furry"\n')
    code, out = run(capsys, "gen", "--model", "v4.5-curated", "--prompt", "x", "--dry-run")
    assert code == 0 and out["body"]["parameters"]["tag_hint_uc_preset"] == presets.UC_PRESET_HINT["heavy"]


def test_images_of_any_mode_are_sent_as_png(capsys, with_token, tmp_path):
    for mode in ("CMYK", "YCbCr", "I;16"):
        img = tmp_path / f"{mode.replace(';', '')}.{'png' if mode == 'I;16' else 'jpg'}"
        Image.new(mode, (64, 64)).save(img)
        code, out = run(capsys, "upscale", str(img), "--yes")
        assert code == 0, (mode, out)
        code, out = run(capsys, "vibe", str(img), "--dry-run")
        assert code == 0, (mode, out)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "I", "I;16", "F"]
)
def test_every_mode_becomes_a_png_without_a_deprecated_save(mode):
    img = Image.new(mode, (4, 4))
    data = images.to_png(img)
    with Image.open(io.BytesIO(data)) as back:
        assert back.size == (4, 4)
    if mode == "I":
        assert data == images.to_png(img.convert("I;16"))
