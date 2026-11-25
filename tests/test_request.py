from dataclasses import replace

import pytest

from nai_cli import cost, errors, models, presets, request
from nai_cli.request import Character, GenSpec, PreciseReference, Vibe


def spec(**kw):
    return GenSpec(
        model=models.resolve(kw.pop("model", "v5")), prompt=kw.pop("prompt", "1girl"), seed=kw.pop("seed", 7), **kw
    )


def test_v5_body_shape_and_order():
    built = request.build(spec(negative="bad hands"))
    body = built.body
    assert list(body) == ["input", "model", "action", "parameters", "use_new_shared_trial"]
    p = body["parameters"]
    keys = list(p)
    assert keys[0] == "params_version" and keys[-2:] == ["noise_schedule", "image_format"]
    assert p["scale"] == 4.0 and p["steps"] == 28
    assert p["negative_prompt"] == presets.uc_preset_text(models.resolve("v5"), "heavy") + ", bad hands"
    assert p["tag_hint_qt"] == 1 and p["tag_hint_uc_preset"] == 2
    assert p["straight_alpha"] is True
    assert "skip_cfg_above_sigma" not in p
    assert body["input"] == "1girl, very aesthetic, masterpiece, no text"


def test_empty_negative_sends_uc_field():
    p = request.build(spec(uc_preset="none")).body["parameters"]
    assert p["uc"] == "" and "negative_prompt" not in p


def test_v45_keeps_noise_schedule_in_preset_block_and_variety_null():
    p = request.build(spec(model="v4.5")).body["parameters"]
    keys = list(p)
    assert keys.index("noise_schedule") < keys.index("legacy_v3_extend")
    assert p["skip_cfg_above_sigma"] is None
    assert p["scale"] == 5.0 and p["steps"] == 23


def test_variety_sigma_scales_with_size():
    p = request.build(spec(model="v4.5", variety=True, width=832, height=1216)).body["parameters"]
    assert p["skip_cfg_above_sigma"] == pytest.approx(58.0)


def test_default_character_centers():
    built = request.build(spec(characters=[Character("a"), Character("b"), Character("c")]))
    centers = [c["center"] for c in built.body["parameters"]["characterPrompts"]]
    assert centers == [{"x": 0.5, "y": 0.5}, {"x": 0.3, "y": 0.5}, {"x": 0.7, "y": 0.5}]


def test_default_center_skips_taken_slots():
    assert request.default_center([(0.52, 0.5)], freeform=True) == (0.3, 0.5)
    assert request.default_center([(0.42, 0.42)], freeform=True) == (0.5, 0.5)
    # Without free placement, slots are compared by snapped grid cell
    assert request.default_center([(0.42, 0.42)], freeform=False) == (0.3, 0.5)


def test_v45_snaps_caption_centers_but_keeps_raw_character_center():
    built = request.build(spec(model="v4.5", characters=[Character("a", x=0.62, y=0.18)], use_coords=True))
    p = built.body["parameters"]
    assert p["v4_prompt"]["caption"]["char_captions"][0]["centers"] == [{"x": 0.7, "y": 0.1}]
    assert p["characterPrompts"][0]["center"] == {"x": 0.62, "y": 0.18}
    assert list(p["characterPrompts"][0]) == ["prompt", "uc", "center", "enabled"]


def test_coords_mode_requires_positions():
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(characters=[Character("a")], use_coords=True))
    assert e.value.code == errors.BAD_ARGS


def test_medium_forces_fixed_settings_and_drops_negatives():
    built = request.build(spec(model="v5-medium", negative="x", steps=40, sampler="k_dpmpp_2m"))
    p = built.body["parameters"]
    assert p["steps"] == 14 and p["sampler"] == "k_euler_ancestral" and p["ucPresetId"] == "heavy"
    assert "cfg_rescale" not in p
    assert built.warnings


def test_v5_rejects_vibe():
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(vibes=[Vibe("enc", 0.5)]))
    assert e.value.code == errors.MODEL_UNSUPPORTED and e.value.detail["feature"] == "vibe"


def test_v5_precise_references_keep_outfits_and_drop_fidelity():
    kinds = ["character_mainoutfit", "character&style_anyoutfit", "costume", "style"]
    built = request.build(spec(references=[PreciseReference(b"x", k, 3.0, 0.5) for k in kinds]))
    p = built.body["parameters"]
    assert [d["caption"]["base_caption"] for d in p["director_reference_descriptions"]] == kinds
    assert p["director_reference_strength_values"] == [3.0] * 4
    assert p["director_reference_secondary_strength_values"] == [0.0] * 4
    assert any("fidelity" in w for w in built.warnings)


def test_v45_sends_outfit_preferences_as_plain_kinds_and_refuses_outfits():
    refs = [
        PreciseReference(b"x", "character_mainoutfit", 1, 1),
        PreciseReference(b"x", "character&style_anyoutfit", 1, 1),
    ]
    built = request.build(spec(model="v4.5", references=refs))
    p = built.body["parameters"]
    assert [d["caption"]["base_caption"] for d in p["director_reference_descriptions"]] == [
        "character",
        "character&style",
    ]
    assert any("outfit preference" in w for w in built.warnings)
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(model="v4.5", references=[PreciseReference(b"x", "costume", 1, 1)]))
    assert e.value.code == errors.MODEL_UNSUPPORTED and e.value.detail["feature"] == "outfit_reference"


def test_v5_takes_no_precise_reference_when_inpainting():
    ref = PreciseReference(b"x", "character", 1, 1)
    for model in ("v5", "v5-curated", "v5-medium"):
        with pytest.raises(errors.NaiError) as e:
            request.build(spec(model=model, action="infill", image=b"img", mask=b"mask", references=[ref]))
        assert e.value.code == errors.MODEL_UNSUPPORTED and e.value.detail["feature"] == "precise_reference"


def test_v4_has_no_precise_references():
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(model="v4", references=[PreciseReference(b"x", "character", 1, 1)]))
    assert e.value.detail["feature"] == "precise_reference"


def test_v45_precise_reference_wins_over_vibe():
    p = request.build(
        spec(model="v4.5", vibes=[Vibe("enc", 0.5)], references=[PreciseReference(b"x", "style", 0.8, 0.6)])
    ).body["parameters"]
    assert "reference_image_multiple" not in p
    assert p["director_reference_descriptions"][0]["caption"]["base_caption"] == "style"
    assert p["director_reference_secondary_strength_values"] == [pytest.approx(0.4)]


def test_vibe_strengths_normalized_when_sum_over_one():
    p = request.build(spec(model="v4.5", vibes=[Vibe("a", 0.8), Vibe("b", 0.8)])).body["parameters"]
    assert p["reference_strength_multiple"] == [0.5, 0.5] and p["normalize_reference_strength_multiple"] is True
    p = request.build(spec(model="v4.5", vibes=[Vibe("a", 0.8), Vibe("b", 0.8)], normalize_vibes=False)).body[
        "parameters"
    ]
    assert p["reference_strength_multiple"] == [0.8, 0.8] and p["normalize_reference_strength_multiple"] is False


def test_infill_uses_inpainting_model_and_noise_seed():
    built = request.build(spec(action="infill", image=b"img", mask=b"mask", inpaint_strength=0.5))
    assert built.body["model"] == "nai-diffusion-5-full-inpainting"
    p = built.body["parameters"]
    assert p["extra_noise_seed"] == 6 and p["add_original_image"] is False
    assert p["img2img"] == {"strength": 0.5, "color_correct": True}


def test_v5_curated_infill_follows_v45_curated_rules():
    built = request.build(spec(model="v5-curated", action="infill", image=b"img", mask=b"mask", uc_preset="furry"))
    assert built.body["model"] == "nai-diffusion-4-5-curated-inpainting"
    assert built.model.id == "nai-diffusion-4-5-curated"
    assert built.prompt == "1girl, " + presets.quality_tags(built.model, "standard")
    p = built.body["parameters"]
    # V4.5 Curated has no furry preset and falls back to heavy; Curated never adds nsfw
    assert p["ucPresetId"] == "heavy" and not built.negative.startswith("nsfw")
    keys = list(p)
    assert keys.index("noise_schedule") < keys.index("legacy_v3_extend")
    assert built.warnings


def test_img2img_fields():
    p = request.build(spec(action="img2img", image=b"img", strength=0.5, noise=0.1)).body["parameters"]
    assert p["strength"] == 0.5 and p["noise"] == 0.1 and p["color_correct"] is False and p["extra_noise_seed"] == 6


def test_transparent_prefix_in_suffix():
    built = request.build(spec(transparent=True))
    assert built.prompt.endswith("transparent background, very aesthetic, masterpiece, no text")
    assert built.body["parameters"]["tag_hint_transparent_background"] is True


def test_nsfw_by_default_only_removed_when_prompt_has_it():
    heavy = presets.uc_preset_text(models.resolve("v5"), "heavy")
    assert request.build(spec(negative="nsfw, x")).negative == heavy + ", nsfw, x"
    built = request.build(spec(prompt="1girl, {nsfw}", negative="nsfw, x"))
    assert built.negative == heavy + ", x" and built.warnings == []
    built = request.build(spec(prompt="nsfw", negative="2::nsfw::, nsfw_x, extra", uc_preset="none"))
    assert built.negative == "2::::, _x, extra"
    assert len(built.warnings) == 1 and "2::nsfw::" in built.warnings[0] and "nsfw_x" in built.warnings[0]


def test_official_nsfw_added_to_negative():
    heavy = presets.uc_preset_text(models.resolve("v5"), "heavy")
    assert request.build(spec(official_nsfw=True)).negative == "nsfw, " + heavy
    # No nsfw when the prompt (with quality tags) has it, without a UC preset, or on Curated; the text is untouched
    assert (
        request.build(spec(prompt="1girl, NSFW", negative="nsfw_x", official_nsfw=True)).negative == heavy + ", nsfw_x"
    )
    assert request.build(spec(uc_preset="none", negative="x", official_nsfw=True)).negative == "x"
    assert not request.build(spec(model="v5-curated", official_nsfw=True)).negative.startswith("nsfw")
    assert not request.build(spec(model="v4.5-curated", official_nsfw=True)).negative.startswith("nsfw")
    assert request.build(spec(model="v4.5", official_nsfw=True)).negative.startswith("nsfw, ")


def test_negative_verbatim_skips_preset_and_hint():
    built = request.build(spec(negative="lowres, x", negative_verbatim=True))
    p = built.body["parameters"]
    assert built.negative == "lowres, x" and "tag_hint_uc_preset" not in p and p["ucPresetId"] == "heavy"
    # Models with fixed settings still use the preset
    medium = request.build(spec(model="v5-medium", negative="lowres", negative_verbatim=True))
    assert medium.negative == presets.uc_preset_text(medium.model, "heavy")
    assert "tag_hint_uc_preset" in medium.body["parameters"]


def test_v5_noise_schedule_is_always_karras():
    built = request.build(spec(noise_schedule="exponential"))
    assert built.body["parameters"]["noise_schedule"] == "karras" and built.warnings


@pytest.mark.parametrize("name", ["v5", "v5-curated"])
def test_v5_with_its_hidden_options_takes_a_schedule_and_variety(name):
    model = models.with_hidden_options(models.resolve(name))
    built = request.build(GenSpec(model=model, prompt="1girl", seed=7, noise_schedule="exponential", variety=True))
    p = built.body["parameters"]
    assert p["noise_schedule"] == "exponential" and p["skip_cfg_above_sigma"] == 58
    # Placed where V5 places karras, as only the value differs
    assert list(p)[-2:] == ["noise_schedule", "image_format"]


def test_hidden_options_leave_v5_medium_and_older_models_as_they_are():
    for name in ("nai-diffusion-5-full-medium", "v4.5"):
        model = models.resolve(name)
        assert models.with_hidden_options(model) is model


def test_limits_match_official():
    with pytest.raises(errors.NaiError):
        request.build(spec(steps=51))
    # The official CFG box has a lower limit only
    assert request.build(spec(scale=10.5)).body["parameters"]["scale"] == 10.5
    for scale in (-0.1, float("inf"), float("nan")):
        with pytest.raises(errors.NaiError):
            request.build(spec(scale=scale))
    with pytest.raises(errors.NaiError):
        request.build(spec(width=1536, height=2048, n_samples=5))
    assert request.build(spec(width=512, height=512, n_samples=8)).body["parameters"]["n_samples"] == 8
    with pytest.raises(errors.NaiError):
        request.build(spec(model="v4.5", quality="light"))
    with pytest.raises(errors.NaiError):
        request.build(spec(model="v4.5-curated", uc_preset="furry"))


def test_only_the_official_v4_and_v5_samplers_are_taken():
    assert models.SAMPLERS == [
        "k_euler_ancestral",
        "k_euler",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_2m",
        "k_dpmpp_2m_sde",
        "k_dpmpp_sde",
    ]
    for name in ("ddim", "ddim_v3", "DDIM", "plms"):
        with pytest.raises(errors.NaiError) as e:
            models.resolve_sampler(name)
        assert e.value.code == errors.BAD_ARGS and e.value.detail["choices"] == models.SAMPLERS
    for name in ("ddim", "ddim_v3"):
        assert models.imported_sampler(name) == "k_euler_ancestral"
    assert models.imported_sampler("plms") is None
    assert request.build(spec(sampler="k_euler", steps=27)).body["parameters"]["steps"] == 27


def test_furry_mode_prefixes_prompt_like_official():
    assert request.build(spec(furry=True)).prompt == "fur dataset, 1girl, very aesthetic, masterpiece, no text"
    # The prefix goes on after quality tags and Enhance down-weighting, before the teXt: block
    quality = "very aesthetic, masterpiece, no text"
    assert request.build(spec(prompt="", furry=True)).prompt == f"fur dataset, {quality}"
    assert request.build(spec(prompt="|dog", furry=True)).prompt == f"fur dataset, {quality}|dog"
    assert request.build(spec(prompt="text: HELLO", furry=True)).prompt == f"fur dataset, {quality}text: HELLO"
    enhanced = request.build(spec(furry=True, enhance_hint=True, action="img2img", image=b"img"))
    assert enhanced.prompt == f"fur dataset, 1girl, {quality}, -2::upscaled, blurry::,"
    for prompt in ("fur dataset 1girl", "background dataset, city"):
        assert request.build(spec(prompt=prompt, furry=True)).prompt.startswith(prompt + ", very aesthetic")
    assert request.build(spec(model="v4.5-curated", furry=True)).prompt.startswith("fur dataset, 1girl, ")
    assert request.build(spec(model="v5-medium", furry=True)).prompt.startswith("fur dataset, 1girl")
    medium = models.resolve("v5-medium")
    with pytest.raises(errors.NaiError) as e:
        request.build(GenSpec(model=replace(medium, furry_mode=False), prompt="1girl", seed=7, furry=True))
    assert e.value.code == errors.MODEL_UNSUPPORTED


def test_characters_without_prompt_dropped_like_official():
    built = request.build(spec(characters=[Character("", "bad"), Character("boy", "x"), Character("girl")]))
    p = built.body["parameters"]
    assert [c["char_caption"] for c in p["v4_prompt"]["caption"]["char_captions"]] == ["boy", "girl"]
    assert [c["char_caption"] for c in p["v4_negative_prompt"]["caption"]["char_captions"]] == ["x", ""]
    # The dropped character still took the first place, so the others keep theirs
    assert [c["center"] for c in p["characterPrompts"]] == [{"x": 0.3, "y": 0.5}, {"x": 0.7, "y": 0.5}]
    assert [c.prompt for c in built.characters] == ["boy", "girl"]
    assert built.warnings == ["character 1 has no prompt and is not sent, as in the official client"]
    # Empty characters still take one of the official client's slots
    blank = [Character("")] * 2 + [Character(str(i)) for i in range(6)]
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(model="v4.5", characters=blank))
    assert e.value.code == errors.MODEL_UNSUPPORTED and e.value.detail["feature"] == "characters"
    assert len(request.build(spec(model="v4.5", characters=blank[2:])).characters) == 6


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"steps": 0}, "steps must be within 1-50"),
        ({"cfg_rescale": 1.2}, "CFG rescale must be within 0-1"),
        ({"cfg_rescale": float("nan")}, "CFG rescale must be within 0-1"),
        ({"seed": -1}, "seed must be within 0-9999999999"),
        ({"action": "img2img", "image": b"i", "strength": 1.0}, "strength must be within 0.01-0.99"),
        ({"action": "img2img", "image": b"i", "noise": 1.0}, "noise must be within 0-0.99"),
        ({"action": "infill", "image": b"i", "mask": b"m", "inpaint_strength": 0}, "inpaint strength must be within"),
        ({"model": "v4.5", "vibes": [Vibe("E", float("inf"))]}, "strengths must be finite"),
    ],
)
def test_values_outside_official_ranges_refused(kw, message):
    with pytest.raises(errors.NaiError) as e:
        request.build(spec(**kw))
    assert e.value.code == errors.BAD_ARGS and message in e.value.message


def test_values_past_reference_sliders_allowed_like_official():
    built = request.build(spec(model="v4.5", vibes=[Vibe("E", 1.5), Vibe("F", -0.2)]))
    assert built.body["parameters"]["reference_strength_multiple"] == [1.5 / 1.7, -0.2 / 1.7]
    built = request.build(spec(model="v4.5", references=[PreciseReference(b"r", "style", 1.4, -0.5)]))
    assert built.body["parameters"]["director_reference_strength_values"] == [1.4]


def test_built_characters_are_the_ones_sent():
    built = request.build(spec(model="v5-medium", characters=[Character("girl", "bad")]))
    assert [(c.prompt, c.negative) for c in built.characters] == [("girl", "")]


def test_cost_v5_portrait():
    v5 = models.resolve("v5")
    est = cost.generation(v5, 832, 1216, 28)
    assert est.anlas == 0 and est.free_by_opus and est.if_quota_exhausted == 30
    assert cost.generation(v5, 832, 1216, 28, quota_exhausted=True).anlas == 30
    assert cost.generation(v5, 1024, 1536, 28).anlas == cost.per_sample(v5, 1024, 1536, 28)
    assert cost.generation(v5, 832, 1216, 28, n_samples=4).anlas == 90


def test_cost_medium_ignores_steps():
    medium = models.resolve("v5-medium")
    assert cost.per_sample(medium, 832, 1216, 14) == cost.per_sample(medium, 832, 1216, 50) == 15


def test_cost_precise_reference_and_vibe():
    v45 = models.resolve("v4.5")
    sample = cost.per_sample(v45, 832, 1216, 23)
    est = cost.generation(v45, 832, 1216, 23, precise_references=2)
    assert est.free_by_opus and est.anlas == 10
    assert cost.generation(v45, 832, 1216, 23, n_samples=2, precise_references=2).anlas == sample + 20
    assert cost.generation(v45, 832, 1216, 23, vibes=6, vibes_to_encode=1).anlas == 4 + 2


def test_cost_upscale_and_augment():
    # Measured: Opus pays 1 for 512x512 too
    assert cost.upscale(512, 512) == 1
    assert cost.upscale(832, 1216) == 1
    assert cost.upscale(1216, 1664) == 3
    assert cost.upscale(1536, 2048) == 4
    assert cost.upscale(1536, 2112) == -1
    # One 28-step nai-diffusion-3 image after scaling to at least 1 MP, as the official client prices tools
    assert cost.augment(832, 1216) == 0
    assert cost.augment(832, 1216, opus=False) == 20
    assert cost.augment(832, 1216, bg_removal=True) == 65
    assert cost.augment(1024, 1536) == 30 and cost.augment(1024, 1536, bg_removal=True) == 95
    assert cost.augment(4096, 4096, opus=False) == 60
    # A narrow image is scaled up like any other
    assert cost.augment(8, 4096) == 0 and cost.augment(8, 4096, opus=False) == 20


def test_tool_request_size():
    assert presets.tool_request_size(512, 512) == (1024, 1024)
    w, h = presets.tool_request_size(4000, 3000)
    assert w * h <= presets.TOOL_REQUEST_MAX_PIXELS
    assert presets.tool_request_size(1024, 1024) == (1024, 1024)
    assert presets.emotion_prompt("happy", "") == "happy;;"


def test_size_presets_and_validation():
    assert presets.resolve_size("landscape") == (1216, 832)
    assert presets.resolve_size("1024×1024") == (1024, 1024)
    with pytest.raises(errors.NaiError):
        presets.resolve_size("1000x1000")
    with pytest.raises(errors.NaiError):
        presets.resolve_size("2048x2048")


def test_inpainting_keeps_hidden_options_when_the_model_does_not_switch():
    model = models.with_hidden_options(models.resolve("v5"))
    assert models.for_action(model, "infill") is model
    assert models.for_action(models.with_hidden_options(models.resolve("v5-curated")), "infill").family == "v4.5"
