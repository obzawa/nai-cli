"""Taggers and SAM run on this computer, tested with stand-in sessions so onnxruntime is not needed."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import run
from PIL import Image

from nai_cli import api, config, errors, inpaint, local_models, sam, tagger
from nai_cli.commands import interrogate


class FakeSession:
    def __init__(self, inputs, outputs, answer):
        self.inputs = [SimpleNamespace(name=n, shape=s) for n, s in inputs]
        self.outputs = [SimpleNamespace(name=n) for n in outputs]
        self.answer = answer
        self.fed = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, feed):
        self.fed = feed
        return [self.answer(feed)]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _interrogate(image, labels, session):
    settings = tagger.Settings(labels.parent / "wd.onnx", labels, tagger.Kind.WD14)
    return tagger.interrogate(image, settings, tagger.load_labels(labels), local_models.Model("wd.onnx", session))


def test_wd_tag_lists_read_names_and_categories(tmp_path):
    labels = tagger.load_labels(
        _write(
            tmp_path / "selected_tags.csv",
            "tag_id,name,category,count\n1,general,9,1\n2,long_hair,0,1\n3,hatsune_miku,4,1\n",
        )
    )
    assert [(lb.name, lb.category.value) for lb in labels.labels] == [
        ("general", "rating"),
        ("long_hair", "general"),
        ("hatsune_miku", "character"),
    ]


def test_plain_lists_count_as_general(tmp_path):
    labels = tagger.load_labels(_write(tmp_path / "labels.txt", "# tags\nsmile\t0\nsky\n"))
    assert [(lb.name, lb.category) for lb in labels.labels] == [
        ("smile", tagger.Category.GENERAL),
        ("sky", tagger.Category.GENERAL),
    ]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {
                "tags": ["a", "b", "c"],
                "tags_split": [["general", 2], ["style", 1]],
                "category_best_threshold": {"style": 0.2},
            },
            [("a", "general"), ("b", "general"), ("c", "artist")],
        ),
        (
            {
                "categories": [
                    {"name": "character", "offset": 1, "count": 1, "tags": ["x"]},
                    {"name": "general", "offset": 0, "count": 1, "tags": ["y"]},
                ]
            },
            [("y", "general"), ("x", "character")],
        ),
        (
            {"idx_to_tag": {"1": "b", "0": "a"}, "tag_to_category": {"a": "4"}, "categories": {"4": "character"}},
            [("a", "character"), ("b", "other")],
        ),
        (
            # CL tagger v2 lists each category's tags under its name
            {
                "tag_to_idx": {"a": 0, "b": 1, "c": 2},
                "tag_to_category": {"a": "General", "b": "Character", "c": "Rating"},
                "categories": {"General": ["a"], "Character": ["b"], "Rating": ["c"]},
            },
            [("a", "general"), ("b", "character"), ("c", "rating")],
        ),
        (
            {"idx_to_tag": ["a", "b"], "categories": {"General": ["a"], "Character": ["b"]}},
            [("a", "general"), ("b", "character")],
        ),
    ],
)
def test_json_tag_lists(tmp_path, data, expected):
    labels = tagger.load_labels(_write(tmp_path / "tags.json", json.dumps(data)))
    assert [(lb.name, lb.category.value) for lb in labels.labels] == expected


def test_thresholds_and_prompt_order():
    labels = [
        tagger.Label("smile", tagger.Category.GENERAL),
        tagger.Label("miku", tagger.Category.CHARACTER),
        tagger.Label("someone", tagger.Category.ARTIST),
        tagger.Label("sky", tagger.Category.GENERAL),
        tagger.Label("explicit", tagger.Category.RATING),
    ]
    s = tagger.Settings(None, None, tagger.Kind.WD14, 0.5, 0.8, tagger.CHOOSABLE)
    tags = tagger.select(labels, np.array([0.6, 0.9, 0.3, 0.7, 1.0]), s, {tagger.Category.ARTIST: 0.2})
    assert [t.text for t in tags] == ["artist:someone", "miku", "sky", "smile"]


def test_kinds_are_guessed_from_names(tmp_path):
    csv = tmp_path / "selected_tags.csv"
    assert tagger.guess_kind(tmp_path / "wd-vit-tagger-v3.onnx", csv) is tagger.Kind.WD14
    assert tagger.guess_kind(tmp_path / "cl_tagger_v2" / "model.onnx", csv) is tagger.Kind.CL_V2
    assert tagger.guess_kind(tmp_path / "pixai" / "model.onnx", csv) is tagger.Kind.PIXAI
    assert tagger.guess_kind(tmp_path / "cl_tagger.onnx", csv) is tagger.Kind.CL


def test_wd_input_is_bgr_channels_last_and_logits_become_scores(tmp_path):
    labels = _write(tmp_path / "selected_tags.csv", "tag_id,name,category,count\n1,red,0,1\n2,blue,0,1\n")
    session = FakeSession([("input", [1, 448, 448, 3])], ["out"], lambda feed: np.array([[3.0, -3.0]]))
    img = Image.new("RGB", (100, 50), (255, 0, 0))
    tags = _interrogate(img, labels, session)
    tensor = session.fed["input"]
    assert tensor.shape == (1, 448, 448, 3)
    # Letterboxed on white, so the middle row is red in BGR order and the top is white
    assert tensor[0, 224, 224].tolist() == [0, 0, 255] and tensor[0, 0, 0].tolist() == [255, 255, 255]
    assert [t.name for t in tags] == ["red"] and tags[0].score > 0.9


def test_a_model_of_another_layout_is_refused(tmp_path):
    labels = _write(tmp_path / "selected_tags.csv", "tag_id,name,category,count\n1,red,0,1\n")
    session = FakeSession([("input", [1, 3, 448, 448])], ["out"], lambda feed: np.array([[1.0]]))
    with pytest.raises(errors.NaiError):
        _interrogate(Image.new("RGB", (8, 8)), labels, session)


def _small_assets(monkeypatch):
    """SAM files a few bytes long, with the sizes and hashes the checks expect."""
    files = {a.url: f"file {a.name}".encode() for a in local_models.SAM}
    assets = tuple(
        local_models.Asset(a.name, a.url, len(files[a.url]), hashlib.sha256(files[a.url]).hexdigest())
        for a in local_models.SAM
    )
    monkeypatch.setattr(local_models, "SAM", assets)
    return files


def _fake_sam(monkeypatch, tmp_path):
    files = _small_assets(monkeypatch)
    for asset in local_models.SAM:
        (tmp_path / asset.name).write_bytes(files[asset.url])
    seen = {}

    def decode(feed):
        seen.update(feed)
        logits = np.full((1, 1, 256, 256), -5.0, np.float32)
        logits[..., :, :128] = 5.0
        return logits

    encoder = FakeSession(
        [("input_image", [1, 3, 512, 512])], ["image_embeddings"], lambda feed: np.zeros((1, 256, 64, 64))
    )
    decoder = FakeSession(
        [("image_embeddings", [1, 256, 64, 64]), ("point_coords", [1, -1, 2]), ("point_labels", [1, -1])],
        ["masks"],
        decode,
    )
    sessions = {"l0_encoder.onnx": encoder, "l0_decoder.onnx": decoder}
    monkeypatch.setattr(local_models, "load", lambda path: local_models.Model(path.name, sessions[path.name]))
    return seen


def test_sam_scales_points_and_crops_the_padding(monkeypatch, tmp_path):
    seen = _fake_sam(monkeypatch, tmp_path)
    mask = np.asarray(sam.segment(Image.new("RGB", (400, 200)), [(0.5, 0.5)], *local_models.sam(tmp_path)))
    assert seen["point_coords"].tolist() == [[[512.0, 256.0]]]
    # The left half of the 1024 square is the left half of the 1024x512 image
    assert mask.shape == (200, 400) and mask[:, :195].all() and not mask[:, 205:].any()


def test_sam_without_its_files_says_how_to_get_them(tmp_path):
    with pytest.raises(errors.NaiError, match="nai local download sam"):
        local_models.sam(tmp_path)


@pytest.mark.parametrize(("index", "damage"), [(0, "missing"), (1, "truncated")])
def test_cached_sam_rechecks_its_files(tmp_path, monkeypatch, index, damage):
    files = _small_assets(monkeypatch)
    for asset in local_models.SAM:
        (tmp_path / asset.name).write_bytes(files[asset.url])
    loaded = []

    def load(path):
        loaded.append(path.name)
        return local_models.Model(path.name, object())

    monkeypatch.setattr(local_models, "load", load)
    pair = local_models.sam(tmp_path)
    asset = local_models.SAM[index]
    path = tmp_path / asset.name
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"")
    with pytest.raises(errors.NaiError) as err:
        local_models.sam(tmp_path)
    assert err.value.code == errors.CONFIG_MISSING and asset.name in err.value.message
    path.write_bytes(files[asset.url])
    assert local_models.sam(tmp_path) == pair
    assert loaded == [a.name for a in local_models.SAM]


def test_object_regions_add_and_subtract():
    assert inpaint.parse_region("-object:0.1,0.2,0.3,0.4").points == [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(errors.NaiError):
        inpaint.parse_region("object:0.1")

    def segment(image, points):
        assert points == [(0.25, 0.5)]
        out = Image.new("L", image.size, 0)
        out.paste(255, (0, 0, image.width // 2, image.height))
        return out

    image = Image.new("RGB", (40, 20))
    regions = [inpaint.Region("rect", [0, 0, 1, 1]), inpaint.parse_region("-object:0.25,0.5")]
    mask = np.asarray(inpaint.draw_regions(Image.new("L", (40, 20), 0), regions, image, segment))
    assert not mask[:, :19].any() and mask[:, 21:].all()
    with pytest.raises(errors.NaiError):
        inpaint.draw_regions(Image.new("L", (4, 4), 0), [inpaint.parse_region("object:0.5,0.5")])


def test_redact_an_object(capsys, monkeypatch, tmp_path):
    _fake_sam(monkeypatch, tmp_path)
    (tmp_path / "local").mkdir(exist_ok=True)
    config.default_path().parent.mkdir(parents=True, exist_ok=True)
    config.default_path().write_text(
        f'novelai_token = ""\n[local]\nsam_dir = "{tmp_path.as_posix()}"\n', encoding="utf-8"
    )
    source = tmp_path / "pic.png"
    Image.new("RGB", (64, 32), (255, 255, 255)).save(source)
    code, out = run(capsys, "redact", str(source), "--region", "object:0.2,0.5", "--effect", "solid")
    assert code == 0, out
    with Image.open(out["files"][0]["file"]) as img:
        pixels = np.asarray(img.convert("RGB"))
    assert (pixels[:, :30] == 0).all() and (pixels[:, 34:] == 255).all()

    target = tmp_path / "second.png"
    (tmp_path / local_models.SAM[1].name).unlink()
    code, out = run(capsys, "redact", str(source), "--region", "object:0.2,0.5", "-o", str(target))
    assert (code, out.get("code")) == (1, errors.CONFIG_MISSING) and not target.exists()


def test_local_status_and_download(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "onnxruntime", None)
    code, out = run(capsys, "local")
    assert code == 0 and out["onnxruntime"] is False and "nai-cli[local]" in out["install"]
    assert out["sam"]["missing"] == [a.name for a in local_models.SAM]
    code, out = run(capsys, "local", "download", "sam", "--dry-run")
    assert code == 0 and out["bytes"] == sum(a.size for a in local_models.SAM)
    monkeypatch.setattr(api, "fetch_file", lambda url, size: b"wrong")
    code, out = run(capsys, "local", "download", "sam")
    assert code == 1 and "hash" in out["message"]
    assert not any((config.default_sam_dir() / a.name).exists() for a in local_models.SAM)
    files = _small_assets(monkeypatch)
    monkeypatch.setattr(api, "fetch_file", lambda url, size: files[url])
    code, out = run(capsys, "local", "download", "sam")
    assert code == 0 and run(capsys, "local")[1]["sam"]["missing"] == []
    encoder, decoder = (config.default_sam_dir() / a.name for a in local_models.SAM)
    # Cut short, and the same size with other bytes: both are fetched again
    encoder.write_bytes(b"cut")
    decoder.write_bytes(b"x" * local_models.SAM[1].size)
    assert run(capsys, "local")[1]["sam"]["missing"] == [encoder.name]
    code, out = run(capsys, "local", "download", "sam")
    assert code == 0 and out["files"] == [encoder.name, decoder.name]
    assert [p.read_bytes() for p in (encoder, decoder)] == [files[a.url] for a in local_models.SAM]
    code, out = run(capsys, "local", "download", "sam")
    assert code == 0 and out["files"] == []
    code, out = run(capsys, "local", "download")
    assert code == 1 and out["code"] == "bad_args"


def test_interrogate_needs_a_model(capsys, tmp_path):
    source = tmp_path / "pic.png"
    Image.new("RGB", (8, 8)).save(source)
    code, out = run(capsys, "interrogate", str(source))
    assert code == 1 and out["code"] == "config_missing"
    code, out = run(
        capsys,
        "interrogate",
        str(source),
        "--model",
        str(tmp_path / "none.onnx"),
        "--labels",
        "x.csv",
        "--threshold",
        "2",
    )
    assert code == 1 and out["code"] == "bad_args"


def _refused(feed):
    raise RuntimeError("InvalidArgument: wrong rank")


@pytest.mark.parametrize(
    ("inputs", "answer", "said"),
    [
        ([("input", [1, 256, 64, 64])], _refused, "cannot run"),
        ([("image_embeddings", [1, 256, 64, 64]), ("point_coords", [1, -1, 2])], np.zeros, "not the model expected"),
        ([("input", [1, 448, 448, 3])], lambda feed: np.zeros((1, 3)), "3 scores but"),
        ([], np.zeros, "takes no image"),
        ([("input", [1, 448, 448, 3])], lambda feed: np.float32(1.0), "does not give tag scores"),
        ([("input", [1, 448, 448, 3])], lambda feed: np.zeros((0, 2)), "does not give tag scores"),
        ([("input", [1, 448, 448, 3])], lambda feed: np.array([["a", "b"]]), "does not give tag scores"),
    ],
)
def test_a_model_that_does_not_fit_is_bad_args(tmp_path, inputs, answer, said):
    labels = _write(tmp_path / "selected_tags.csv", "tag_id,name,category,count\n1,red,0,1\n2,blue,0,1\n")
    session = FakeSession(inputs, ["out"], answer)
    with pytest.raises(errors.NaiError) as e:
        _interrogate(Image.new("RGB", (8, 8)), labels, session)
    assert e.value.code == errors.BAD_ARGS and said in e.value.message


def test_a_sam_folder_holding_other_models_is_bad_args(monkeypatch, tmp_path):
    _fake_sam(monkeypatch, tmp_path)
    encoder, decoder = local_models.sam(tmp_path)
    decoder.session.inputs = decoder.session.inputs[:1]
    with pytest.raises(errors.NaiError) as e:
        sam.segment(Image.new("RGB", (40, 20)), [(0.5, 0.5)], encoder, decoder)
    assert e.value.code == errors.BAD_ARGS and "l0_decoder.onnx" in e.value.message


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("tags.csv", "name,category\n" + "x" * (1 << 18) + '"\n'),
        (
            "tags.json",
            '{"num_classes": 100000000000000000000, "categories": [{"name": "general", "offset": 0, '
            '"count": 1, "tags": ["a"]}]}',
        ),
        (
            "tags.json",
            '{"num_classes": 5, "categories": [{"name": "general", "offset": 0, "count": 1, "tags": ["a"]}]}',
        ),
        # A count far beyond the tags listed, with no num_classes to stop it
        (
            "tags.json",
            '{"categories": [{"name": "general", "offset": 0, "count": 100000000000000000000, "tags": ["a"]}]}',
        ),
        ("tags.json", '{"categories": [{"name": "general", "offset": true, "count": 1, "tags": ["a"]}]}'),
        ("tagger_config.json", '{"tags": ["a", "b"], "tags_split": [["general", -1], ["character", 3]]}'),
        ("tagger_config.json", '{"tags": ["a"], "tags_split": [1]}'),
    ],
)
def test_a_malformed_tag_list_is_bad_args(tmp_path, name, text):
    # The CSV field is longer than the csv module's default limit
    path = _write(tmp_path / name, text)
    with pytest.raises(errors.NaiError) as e:
        tagger.load_labels(path)
    assert e.value.code == errors.BAD_ARGS


def test_a_tag_list_that_is_not_utf8_is_bad_args(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("tag_id,name\n1,smile\n", encoding="utf-16")
    with pytest.raises(errors.NaiError) as e:
        tagger.load_labels(path)
    assert e.value.code == errors.BAD_ARGS


def test_a_missing_tag_list_is_config_missing(capsys, tmp_path):
    model = tmp_path / "wd.onnx"
    model.write_bytes(b"x")
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    code, out = run(capsys, "interrogate", str(img), "--model", str(model), "--labels", str(tmp_path / "gone.json"))
    assert (code, out["code"]) == (1, "config_missing"), out


def test_a_list_without_categories_but_ratings_still_gives_tags(tmp_path):
    labels = tagger.load_labels(_write(tmp_path / "labels.json", '["1girl", "solo", "rating:general"]'))
    assert [label.category for label in labels.labels] == [
        tagger.Category.GENERAL,
        tagger.Category.GENERAL,
        tagger.Category.RATING,
    ]


def test_tags_json_and_a_tagger_config_are_found_beside_the_model(tmp_path):
    model = tmp_path / "model.onnx"
    _write(tmp_path / "config.json", '{"architectures": ["x"]}')
    with pytest.raises(errors.NaiError):
        interrogate._labels_beside(model)
    split = _write(tmp_path / "config.json", '{"tags_split": {"general": 1}, "tags": ["a"]}')
    assert interrogate._labels_beside(model) == split
    # A tag list of a usual name comes first
    tags = _write(tmp_path / "tags.json", "{}")
    assert interrogate._labels_beside(model) == tags
