import base64
import hashlib
import io
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from helpers import FakeClient, fake_clients, run, set_cap
from PIL import Image, PngImagePlugin
from samples import CUT, bomb_png, v45_image, vibe_entry

from nai_cli import caches, decode, errors, fs, ledger, models, vibe
from nai_cli.commands.references import load_precise, precise_ref

V45 = "nai-diffusion-4-5-full"


def vibe_data(name, strength=0.5, info=0.7, mask=None, encoding="ENC"):
    import_info = {"model": V45, "information_extracted": info, "strength": strength}
    if mask is not None:
        import_info["mask"] = mask
    return vibe_entry(
        encodings={
            "v4-5full": {vibe.params_key(info, mask): {"encoding": encoding, "params": {"information_extracted": info}}}
        },
        name=name,
        importInfo=import_info,
    )


def test_params_key_like_official():
    assert vibe.params_key(0.7) == hashlib.sha256(b"information_extracted:0.7").hexdigest()
    assert vibe.params_key(1.0) == hashlib.sha256(b"information_extracted:1").hexdigest()
    assert vibe.params_key(0.7, "M") == hashlib.sha256(b"information_extracted:0.7,mask:M").hexdigest()
    assert vibe.params_key(1, {"a": [1, 2]}) == hashlib.sha256(b'information_extracted:1,mask:{"a":[1,2]}').hexdigest()


def test_bundle_loads_every_vibe(tmp_path):
    path = tmp_path / "set.naiv4vibebundle"
    bundle = {
        "identifier": vibe.BUNDLE_IDENTIFIER,
        "version": 1,
        "vibes": [vibe_data("one", 0.4), vibe_data("", 0.9, encoding="TWO")],
    }
    path.write_text(json.dumps(bundle), encoding="utf-8")
    sources = vibe.load(path, None, None, V45)
    assert [(s.name, s.strength) for s in sources] == [("one", 0.4), ("set 2", 0.9)]
    assert [caches.vibe_encoding(s, V45) for s in sources] == ["ENC", "TWO"]
    # Values given on the command line apply to every Vibe in the bundle
    assert {s.strength for s in vibe.load(path, 0.3, None, V45)} == {0.3}


@pytest.mark.parametrize("vibes", [[], None, ["x"]])
def test_bundle_without_vibes(tmp_path, vibes):
    path = tmp_path / "empty.naiv4vibebundle"
    path.write_text(json.dumps({"identifier": vibe.BUNDLE_IDENTIFIER, "vibes": vibes}), encoding="utf-8")
    with pytest.raises(errors.NaiError) as e:
        vibe.load(path, None, None, V45)
    assert e.value.code == errors.BAD_ARGS


def test_vibe_embedded_in_png(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("naidata", base64.b64encode(json.dumps(vibe_data("inside")).encode()).decode())
    path = tmp_path / "export.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    [source] = vibe.load(path, None, None, V45)
    assert source.name == "inside" and source.image_b64 is None and caches.vibe_encoding(source, V45) == "ENC"


def test_plain_image_is_encoded_later(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("naidata", base64.b64encode(b'{"identifier": "other"}').decode())
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    [source] = vibe.load(path, None, None, V45)
    assert source.image_b64 and source.encodings == {} and source.information_extracted == 0.7
    assert caches.vibe_encoding(source, V45) is None


def test_mask_selects_its_own_encoding(tmp_path):
    path = tmp_path / "masked.naiv4vibe"
    path.write_text(json.dumps(vibe_data("masked", mask="TQ==")), encoding="utf-8")
    [source] = vibe.load(path, None, None, V45)
    assert source.mask == "TQ==" and caches.vibe_encoding(source, V45) == "ENC"
    # Cached encodings keep their old names unless there is a mask
    plain = replace(source, mask=None, image_b64="IMG")
    assert caches._vibe_path(plain, V45).name == hashlib.sha256(f"IMG|{V45}|0.7".encode()).hexdigest() + ".txt"
    assert caches._vibe_path(replace(plain, mask="TQ=="), V45) != caches._vibe_path(plain, V45)


def test_missing_stored_values_take_defaults_like_official(tmp_path):
    path = tmp_path / "null.naiv4vibe"
    data = vibe_entry(
        encodings={"v4-5full": {"k": {"encoding": "E"}}}, importInfo={"strength": None, "information_extracted": None}
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    (source,) = vibe.load(path, None, None, "nai-diffusion-4-5-full")
    assert source.strength == vibe.DEFAULT_STRENGTH and source.information_extracted == 0.7


@pytest.mark.parametrize(
    "encodings", [[], {"v4-5full": []}, {"v4-5full": {"k": []}}, {"v4-5full": {"k": {"encoding": 1}}}]
)
def test_malformed_encodings_refused(tmp_path, encodings):
    path = tmp_path / "bad.naiv4vibe"
    path.write_text(json.dumps({"identifier": "novelai-vibe-transfer", "version": 1, "encodings": encodings}))
    with pytest.raises(errors.NaiError) as e:
        vibe.load(path, None, None, "nai-diffusion-4-5-full")
    assert e.value.code == errors.BAD_ARGS


def test_empty_cache_entry_is_encoded_again(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(img)
    (source,) = vibe.load(img, None, None, V45)
    caches._vibe_path(source, V45).parent.mkdir(parents=True, exist_ok=True)
    caches._vibe_path(source, V45).write_text("", encoding="utf-8")
    assert caches.vibe_encoding(source, V45) is None
    caches.remember_vibe(source, V45, "E")
    assert caches.vibe_encoding(source, V45) == "E"


def _vibe_file(tmp_path, **fields):
    path = tmp_path / "v.naiv4vibe"
    path.write_text(json.dumps(vibe_entry(**fields)), encoding="utf-8")
    return path


def test_empty_image_counts_as_none(tmp_path):
    stored = {"v4-5full": {vibe.params_key(0.7): {"encoding": "E"}}}
    (source,) = vibe.load(_vibe_file(tmp_path, image="", encodings=stored), None, None, V45)
    assert source.image_b64 is None and caches.vibe_encoding(source, V45) == "E"
    (source,) = vibe.load(_vibe_file(tmp_path, image="", encodings={}), None, None, V45)
    with pytest.raises(errors.NaiError) as e:
        caches.vibe_encoding(source, V45)
    assert e.value.code == errors.BAD_ARGS


@pytest.mark.parametrize(
    "fields",
    [
        {"image": "!!!not base64!!!"},
        {"image": base64.b64encode(b"not an image").decode()},
        {"image": 5},
        {"image": base64.b64encode(bomb_png()).decode()},
        {"importInfo": {"mask": 5}},
    ],
)
def test_broken_vibe_file_fails_on_load(tmp_path, fields):
    with pytest.raises(errors.NaiError) as e:
        vibe.load(_vibe_file(tmp_path, encodings={}, **fields), None, None, V45)
    assert e.value.code == errors.BAD_ARGS


def test_line_broken_base64_image_accepted(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
    wrapped = base64.encodebytes(buf.getvalue()).decode()
    assert "\n" in wrapped
    (source,) = vibe.load(_vibe_file(tmp_path, image=wrapped, encodings={}), None, None, V45)
    assert source.image_b64 == base64.b64encode(buf.getvalue()).decode()
    vibe.check_image(source)


def test_truncated_image_found_before_encoding(tmp_path):
    # The header survives, so loading the file accepts it
    (source,) = vibe.load(_vibe_file(tmp_path, image=base64.b64encode(CUT).decode(), encodings={}), None, None, V45)
    with pytest.raises(errors.NaiError) as e:
        vibe.check_image(source)
    assert e.value.code == errors.BAD_ARGS


def test_cache_written_whole(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(img)
    (source,) = vibe.load(img, None, None, V45)
    caches.remember_vibe(source, V45, "E")
    folder = caches._vibe_path(source, V45).parent
    assert caches.vibe_encoding(source, V45) == "E" and not list(folder.glob("*.tmp"))

    def failed(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fs.os, "replace", failed)
    with pytest.raises(OSError):
        caches.remember_vibe(source, V45, "F")
    assert caches.vibe_encoding(source, V45) == "E" and not list(folder.glob("*.tmp"))


def _png(color="red", size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"id": "abc"}, "no valid id"),
        ({"id": "A" * 64}, "no valid id"),
        ({"id": "0" * 64}, "encoding does not match"),
        ({"thumbnail": "http://example.com/a.png"}, "invalid thumbnail"),
    ],
)
def test_vibe_id_checked_like_official_import(tmp_path, change, reason):
    path = tmp_path / "v.naiv4vibe"
    path.write_text(json.dumps({**vibe_data("v"), **change}), encoding="utf-8")
    with pytest.raises(errors.NaiError, match=reason) as e:
        vibe.load(path, None, None, V45)
    assert e.value.code == errors.BAD_ARGS


def test_image_vibe_id_is_the_hash_of_its_image(tmp_path):
    image = base64.b64encode(_png()).decode()
    path = tmp_path / "v.naiv4vibe"
    path.write_text(json.dumps(vibe_entry(image=image, encodings={})), encoding="utf-8")
    assert vibe.load(path, None, None, V45)[0].image_b64 == image
    path.write_text(json.dumps({**vibe_entry(image=image, encodings={}), "id": "1" * 64}), encoding="utf-8")
    with pytest.raises(errors.NaiError, match="image does not match"):
        vibe.load(path, None, None, V45)


def test_encoding_only_vibe_keeps_its_information_extracted(tmp_path):
    path = tmp_path / "v.naiv4vibe"
    path.write_text(json.dumps(vibe_data("v", info=0.7)), encoding="utf-8")
    assert vibe.load(path, None, 0.7, V45)[0].information_extracted == 0.7
    # It cannot be encoded again, so the official client locks the value
    with pytest.raises(errors.NaiError, match=r"stays at 0\.7") as e:
        vibe.load(path, None, 0.5, V45)
    assert e.value.code == errors.BAD_ARGS


def _settings_image(path, source="NovelAI Diffusion V4.5 4BDE2A90", **settings):
    info = PngImagePlugin.PngInfo()
    info.add_text("Source", source)
    info.add_text("Comment", json.dumps({"prompt": "x", "seed": 1, **settings}))
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    return path


def test_vibes_restored_from_image_settings(tmp_path):
    img = _settings_image(
        tmp_path / "made.png",
        reference_image_multiple=["E1", "E2"],
        reference_strength_multiple=[0.4, 0.8],
        reference_information_extracted_multiple=[1, 0.5],
    )
    sources = vibe.from_settings(img)
    assert [(s.strength, s.information_extracted, s.image_b64) for s in sources] == [(0.4, 1.0, None), (0.8, 0.5, None)]
    assert [caches.vibe_encoding(s, V45) for s in sources] == ["E1", "E2"]
    # Kept for the model that made the image, as the official import stores them
    with pytest.raises(errors.NaiError, match="no encoding"):
        caches.vibe_encoding(sources[0], "nai-diffusion-4-5-curated")


@pytest.mark.parametrize("encoding", ["bad\x00text", "bad\ud800text"])
@pytest.mark.parametrize("kind", ["settings", "file", "bundle", "embedded"])
def test_unsendable_vibe_never_reaches_a_paid_request(capsys, with_token, tmp_path, monkeypatch, kind, encoding):
    clients = fake_clients(monkeypatch)
    flag = "--vibe"
    if kind == "settings":
        path = _settings_image(tmp_path / "made.png", reference_image_multiple=[encoding])
        flag = "--vibes-from"
    else:
        entry = vibe_data("sample")
        variants = entry["encodings"]["v4-5full"]
        variants["first"] = variants.pop(vibe.params_key(0.7))
        variants[vibe.params_key(0.7)] = {"encoding": encoding}
        data = entry if kind == "file" else vibe.bundle([entry])
        if kind == "embedded":
            path = tmp_path / "embedded.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("naidata", base64.b64encode(json.dumps(data).encode()).decode())
            Image.new("RGB", (8, 8)).save(path, pnginfo=info)
        else:
            path = tmp_path / "vibe.json"
            path.write_text(json.dumps(data), encoding="utf-8")
    code, result = run(capsys, "gen", "--prompt", "cat", "--model", "v4.5", flag, str(path), "--yes")
    assert code == 1 and result.get("code") == "bad_args", result
    assert not any(call in ("encode_vibe", "generate") for c in clients for call in c.calls)


@pytest.mark.parametrize("encoding", ["bad\x00text", "bad\ud800text"])
def test_an_unsendable_cached_encoding_is_encoded_again(capsys, with_token, tmp_path, monkeypatch, encoding):
    clients = fake_clients(monkeypatch)
    path = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(path)
    [source] = vibe.load(path, None, None, V45)
    cache = caches._vibe_path(source, V45)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(encoding.encode("utf-8", errors="surrogatepass"))
    code, result = run(capsys, "gen", "--prompt", "cat", "--model", "v4.5", "--vibe", str(path), "--yes")
    assert code == 0, result
    sent = [call for c in clients for call in c.calls if call in ("encode_vibe", "generate")]
    assert sent == ["encode_vibe", "generate"]


def test_missing_vibe_values_in_settings_take_defaults(tmp_path):
    img = _settings_image(tmp_path / "made.png", reference_image_multiple=["E1"])
    [source] = vibe.from_settings(img)
    assert source.strength == vibe.DEFAULT_STRENGTH and source.information_extracted == 0.7


@pytest.mark.parametrize(
    "source, settings, reason",
    [
        ("NovelAI Diffusion V4.5 4BDE2A90", {}, "keeps no Vibes"),
        ("NovelAI Diffusion V4.5 4BDE2A90", {"reference_image_multiple": [5]}, "invalid encoding"),
        ("NovelAI Diffusion V5 0ADF9AB7", {"reference_image_multiple": ["E"]}, "model that has Vibes"),
    ],
)
def test_vibes_from_settings_refused(tmp_path, source, settings, reason):
    img = _settings_image(tmp_path / "made.png", source=source, **settings)
    with pytest.raises(errors.NaiError, match=reason) as e:
        vibe.from_settings(img)
    assert e.value.code == errors.BAD_ARGS


def test_image_vibe_exported_like_official(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(_png(size=(512, 256)))
    [source] = vibe.load(img, 0.5, None, V45)
    data = vibe.export(source, V45, "E", without_image=False)
    fields = ["identifier", "version", "type", "image", "id", "encodings", "name", "thumbnail", "createdAt"]
    assert list(data) == [*fields, "importInfo"]
    assert data["type"] == "image" and data["id"] == hashlib.sha256(data["image"].encode()).hexdigest()
    params = {"information_extracted": 0.7}
    assert data["encodings"] == {"v4-5full": {vibe.params_key(0.7): {"encoding": "E", "params": params}}}
    assert data["importInfo"] == {"model": V45, "information_extracted": 0.7, "strength": 0.5}
    assert list(data["importInfo"]) == ["model", "information_extracted", "strength"]
    assert data["name"] == "pic" and isinstance(data["createdAt"], int)
    # A JPEG of 256 on the longer side, as the official client keeps with an image Vibe
    prefix = "data:image/jpeg;base64,"
    assert data["thumbnail"].startswith(prefix)
    with Image.open(io.BytesIO(base64.b64decode(data["thumbnail"][len(prefix) :]))) as thumb:
        assert thumb.format == "JPEG" and thumb.size == (256, 128)
    # What it writes, it reads back
    path = tmp_path / "pic.naiv4vibe"
    path.write_text(vibe.dumps(data), encoding="utf-8")
    [again] = vibe.load(path, None, None, V45)
    assert again.strength == 0.5 and caches.vibe_encoding(again, V45) == "E"


def test_vibe_without_image_keeps_only_its_encoding(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(_png())
    [source] = vibe.load(img, None, None, V45)
    data = vibe.export(source, V45, "E", without_image=True)
    assert data["type"] == "encoding" and data["id"] == hashlib.sha256(b"E").hexdigest()
    assert "image" not in data and "thumbnail" not in data
    assert data["encodings"] == {"v4-5full": {"unknown": {"encoding": "E", "params": {"information_extracted": 0.7}}}}
    path = tmp_path / "pic.naiv4vibe"
    path.write_text(vibe.dumps(data), encoding="utf-8")
    [again] = vibe.load(path, None, None, V45)
    assert again.image_b64 is None and caches.vibe_encoding(again, V45) == "E"


def test_reexport_keeps_the_entry_and_drops_unknown_fields(tmp_path):
    path = tmp_path / "v.naiv4vibe"
    path.write_text(json.dumps({**vibe_data("kept", info=1.0), "extra": 1}), encoding="utf-8")
    [source] = vibe.load(path, 0.3, None, V45)
    data = vibe.export(source, V45, caches.vibe_encoding(source, V45), without_image=False)
    assert "extra" not in data and data["name"] == "kept" and data["id"] == hashlib.sha256(b"ENC").hexdigest()
    assert data["importInfo"]["strength"] == 0.3
    # Written as JSON.stringify would, with whole numbers bare
    assert '"information_extracted": 1,' in vibe.dumps(data)


def test_image_vibe_needs_its_encoding_to_drop_the_image(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(_png())
    [source] = vibe.load(img, None, None, V45)
    with pytest.raises(errors.NaiError, match="no encoding"):
        vibe.export(source, V45, None, without_image=True)


def test_bundle_written_into_a_png_reads_back(tmp_path):
    entries = [vibe_data("one"), vibe_data("two", encoding="TWO")]
    out = vibe.embed(_png(), vibe.dumps(vibe.bundle(entries)))
    # Right after IHDR, as the official export places it
    kinds = [kind for kind, *_ in decode.png_chunks(out)]
    assert kinds[:2] == [b"IHDR", b"iTXt"]
    path = tmp_path / "two.naiv4vibebundle.png"
    path.write_bytes(out)
    assert [caches.vibe_encoding(s, V45) for s in vibe.load(path, None, None, V45)] == ["ENC", "TWO"]
    with Image.open(path) as img:
        assert img.size == (64, 48)


@pytest.mark.parametrize("kind", ["unknown", None])
def test_vibe_of_another_type_refused(tmp_path, kind):
    data = vibe_data("x")
    data["type"] = kind
    path = tmp_path / "odd.naiv4vibe"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(errors.NaiError) as e:
        vibe.load(path, None, None, V45)
    assert e.value.code == errors.BAD_ARGS


def test_embed_replaces_a_bundle_already_held(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    once = vibe.embed(buf.getvalue(), json.dumps(vibe_data("first")))
    twice = vibe.embed(once, json.dumps(vibe_data("second")))
    kinds = [kind for kind, *_ in decode.png_chunks(twice)]
    assert kinds.count(b"iTXt") == 1
    path = tmp_path / "held.png"
    fs.create(path, twice)
    [source] = vibe.load(path, None, None, V45)
    assert source.name == "second"


_WRAPPED_MASK = "\n".join(["TUFT", "Sw=="])


@pytest.mark.parametrize("mask", ["", _WRAPPED_MASK])
def test_an_empty_or_wrapped_mask_loads_and_finds_its_encoding_as_official(tmp_path, mask):
    path = tmp_path / "masked.naiv4vibe"
    path.write_text(json.dumps(vibe_data("masked", mask=mask)), encoding="utf-8")
    [source] = vibe.load(path, None, None, V45)
    # The official client keys encodings by the mask as stored, empty or wrapped
    assert source.mask == mask and caches.vibe_encoding(source, V45) == "ENC"


@pytest.mark.parametrize("argv", [["vibe"], ["gen", "--model", "v4.5", "--prompt", "x", "--vibe"]], ids=["vibe", "gen"])
@pytest.mark.parametrize("mask", ["", _WRAPPED_MASK])
def test_every_paid_encode_sends_the_mask_as_stored(capsys, with_token, monkeypatch, tmp_path, argv, mask):
    sent = []
    monkeypatch.setattr(
        FakeClient, "encode_vibe", lambda self, image, model, info=1.0, mask=None: sent.append(mask) or "ENC"
    )
    picture = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(picture, format="PNG")
    image = base64.b64encode(picture.getvalue()).decode()
    path = tmp_path / "masked.naiv4vibe"
    entry = vibe_entry(image=image, importInfo={"information_extracted": 0.7, "strength": 0.5, "mask": mask})
    path.write_text(json.dumps(entry), encoding="utf-8")
    code, out = run(capsys, *argv, str(path), "--yes")
    assert code == 0 and sent == [mask], out


def test_a_vibe_source_past_the_pixel_limit_is_refused_as_too_large(capsys, tmp_path, monkeypatch):
    source = tmp_path / "big.png"
    Image.new("RGB", (256, 256), "red").save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 256 * 256 // 3)
    code, out = run(capsys, "vibe", str(source), "--dry-run")
    assert (code, out["code"]) == (1, "bad_args") and "damaged" not in out["message"], out


@pytest.mark.parametrize("command", ["gen", "vibe"])
def test_all_vibe_masks_are_sendable_before_first_encode(capsys, with_token, monkeypatch, tmp_path, command):
    clients = fake_clients(monkeypatch)
    img = tmp_path / "first.png"
    Image.new("RGB", (64, 64)).save(img)
    other = _vibe_file(tmp_path, image=base64.b64encode(img.read_bytes()).decode(), importInfo={"mask": "TQ\u2003=="})
    args = (
        ["vibe", str(img), str(other)]
        if command == "vibe"
        else ["gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img), "--vibe", str(other)]
    )
    code, out = run(capsys, *args)
    assert code == 1 and out["code"] == errors.BAD_ARGS and "mask" in out["message"]
    assert not any("encode_vibe" in client.calls or "generate" in client.calls for client in clients)


def test_v5_vibe_rejected(capsys, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64)).save(img)
    _, out = run(capsys, "gen", "--vibe", str(img), "--dry-run")
    assert out["code"] == "model_unsupported" and out["feature"] == "vibe"


def test_vibe_command_dry_run_cap_and_cache(capsys, with_token, monkeypatch, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "vibe", str(img), "--dry-run")
    assert code == 0 and out["dry_run"] and out["cost"] == {"estimate": 2, "anlas_left": 1000}
    assert not Path(out["file"]).exists()
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", str(img))
    assert code == 0 and out["cost"] == {"estimate": 2, "spent": 2, "anlas_left": 998} and out["cached"] is False
    assert json.loads(Path(out["file"]).read_text(encoding="utf-8"))["identifier"] == "novelai-vibe-transfer"
    # The same image and settings are already encoded, so nothing is sent or charged
    code, out = run(capsys, "vibe", str(img), "-o", str(tmp_path / "again.naiv4vibe"))
    assert code == 0 and out["cached"] is True and out["cost"] == {"estimate": 0, "spent": 0}
    assert [c.calls for c in clients] == [["account", "encode_vibe", "account"]]
    set_cap(0)
    code, out = run(capsys, "vibe", str(img), "--info", "0.5")
    assert code == 1 and out["code"] == "need_confirm" and out["cost"]["estimate"] == 2
    code, out = run(capsys, "vibe", str(img), "--info", "1.5", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


@pytest.mark.parametrize("suffix", [":abc", ":nan", ":0.5:2", ":0.5:0.5:1"])
def test_bad_vibe_values(capsys, tmp_path, suffix):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--vibe", f"{img}{suffix}", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


def test_precise_reference_fields(capsys, tmp_path):
    img = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--ref", f"{img}:character::0.25", "--dry-run")
    p = out["body"]["parameters"]
    assert code == 0 and p["director_reference_strength_values"] == [1.0]
    assert p["director_reference_secondary_strength_values"] == [0.75]
    code, out = run(capsys, "gen", "--model", "v4.5", "--ref", f"{img}:style:x", "--dry-run")
    assert code == 1 and out["code"] == "bad_args"


@pytest.mark.parametrize(
    ("kind", "sent"),
    [
        ("both", "character&style"),
        ("both-any-outfit", "character&style_anyoutfit"),
        ("character", "character"),
        ("character+outfit", "character_mainoutfit"),
        ("outfit", "costume"),
        ("costume", "costume"),
    ],
)
def test_v5_precise_reference_kinds(capsys, tmp_path, kind, sent):
    img = tmp_path / "r.png"
    Image.new("RGBA", (64, 64)).save(img)
    code, out = run(capsys, "gen", "--prompt", "x", "--ref", f"{img}:{kind}", "--dry-run")
    p = out["body"]["parameters"]
    assert code == 0 and p["director_reference_descriptions"][0]["caption"]["base_caption"] == sent
    assert p["director_reference_secondary_strength_values"] == [0]


def test_v5_reference_is_bucketed_and_keeps_transparency(tmp_path):
    img = tmp_path / "r.png"
    Image.new("RGBA", (70, 120), (10, 20, 30, 0)).save(img)
    v5, v45 = models.resolve("v5"), models.resolve("v4.5")
    sent = Image.open(io.BytesIO(load_precise((precise_ref(str(img)),), v5)[0].image))
    assert sent.size == (768, 1344) and sent.mode == "RGBA" and sent.getpixel((0, 0))[3] == 0
    fitted = Image.open(io.BytesIO(load_precise((precise_ref(str(img)),), v45)[0].image))
    assert fitted.size == (1024, 1536) and fitted.mode == "RGB"


def test_v5_medium_with_references_runs_at_high_effort(capsys, tmp_path):
    img = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v5-medium", "--ref", str(img), "--dry-run")
    assert code == 0 and out["body"]["model"] == "nai-diffusion-5-full"
    assert any("high effort" in w for w in out["warnings"])
    code, out = run(capsys, "gen", "--prompt", "x", "--effort", "medium", "--dry-run")
    assert code == 0 and out["body"]["model"] == "nai-diffusion-5-full-medium"


def test_outfit_reference_needs_v5(capsys, tmp_path):
    img = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(img)
    code, out = run(capsys, "gen", "--prompt", "x", "--model", "v4.5", "--ref", f"{img}:outfit", "--dry-run")
    assert code == 1 and out["code"] == "model_unsupported" and out["feature"] == "outfit_reference"


def test_body_shows_the_vibes_encoded_for_the_request(capsys, with_token, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img), "--show-body")
    assert code == 0 and out["body"]["parameters"]["reference_image_multiple"] == ["<7 base64 characters>"]


def test_encoded_vibe_recorded_when_generation_fails(capsys, with_token, monkeypatch, tmp_path):
    def fail(self, body):
        self.calls.append("generate")
        raise errors.NaiError(errors.NETWORK, "lost the connection", maybe_processed=True)

    monkeypatch.setattr(FakeClient, "generate", fail)
    clients = fake_clients(monkeypatch)
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img))
    assert code == 1 and out["code"] == "network" and out["cost"]["spent"] == 2
    assert clients[0].calls == ["account", "encode_vibe", "generate", "account"]
    (entry,) = ledger.entries()
    assert entry["error"] == "network" and entry["anlas_spent"] == 2


def test_vibe_kept_when_the_cache_cannot_be_written(capsys, with_token, monkeypatch, tmp_path):
    def locked(source, model, encoding):
        raise PermissionError("read-only")

    monkeypatch.setattr(caches, "remember_vibe", locked)
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "vibe", str(img))
    assert code == 0 and Path(out["file"]).exists() and any("cached" in w for w in out["warnings"])
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img), "--show-body")
    assert code == 0 and out["body"]["parameters"]["reference_image_multiple"] == ["<7 base64 characters>"]


def test_vibe_rechecks_the_cache_under_the_lock(capsys, with_token, monkeypatch, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    real_lock = ledger.generation_lock

    @contextmanager
    def lock():
        # Another process encodes the same image while this one waits
        caches.remember_vibe(vibe.load(img, None, None, "nai-diffusion-4-5-full")[0], "nai-diffusion-4-5-full", "E")
        with real_lock():
            yield

    monkeypatch.setattr(ledger, "generation_lock", lock)
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", str(img))
    # Found before the balance is read, so nothing at all is sent
    assert code == 0 and out["cached"] is True and out["cost"] == {"estimate": 0, "spent": 0}
    assert clients[0].calls == [] and list(ledger.entries()) == [] and "warnings" not in out


def test_vibe_output_checked_before_paying(capsys, with_token, monkeypatch, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", str(img), "-o", str(img / "out.naiv4vibe"))
    assert code == 1 and out["code"] == "bad_args" and clients == []


def test_paid_vibe_still_settles_when_generation_never_left(capsys, with_token, monkeypatch, tmp_path):
    def unsent(self, body):
        raise errors.NaiError(errors.NETWORK, "could not connect")

    monkeypatch.setattr(FakeClient, "generate", unsent)
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(img))
    assert code == 1 and out["cost"]["spent"] == 2
    (entry,) = ledger.entries()
    assert entry["error"] == "network" and entry["anlas_spent"] == 2


def test_vibe_normalization_switch(capsys, tmp_path):
    vibe_file = tmp_path / "v.naiv4vibe"
    import_info = {"model": "nai-diffusion-4-5-full", "information_extracted": 1.0, "strength": 0.8}
    data = vibe_entry(
        encodings={"v4-5full": {"unknown": {"encoding": "E", "params": {"information_extracted": 1.0}}}},
        importInfo=import_info,
    )
    vibe_file.write_text(json.dumps(data), encoding="utf-8")
    argv = ["gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(vibe_file), "--vibe", str(vibe_file), "--dry-run"]
    code, out = run(capsys, *argv)
    p = out["body"]["parameters"]
    assert code == 0 and p["reference_strength_multiple"] == [0.5, 0.5] and p["normalize_reference_strength_multiple"]
    code, out = run(capsys, *argv, "--no-normalize-vibe-strength")
    p = out["body"]["parameters"]
    assert p["reference_strength_multiple"] == [0.8, 0.8] and p["normalize_reference_strength_multiple"] is False


def test_damaged_vibe_image_fails_before_any_encoding(capsys, with_token, monkeypatch, tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (64, 64), "red").save(good)
    bad = tmp_path / "bad.naiv4vibe"
    bad.write_text(json.dumps(vibe_entry(image=base64.b64encode(CUT).decode(), encodings={})), encoding="utf-8")
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(good), "--vibe", str(bad))
    assert code == 1 and out["code"] == "bad_args"
    assert all("encode_vibe" not in c.calls for c in clients) and list(ledger.entries()) == []


def test_vibe_command_bundles_strips_and_embeds(capsys, with_token, monkeypatch, tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (64, 64), "red").save(a)
    Image.new("RGB", (64, 64), "blue").save(b)
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", f"{a}:0.4", str(b))
    # Several Vibes make a bundle named by the date, as the official export names it
    today = datetime.now(UTC).date().isoformat()
    assert code == 0 and out["vibes"] == 2 and out["cost"]["spent"] == 4
    assert Path(out["file"]) == tmp_path / f"{today}.naiv4vibebundle"
    assert [c.calls.count("encode_vibe") for c in clients] == [2]
    bundle = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
    assert bundle["identifier"] == "novelai-vibe-transfer-bundle"
    assert [(v["type"], v["importInfo"]["strength"]) for v in bundle["vibes"]] == [("image", 0.4), ("image", 0.6)]
    # Encoded already, so keeping only the encodings costs nothing
    stripped = tmp_path / "set.naiv4vibebundle"
    code, out = run(capsys, "vibe", out["file"], "--no-image", "-o", str(stripped))
    assert code == 0 and out["cached"] is True and out["cost"]["spent"] == 0
    assert [v["type"] for v in json.loads(stripped.read_text(encoding="utf-8"))["vibes"]] == ["encoding", "encoding"]
    # Into a copy of an image, named after it
    cover = tmp_path / "cover.png"
    Image.new("RGB", (32, 32)).save(cover)
    code, out = run(capsys, "vibe", str(stripped), "--into", str(cover))
    assert code == 0 and Path(out["file"]) == tmp_path / "cover.naiv4vibebundle.png"
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", out["file"], "--dry-run")
    assert code == 0 and len(out["body"]["parameters"]["reference_image_multiple"]) == 2
    # A name given and taken gets the next number, as every name does
    code, out = run(capsys, "vibe", str(stripped), "-o", str(stripped))
    assert code == 0 and Path(out["file"]) == stripped.with_name(stripped.stem + "-1" + stripped.suffix)
    assert sum(c.calls.count("encode_vibe") for c in clients) == 2


def test_vibe_command_reexports_with_new_settings(capsys, tmp_path):
    source = tmp_path / "v.naiv4vibe"
    entry = vibe_entry(encodings={"v4-5full": {"unknown": {"encoding": "E"}}}, importInfo={"strength": 0.6})
    source.write_text(json.dumps(entry), encoding="utf-8")
    code, out = run(capsys, "vibe", f"{source}:0.3", "-o", str(tmp_path / "w.naiv4vibe"))
    data = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
    assert code == 0 and out["cached"] is True and data["importInfo"]["strength"] == 0.3 and data["id"] == entry["id"]
    # Without its image it cannot be encoded with other settings
    code, out = run(capsys, "vibe", f"{source}::0.5", "-o", str(tmp_path / "x.naiv4vibe"))
    assert code == 1 and out["code"] == "bad_args"


def test_vibes_from_an_image_rerun(capsys, tmp_path):
    img = tmp_path / "made.png"
    v45_image(
        img,
        reference_image_multiple=["E1", "E22"],
        reference_strength_multiple=[0.4, 0.8],
        reference_information_extracted_multiple=[1, 0.5],
    )
    argv = ["gen", "--model", "v4.5", "--prompt", "x", "--vibes-from", str(img), "--dry-run"]
    code, out = run(capsys, *argv, "--no-normalize-vibe-strength")
    p = out["body"]["parameters"]
    assert code == 0 and out["cost"]["estimate"] == 0
    assert p["reference_image_multiple"] == ["<2 base64 characters>", "<3 base64 characters>"]
    assert p["reference_strength_multiple"] == [0.4, 0.8]
    # Kept for the model that made the image
    code, out = run(capsys, *argv, "--model", "v4.5-curated")
    assert code == 1 and out["code"] == "bad_args"
    code, record = run(capsys, "meta", str(img))
    assert record["stored_vibes"] == 2 and f"--vibes-from={img.resolve()}" in record["gen_args"]
    assert record["warnings"] == []
    code, again = run(capsys, "gen", *record["gen_args"], "--dry-run")
    assert code == 0 and again["body"]["parameters"]["reference_image_multiple"] == p["reference_image_multiple"]
    # Variations carry them too, as the official client sends the image's
    code, again = run(capsys, "variations", "--image", str(img), "--dry-run")
    assert code == 0 and again["body"]["parameters"]["reference_image_multiple"] == p["reference_image_multiple"]


def test_the_same_vibe_twice_is_encoded_once(capsys, with_token, monkeypatch, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    code, out = run(capsys, "vibe", str(img), str(img), "--dry-run")
    assert code == 0 and out["cost"]["estimate"] == 2
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "vibe", str(img), str(img))
    assert code == 0 and out["cost"]["spent"] == 2 and sum(c.calls.count("encode_vibe") for c in clients) == 1
    other = tmp_path / "w.png"
    Image.new("RGB", (64, 64), "blue").save(other)
    clients.clear()
    code, out = run(capsys, "gen", "--model", "v4.5", "--prompt", "x", "--vibe", str(other), "--vibe", str(other))
    assert code == 0 and sum(c.calls.count("encode_vibe") for c in clients) == 1


def test_a_damaged_holder_image_is_refused_before_paying(capsys, with_token, monkeypatch, tmp_path):
    img = tmp_path / "v.png"
    Image.new("RGB", (64, 64), "red").save(img)
    holder = tmp_path / "h.png"
    holder.write_bytes(b"\x89PNG\r\n\x1a\n")
    clients = fake_clients(monkeypatch)
    for extra in ([], ["--dry-run"]):
        code, out = run(capsys, "vibe", str(img), "--into", str(holder), *extra)
        assert code == 1 and out["code"] == "bad_args"
    assert all("encode_vibe" not in c.calls for c in clients)
    # A CMYK JPEG holds the Vibes as a PNG
    jpeg = tmp_path / "h.jpg"
    Image.new("CMYK", (32, 32)).save(jpeg)
    code, out = run(capsys, "vibe", str(img), "--into", str(jpeg))
    assert code == 0 and Path(out["file"]).read_bytes().startswith(b"\x89PNG")
