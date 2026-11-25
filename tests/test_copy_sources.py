"""Sharing copies of folders of images, and copies named by date and number instead of after their source."""

import json
import re
from datetime import datetime
from pathlib import Path

from helpers import run, run_batch
from PIL import Image
from samples import nai_image

from nai_cli import cli, config, fs, naming


def _tree(root: Path) -> None:
    (root / "sub" / "deep").mkdir(parents=True)
    nai_image(root / "a.png", w=64, h=64)
    nai_image(root / "sub" / "b.png", w=64, h=64)
    Image.new("RGB", (8, 8), "red").save(root / "sub" / "c.WEBP")
    Image.new("RGB", (8, 8), "blue").save(root / "sub" / "deep" / "d.jpg")
    (root / "notes.txt").write_text("not an image")


def _made(out):
    return [Path(f["file"]) for f in out["files"]]


def test_a_folder_stands_for_its_images_and_r_for_those_inside(capsys, tmp_path):
    src = tmp_path / "src"
    _tree(src)
    code, out = run(capsys, "strip", str(src), "-o", str(tmp_path / "flat"))
    assert code == 0, out
    assert [f["source"] for f in out["files"]] == [str(src / "a.png")]
    code, out = run(capsys, "strip", str(src), "-r", "-o", str(tmp_path / "shared"))
    assert code == 0, out
    # Each copy keeps its source's place under the -o folder
    assert [p.relative_to(tmp_path / "shared").as_posix() for p in _made(out)] == [
        "a.png",
        "sub/b.png",
        "sub/c.png",
        "sub/deep/d.png",
    ]
    assert cli.main(["meta", str(tmp_path / "shared"), "-r"]) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["meta"] == {"images": 4, "without_settings": 4, "damaged": 0, "failed": 0}, summary


def test_folder_copies_without_o_go_to_the_output_folder(capsys, tmp_path):
    src = tmp_path / "src"
    _tree(src)
    code, out = run(capsys, "watermark", str(src), str(src / "sub" / "b.png"), "-r")
    assert code == 0, out
    names = sorted(p.name for p in _made(out))
    # The same image given twice is copied twice, the second under the next number
    assert names == [
        "a_watermarked.png",
        "b_watermarked-1.png",
        "b_watermarked.png",
        "c_watermarked.png",
        "d_watermarked.png",
    ]
    output_dir = config.load(config.default_path()).output_dir
    assert all(p.is_relative_to(output_dir) for p in _made(out))


def test_folder_input_is_refused_where_it_cannot_work(capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("x")
    code, out = run(capsys, "strip", str(empty))
    assert (code, out["code"]) == (1, "bad_args") and "no images" in out["message"]
    one = tmp_path / "one"
    one.mkdir()
    nai_image(one / "a.png", w=64, h=64)
    # A folder may hold more images by the time the copies are made, so it never takes a file name
    code, out = run(capsys, "redact", str(one), "--region", "rect:0,0,0.5,0.5", "-o", str(tmp_path / "x.png"))
    assert (code, out["code"]) == (1, "bad_args") and "-o names one file" in out["message"]


def test_rename_names_copies_by_date_and_number(capsys, tmp_path):
    src = tmp_path / "src"
    _tree(src)
    shared = tmp_path / "shared"
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(["strip", str(src), "-r", "-o", str(shared), "--rename"]) + "\n", encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    planned = lines[0]
    assert code == 0 and planned["dry_run"] and not shared.exists(), lines
    code, out = run(capsys, "strip", str(src), "-r", "-o", str(shared), "--rename")
    assert code == 0 and "warnings" not in out, out
    made = [p.relative_to(shared).as_posix() for p in _made(out)]
    day = made[0][:8]
    assert re.fullmatch(r"[0-9]{8}", day)
    # Numbered in each folder on its own, in the order of the sources
    assert made == [f"{day}-0001.png", f"sub/{day}-0001.png", f"sub/{day}-0002.png", f"sub/deep/{day}-0001.png"]
    assert [f["file"] for f in planned["files"]] == [f["file"] for f in out["files"]]
    # A second run counts on from the highest number of the day in the folder
    code, out = run(capsys, "strip", str(src / "a.png"), "-o", str(shared), "--rename")
    assert _made(out)[0].name == f"{day}-0002.png"
    code, out = run(capsys, "strip", str(src / "a.png"), "-o", str(tmp_path / "x.png"), "--rename")
    assert (code, out["code"]) == (1, "bad_args") and "--rename" in out["message"]


def test_day_numbers_count_on_from_the_highest_of_the_day(tmp_path):
    for name in ("20261125-0007.webp", "20261125-0003.png", "20261124-0099.png", "20261125-12.png", "x.png"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "20261125-0009-1.png").write_bytes(b"")
    days = fs.DayNumbers(datetime(2026, 11, 25, 23, 59))
    assert days.next_in(tmp_path).name == "20261125-0008.png"
    assert days.next_in(tmp_path).name == "20261125-0009.png"
    assert days.next_in(tmp_path / "new").name == "20261125-0001.png"


def test_copies_named_after_a_download_name_warn(capsys, tmp_path):
    source = tmp_path / "1girl, smile s-123456 (2).png"
    nai_image(source, w=64, h=64)
    plain = tmp_path / "plain.png"
    nai_image(plain, w=64, h=64)
    code, out = run(capsys, "strip", str(source), str(plain))
    assert code == 0 and len(out["warnings"]) == 1 and "a copy is" in out["warnings"][0], out
    assert "--rename" in out["warnings"][0]
    code, out = run(capsys, "strip", str(source), "--rename")
    assert code == 0 and "warnings" not in out, out
    code, out = run(capsys, "strip", str(plain))
    assert "warnings" not in out, out


def test_the_official_download_name_is_recognised():
    assert naming.is_download_name("1girl, smile s-123")
    assert naming.is_download_name("1girl s-123 (2)")
    assert naming.is_download_name("1girl s-123 (2) (1)")
    assert naming.is_download_name(naming.fallback(naming.Meta(prompt="1girl", seed=42)))
    assert not naming.is_download_name("1girl s-")
    assert not naming.is_download_name("120501_123")
    assert not naming.is_download_name("pics-123")
