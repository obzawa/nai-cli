"""Grids: axes parsed with the official client's syntax, one job per cell, and the sheet of the results."""

import json
from pathlib import Path

import pytest
from helpers import FakeClient, fake_clients, run, set_cap
from PIL import Image
from samples import nai_image, png_bytes

from nai_cli import cli, errors, grid, sheet


def _grid(capsys, *argv):
    code = cli.main(["grid", *argv])
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def _axis(*entries):
    return grid.parse_axis([tuple(e.split("=", 1)) for e in entries])


@pytest.mark.parametrize(
    ("entry", "values"),
    [
        ("steps=20-28:4", ["20", "24", "28"]),
        ("steps=20, 23,", ["20", "23"]),
        ("cfg=4-5:0.25", ["4", "4.25", "4.5", "4.75", "5"]),
        ("cfg=0.1-0.3:0.1", ["0.1", "0.2", "0.3"]),
        # Rounded half up, as the official client rounds
        ("steps=1-2:0.5", ["1", "2", "2"]),
        ("scale=5.5", ["5.5"]),
        ("sampler=k_euler, k_dpmpp_2m", ["k_euler", "k_dpmpp_2m"]),
        ("prompt=cat\n\n dog ", ["cat", "dog"]),
    ],
)
def test_axis_values(entry, values):
    axis = _axis(entry)
    assert [step[0] for step in axis.steps] == values


def test_random_seeds_and_params_stepping_together():
    axis = _axis("seed=1,r2", "steps=20,28")
    seeds = [step[0] for step in axis.steps]
    assert seeds[0] == "1" and len(seeds) == 3 and all(s.isdigit() for s in seeds)
    # The shorter list keeps its last value
    assert [step[1] for step in axis.steps] == ["20", "28", "28"]
    assert axis.title == "seed, steps" and axis.labels[0] == "1, 20"


@pytest.mark.parametrize(
    "entries",
    [
        ["steps=28-20"],
        ["steps=20-28:0"],
        ["steps=1-41"],
        ["cfg=abc"],
        ["seed=r41"],
        ["width=832"],
        ["model=v9"],
        ["sampler=nope"],
        ["size=1x1"],
        ["steps=20", "steps=28"],
    ],
)
def test_bad_axes_are_refused(entries):
    with pytest.raises(errors.NaiError):
        _axis(*entries)


def test_cells_run_row_by_row_with_the_values_last():
    g = grid.build(["gen", "--prompt", "cat", "--steps", "10"], _axis("steps=20,28"), _axis("cfg=4,6"), cli._parse_job)
    assert [c.job.id for c in g.cells] == ["x1y1", "x2y1", "x1y2", "x2y2"]
    last = g.cells[-1].job.argv
    assert last[-2:] == ["--steps=28", "--cfg=6"]
    # One seed for the whole grid, so only the axes change
    seeds = {next(a for a in c.job.argv if a.startswith("--seed=")) for c in g.cells}
    assert len(seeds) == 1
    kept = grid.build(["gen", "--prompt", "cat", "--seed", "5"], _axis("steps=20,28"), None, cli._parse_job)
    assert not any(a.startswith("--seed=") for a in kept.cells[0].job.argv)


@pytest.mark.parametrize(
    ("argv", "x", "y"),
    [
        (["meta", "a.png"], "steps=20", None),
        (["gen"], None, None),
        (["gen"], "steps=20", "steps=28"),
        (["gen"], "steps=1-8", "cfg=1-6"),
    ],
)
def test_bad_grids_are_refused(argv, x, y):
    with pytest.raises(errors.NaiError):
        grid.build(argv, _axis(x) if x else None, _axis(y) if y else None, cli._parse_job)


def test_grid_runs_each_cell_and_saves_a_sheet(capsys, with_token, monkeypatch, tmp_path):
    sizes = iter([(64, 96), (64, 96), (96, 64)])
    monkeypatch.setattr(FakeClient, "generate", lambda self, body: [png_bytes(*next(sizes))])
    code, lines = _grid(capsys, "--x", "steps=20,24,28", "gen", "--prompt", "cat", "-o", str(tmp_path))
    summary = lines[-1]
    assert code == 0 and [line["job"] for line in lines[:-1]] == ["x1", "x2", "x3"]
    assert summary["type"] == "summary" and summary["batch"]["succeeded"] == 3
    sheet = Path(summary["grid"]["sheet"])
    assert summary["grid"]["columns"] == 3 and sheet.name.endswith("_grid.png")
    with Image.open(sheet) as img:
        assert img.width > 3 * 64 and img.height > 96


def test_grid_dry_run_prices_every_cell(capsys, with_token):
    code, lines = _grid(capsys, "--dry-run", "--x", "steps=20,28", "--y", "cfg=4,5", "gen", "--prompt", "cat")
    assert code == 0 and len(lines) == 5 and all(line["dry_run"] for line in lines[:-1])
    assert "sheet" not in lines[-1]["grid"] and "estimate" in lines[-1]["batch"]


def test_grid_varies_every_precise_reference(capsys, with_token, tmp_path):
    ref = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(ref)
    argv = ["--dry-run", "--x", "pr_strength=0.5,2", "gen", "--prompt", "cat", "--ref", f"{ref}::1", "--ref", str(ref)]
    code, lines = _grid(capsys, *argv)
    strengths = [line["body"]["parameters"]["director_reference_strength_values"] for line in lines[:-1]]
    assert code == 0 and strengths == [[0.5, 0.5], [2, 2]]
    assert _axis("pr_fidelity=0.2,0.4").params == (grid.PARAMS["ref_fidelity"],)


def test_bad_grid_prints_a_summary(capsys):
    code, lines = _grid(capsys, "--x", "steps=nope", "gen", "--prompt", "cat")
    assert code == 1 and lines[-1]["type"] == "summary" and lines[-1]["code"] == "bad_args"


def test_batch_refuses_a_grid_job(capsys, tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(["grid", "--x", "steps=20", "gen"]) + "\n", encoding="utf-8")
    code = cli.main(["batch", str(jobs)])
    assert code == 1 and "batch or grid" in json.loads(capsys.readouterr().out)["message"]


def _jobs(lines):
    return [line for line in lines if line.get("type") == "job"]


def test_each_cell_is_one_image(capsys):
    code, lines = _grid(capsys, "--dry-run", "--x", "steps=20,28", "gen", "--prompt", "x")
    assert code == 0 and [j["body"]["parameters"]["n_samples"] for j in _jobs(lines)] == [1, 1]
    for extra in (["--n", "4"], ["-o", "one.png"], ["--random-prompt"]):
        code, out = run(capsys, "grid", "--dry-run", "--x", "steps=20,28", "gen", "--prompt", "x", *extra)
        assert code == 1 and out["code"] == "bad_args", extra


def test_axis_text_is_sent_as_written(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "secret.txt").write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "prompts.txt").write_text("@secret.txt\ncat", encoding="utf-8")
    for entry in ("prompt=@@secret.txt", "prompt=@prompts.txt"):
        code, lines = _grid(capsys, "--dry-run", "--x", entry, "gen")
        prompts = [j["prompt"] for j in _jobs(lines)]
        assert code == 0 and prompts[0].startswith("@secret.txt") and "PRIVATE" not in json.dumps(lines), entry


def test_cells_share_a_seed_unless_the_image_gives_one(capsys, tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, seed=77)
    code, lines = _grid(capsys, "--dry-run", "--x", "steps=20,28", "gen", "--from-image", str(img))
    seeds = {j["seed"] for j in _jobs(lines)}
    assert code == 0 and len(seeds) == 1 and seeds != {77}
    code, lines = _grid(capsys, "--dry-run", "--x", "steps=20,28", "gen", "--from-image", str(img), "--import", "all")
    assert code == 0 and {j["seed"] for j in _jobs(lines)} == {77}


@pytest.mark.parametrize(
    "options",
    [
        ["-o", "cell.png"],
        ["-ocell.png"],
        ["--output=cell.png"],
        ["--n", "2"],
        ["--n=2"],
        ["--random-prompt"],
        # Shortened names are not read as the options they start, here or in a command
        ["--outp", "folder"],
        ["--steps", "abc"],
        ["--", "stray"],
    ],
)
def test_options_that_cannot_hold_are_refused_before_any_cell_sends(capsys, with_token, monkeypatch, options):
    clients = fake_clients(monkeypatch)
    for extra in (["--dry-run"], []):
        code, lines = _grid(capsys, *extra, "--x", "steps=20,21", "gen", "--prompt", "cat", *options)
        assert code == 1 and lines[-1]["code"] == "bad_args" and len(lines) == 1, (options, lines)
    assert not any("generate" in c.calls for c in clients)


@pytest.mark.parametrize(
    "argv",
    [
        ["--x", "steps=28,60", "gen", "--prompt", "cat"],
        ["--x", "model=v4,v5", "gen", "--prompt", "cat", "--legacy-uc"],
    ],
)
def test_a_cell_the_command_refuses_stops_the_grid_before_any_is_sent(capsys, with_token, monkeypatch, argv):
    clients = fake_clients(monkeypatch)
    code, lines = _grid(capsys, *argv)
    assert code == 1 and lines[-1]["type"] == "summary" and "x2" in lines[-1]["message"], lines
    assert not any("generate" in c.calls for c in clients)


@pytest.mark.parametrize("dry", [False, True])
def test_grid_checks_each_cells_request_limit(capsys, with_token, monkeypatch, dry):
    clients = fake_clients(monkeypatch)
    set_cap(0)
    code, lines = _grid(capsys, *(["--dry-run"] if dry else []), "--x", "steps=28,35", "gen", "--prompt", "x")
    assert code == 1
    if dry:
        assert lines[1]["code"] == "need_confirm"
    else:
        assert len(lines) == 1 and lines[0]["code"] == "need_confirm" and "x2" in lines[0]["message"]
    assert not any("generate" in c.calls for c in clients)


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_sheet_failure_preserves_the_paid_grid_summary(capsys, with_token, monkeypatch, failure):
    def write(*args):
        raise failure("sheet not saved")

    def generate(self, body):
        self.anlas -= 7
        return [png_bytes()]

    monkeypatch.setattr(FakeClient, "generate", generate)
    monkeypatch.setattr(sheet, "render", write)
    code, lines = _grid(capsys, "--x", "steps=20,21", "gen", "--prompt", "cat")
    assert code == (130 if failure is KeyboardInterrupt else 0)
    assert lines[-1]["type"] == "summary" and lines[-1]["batch"]["succeeded"] == 2
    assert lines[-1]["batch"]["spent"] == 14 and lines[-1]["grid"]["sheet"] is None


def test_grid_base_parse_error_keeps_command_usage(capsys):
    code, (summary,) = _grid(capsys, "--dry-run", "--x", "steps=20,21", "gen", "--unknown")
    assert (code, summary["type"], summary["code"]) == (1, "summary", "bad_args")
    assert summary["usage"].startswith("usage: nai gen ")
