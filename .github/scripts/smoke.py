"""Runs the installed nai command once per offline feature in a throwaway home, checking exit codes and JSON.

Usage: smoke.py TOKENIZER_DIR, the folder tokenizers.py downloads into; no case may open a connection.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, PngImagePlugin

SETTINGS = {
    "prompt": "1girl, 猫, garden",
    "uc": "lowres",
    "steps": 23,
    "scale": 5.0,
    "seed": 1234,
    "sampler": "k_euler_ancestral",
    "noise_schedule": "karras",
    "width": 832,
    "height": 1216,
    "n_samples": 1,
}


# Loaded by every Python the cases start, ahead of any other sitecustomize, as PYTHONPATH comes first
_BLOCK_NETWORK = """
import os, socket

def _refuse(*args, **kwargs):
    with open(os.environ["SMOKE_NETWORK_LOG"], "a", encoding="utf-8") as log:
        log.write(repr(args[:2]) + "\\n")
    raise ConnectionRefusedError("smoke cases run offline")

socket.getaddrinfo = _refuse
socket.create_connection = _refuse
socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = _isolated_env(root)
        _install_tokenizers(Path(sys.argv[1]), env)
        if not _network_blocked(env):
            print("FAIL the network block did not load")
            return 1
        failures = [name for name, argv, expect in _cases(root) if not _check(name, argv, expect, env)]
    print(f"{len(failures)} failed: {', '.join(failures)}" if failures else "all passed")
    return 1 if failures else 0


def _isolated_env(root: Path) -> dict[str, str]:
    # NOVELAI_TOKEN too, so no case can reach an account
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("NAI", "NOVELAI", "XDG_", "PYTHON"))}
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        env[name] = str(root / "home" / name.lower())
        Path(env[name]).mkdir(parents=True)
    guard = root / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(_BLOCK_NETWORK, encoding="utf-8")
    env["PYTHONPATH"] = str(guard)
    env["SMOKE_NETWORK_LOG"] = str(root / "network.log")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _install_tokenizers(source: Path, env: dict[str, str]) -> None:
    """Into the data folder the isolated home gives, found by the same Python, so counting needs no download."""
    script = "from nai_cli import caches, tokenizer\nfor kind in tokenizer.FILES: print(caches.tokenizer_path(kind))"
    found = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
    for line in found.stdout.splitlines():
        target = Path(line)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / target.name, target)


def _network_blocked(env: dict[str, str]) -> bool:
    probe = "import socket\ntry:\n    socket.getaddrinfo('localhost', 80)\nexcept ConnectionRefusedError:\n    print('blocked')"
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    log = Path(env["SMOKE_NETWORK_LOG"])
    blocked = run.stdout.strip() == "blocked" and log.is_file()
    log.unlink(missing_ok=True)
    return blocked


def _sample(path: Path) -> Path:
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "NovelAI")
    info.add_text("Source", "NovelAI Diffusion V4.5 4BDE2A90")
    info.add_text("Comment", json.dumps(SETTINGS, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (832, 1216), (90, 140, 200)).save(path, pnginfo=info)
    return path


def _cases(root: Path) -> list[tuple[str, list[str], str]]:
    # Non-ASCII and spaced folders, as users name them
    work = root / "作品 folder"
    pic = _sample(work / "猫 sample.png")
    out = work / "出力"
    jobs = work / "jobs.jsonl"
    jobs.write_text(
        '["gen", "--prompt", "1girl, 猫"]\n{"args": ["gen", "--prompt", "dog", "--model", "v4.5"]}\n', encoding="utf-8"
    )
    art = work / "pixel art.png"
    Image.effect_noise((32, 32), 80).convert("RGB").resize((256, 256), Image.Resampling.NEAREST).save(art)
    small = work / "small.png"
    Image.new("RGB", (512, 512), (200, 120, 90)).save(small)
    region = "rect:0.1,0.1,0.5,0.5"
    return [
        ("version", ["--version"], "text"),
        ("config init", ["config", "init"], "ok"),
        ("config", ["config"], "ok"),
        ("models", ["models"], "ok"),
        ("account list", ["account", "list"], "ok"),
        ("stats", ["stats"], "ok"),
        ("local status", ["local", "status"], "ok"),
        ("cost", ["cost", "--size", "1024x1536", "--steps", "28"], "ok"),
        ("lint", ["lint", "--prompt", "1girl,, {{猫}} , garden"], "ok"),
        ("gen dry run", ["gen", "--prompt", "1girl, 猫", "--char", "girl, smile", "--dry-run"], "ok"),
        ("gen v4.5 dry run", ["gen", "--model", "v4.5", "--prompt", "x", "--sampler", "dpm++ 2m", "--dry-run"], "ok"),
        ("gen from image", ["gen", "--from-image", str(pic), "--import", "all", "--dry-run"], "ok"),
        ("gen bad sampler", ["gen", "--prompt", "x", "--sampler", "ddim", "--dry-run"], "bad_args"),
        ("i2i dry run", ["i2i", "--image", str(pic), "--prompt", "x", "--dry-run"], "ok"),
        ("inpaint dry run", ["inpaint", "--image", str(pic), "--region", region, "--prompt", "x", "--dry-run"], "ok"),
        (
            "inpaint preview",
            ["inpaint", "--image", str(pic), "--region", region, "--prompt", "x", "--preview", "-o", str(out)],
            "ok",
        ),
        ("enhance dry run", ["enhance", "--image", str(pic), "--dry-run"], "ok"),
        ("variations dry run", ["variations", "--image", str(pic), "--dry-run"], "ok"),
        ("upscale dry run", ["upscale", str(small), "--dry-run"], "ok"),
        ("vibe dry run", ["vibe", str(small), "--dry-run"], "ok"),
        ("grid dry run", ["grid", "--dry-run", "--x", "seed=1,2", "--y", "cfg=5,6", "gen", "--prompt", "x"], "lines"),
        ("batch dry run", ["batch", "--dry-run", str(jobs)], "lines"),
        ("meta", ["meta", str(pic)], "ok"),
        ("meta folder", ["meta", str(work)], "lines"),
        ("strip", ["strip", str(pic), "-o", str(out / "stripped.png")], "ok"),
        ("strip folder", ["strip", str(work), "-r", "--rename", "-o", str(out / "shared")], "ok"),
        ("redact", ["redact", str(pic), "--region", "rect:0.1,0.1,0.5,0.5", "-o", str(out / "redacted.png")], "ok"),
        ("watermark", ["watermark", str(pic), "-o", str(out / "marked.png")], "ok"),
        ("watermark keep", ["watermark", str(pic), "--keep-metadata", "-o", str(out / "kept.png")], "ok"),
        ("meta of kept", ["meta", str(out / "kept.png")], "ok"),
        ("pixel-snap", ["tool", "pixel-snap", str(art), "-o", str(out / "snapped.png")], "ok"),
        # Both stop before any request: tags has no token here, interrogate no tagger model
        ("tags without a token", ["tags", "あお", "blue h"], "error"),
        ("interrogate without a model", ["interrogate", str(pic)], "error"),
        ("missing image", ["meta", str(work / "none.png")], "error"),
    ]


def _check(name: str, argv: list[str], expect: str, env: dict[str, str]) -> bool:
    run = subprocess.run(["nai", *argv], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    log = Path(env["SMOKE_NETWORK_LOG"])
    problem = _problem(run, expect)
    if log.is_file():
        problem = f"tried to connect: {log.read_text(encoding='utf-8').strip()}"
        log.unlink()
    print(f"{'FAIL' if problem else 'ok  '} {name}{': ' + problem if problem else ''}")
    if problem:
        print(run.stdout[-2000:], run.stderr[-2000:], sep="\n")
    return not problem


def _problem(run: subprocess.CompletedProcess[str], expect: str) -> str | None:
    if "Traceback" in run.stderr or "Traceback" in run.stdout:
        return "traceback"
    if expect == "text":
        return None if run.returncode == 0 and run.stdout.strip() else f"exit {run.returncode}"
    try:
        lines = [json.loads(run.stdout)] if expect != "lines" else [json.loads(x) for x in run.stdout.splitlines()]
    except ValueError:
        return "output is not JSON"
    last = lines[-1]
    if expect == "bad_args":
        return None if run.returncode == 1 and last.get("code") == "bad_args" else f"expected bad_args, got {last}"
    if expect == "error":
        return None if run.returncode != 0 and last.get("ok") is False else f"expected an error, got {last}"
    return None if run.returncode == 0 and last.get("ok") is True else f"exit {run.returncode}: {last.get('message')}"


if __name__ == "__main__":
    sys.exit(main())
