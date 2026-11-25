"""Installed fonts offered for a watermark: found in the font folders by name, in an order set by the text's script."""

import subprocess
import sys

import pytest

from nai_cli import fonts, share

_TABLE = {
    "latin": ("latin.ttf",),
    "zh": ("zh.ttc",),
    "ja": ("ja.ttc",),
    "ko": ("ko.ttf",),
    "symbols": ("symbols.ttf",),
}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    nested = tmp_path / "sub"
    nested.mkdir()
    for name in ("latin.ttf", "zh.ttc", "ko.ttf", "symbols.ttf"):
        (tmp_path / name).write_bytes(b"")
    # Names are matched in any case, as Windows and macOS file systems do, and in folders inside folders
    (nested / "JA.TTC").write_bytes(b"")
    monkeypatch.setattr(fonts, "_SHIPPED", dict.fromkeys(("win32", "darwin", "linux"), _TABLE))
    monkeypatch.setattr(fonts, "_folders", lambda: [tmp_path / "absent", tmp_path])
    monkeypatch.setattr(fonts, "_fontconfig", lambda text: [])
    return tmp_path


@pytest.mark.parametrize(
    ("text", "order"),
    [
        ("作者", ["latin.ttf", "zh.ttc", "JA.TTC", "ko.ttf", "symbols.ttf"]),
        ("作者です", ["latin.ttf", "JA.TTC", "zh.ttc", "ko.ttf", "symbols.ttf"]),
        ("작가", ["latin.ttf", "ko.ttf", "zh.ttc", "JA.TTC", "symbols.ttf"]),
        ("café", ["latin.ttf", "zh.ttc", "JA.TTC", "ko.ttf", "symbols.ttf"]),
    ],
)
def test_fonts_are_offered_for_the_script_first(folder, text, order):
    assert [p.name for p in fonts.offered(text)] == order


def test_fonts_are_looked_for_only_when_asked(monkeypatch):
    def walked():
        raise AssertionError("font folders walked")

    monkeypatch.setattr(fonts, "_installed", walked)
    fonts.offered("作者")


def test_fontconfig_adds_its_picks_after_the_shipped_fonts(folder, monkeypatch):
    picked = [folder / "zh.ttc", folder / "other.otf"]
    monkeypatch.setattr(fonts, "_fontconfig", lambda text: picked)
    offered = [p.name for p in fonts.offered("作者")]
    if sys.platform in ("win32", "darwin"):
        assert "other.otf" not in offered
    else:
        assert offered[-1] == "other.otf" and offered.count("zh.ttc") == 1


def test_fontconfig_is_asked_for_the_text_and_its_absence_is_no_error(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="/f/a.ttc\n/f/a.ttc\n/f/b.ttf\n")

    monkeypatch.setattr(fonts.subprocess, "run", run)
    assert [p.name for p in fonts._fontconfig("作 作者\n")] == ["a.ttc", "b.ttf"]
    assert calls[0][-1] == "sans-serif:charset=4f5c 8005"
    assert fonts._fontconfig(" \n") == [] and len(calls) == 1

    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(fonts.subprocess, "run", missing)
    assert fonts._fontconfig("作者") == []


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="Linux installs differ; fontconfig picks there")
@pytest.mark.parametrize("text", ["作者 me", "作者です", "작가", "café", "♥"])
def test_every_desktop_system_ships_a_font_for_each_script(text):
    assert share.text_font(text, None, fonts.offered(text)) is not None
