"""Fonts installed on this computer, offered for a watermark's text when the built-in font cannot draw it."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

# The fonts each system ships, by the script they cover; the first that draws the whole text is used
_SHIPPED = {
    "win32": {
        "latin": ("segoeui.ttf", "arial.ttf"),
        "zh": ("msyh.ttc", "msjh.ttc", "simsun.ttc"),
        "ja": ("YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"),
        "ko": ("malgun.ttf",),
        "symbols": ("seguisym.ttf",),
    },
    "darwin": {
        "latin": ("Helvetica.ttc",),
        "zh": ("Hiragino Sans GB.ttc", "STHeiti Medium.ttc", "Songti.ttc"),
        "ja": ("ヒラギノ角ゴシック W3.ttc",),
        "ko": ("AppleSDGothicNeo.ttc",),
        "symbols": ("Apple Symbols.ttf", "Arial Unicode.ttf"),
    },
    "linux": {
        "latin": ("DejaVuSans.ttf",),
        "zh": ("NotoSansCJK-Regular.ttc", "wqy-microhei.ttc", "wqy-zenhei.ttc"),
        "ja": (),
        "ko": (),
        "symbols": ("DejaVuSans.ttf",),
    },
}
# Fontconfig's own order is tried after these, and only so far
_FONTCONFIG_TRIES = 8
_FONTCONFIG_TIMEOUT = 10


def offered(text: str) -> Iterator[Path]:
    """Installed font files that may draw text, best first: those the system ships for the scripts in it, then on
    Linux and the like those fontconfig picks for its characters. Found as they are asked for, so the folders are
    walked and fontconfig run only when the built-in font has already failed."""
    shipped = _SHIPPED.get(sys.platform, _SHIPPED["linux"])
    names = [name for script in _order(text) for name in shipped[script]]
    seen: set[Path] = set()
    found = _installed()
    for name in names:
        path = found.get(name.lower())
        if path is not None and path not in seen:
            seen.add(path)
            yield path
    if sys.platform not in ("win32", "darwin"):
        for path in _fontconfig(text):
            if path not in seen:
                seen.add(path)
                yield path


def _order(text: str) -> list[str]:
    # Han characters are drawn in a different form in each language, so kana or hangul pick their own fonts first
    if any("\u3040" <= ch <= "\u30ff" or "\u31f0" <= ch <= "\u31ff" for ch in text):
        cjk = ["ja", "zh", "ko"]
    elif any("\uac00" <= ch <= "\ud7af" or "\u1100" <= ch <= "\u11ff" or "\u3130" <= ch <= "\u318f" for ch in text):
        cjk = ["ko", "zh", "ja"]
    else:
        cjk = ["zh", "ja", "ko"]
    return ["latin", *cjk, "symbols"]


def _installed() -> dict[str, Path]:
    """Font files in the system's and the user's font folders by lowercased name; the first of a name wins."""
    out: dict[str, Path] = {}
    for root in _folders():
        for folder, _, files in os.walk(root):
            for name in files:
                out.setdefault(name.lower(), Path(folder) / name)
    return out


def _folders() -> list[Path]:
    home = Path.home()
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
        local = os.environ.get("LOCALAPPDATA")
        return [Path(windir) / "Fonts", *([Path(local) / "Microsoft" / "Windows" / "Fonts"] if local else [])]
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library" / "Fonts"]
    # The XDG base directories, as fontconfig reads them, and the older ~/.fonts
    data_home = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(d) / "fonts" for d in [data_home, *data_dirs.split(":")] if d] + [home / ".fonts"]


def _fontconfig(text: str) -> list[Path]:
    """The sans-serif fonts fontconfig ranks best for the text's characters; none without fontconfig."""
    codes = " ".join(f"{ord(ch):x}" for ch in dict.fromkeys(text) if not ch.isspace())
    if not codes:
        return []
    try:
        listed = subprocess.run(
            ["fc-match", "-s", "--format", "%{file}\n", f"sans-serif:charset={codes}"],
            capture_output=True,
            text=True,
            timeout=_FONTCONFIG_TIMEOUT,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [Path(line) for line in dict.fromkeys(listed.splitlines()) if line][:_FONTCONFIG_TRIES]
