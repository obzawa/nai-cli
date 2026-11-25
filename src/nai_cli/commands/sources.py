"""The images a command works on: those named on the command line, and those inside the folders named there."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nai_cli import fs


@dataclass(frozen=True)
class Source:
    path: Path
    # Where the image lies inside the folder given for it; its name alone when the image itself was given
    relative: Path


def expand(given: Iterable[Path], suffixes: tuple[str, ...], recursive: bool) -> tuple[Source, ...]:
    """An image stands for itself and a folder for the images in it, known by suffix in any case; recursive also
    takes those in the folders inside it."""
    out: list[Source] = []
    for path in given:
        if not fs.is_folder(path):
            out.append(Source(path, Path(path.name)))
            continue
        pattern = "**/*" if recursive else "*"
        found = sorted(p for p in path.glob(pattern) if p.suffix.lower() in suffixes and fs.is_file(p))
        out.extend(Source(p, p.relative_to(path)) for p in found)
    return tuple(out)
