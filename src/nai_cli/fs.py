"""Every file nai-cli writes: new files never replace one, and replaced files are swapped in whole."""

import contextlib
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import BinaryIO

from nai_cli import values


class _Kind(Enum):
    FILE = "file"
    FOLDER = "folder"
    # A device or the like, which takes a name but is neither
    OTHER = "other"


def numbered(path: Path, n: int) -> Path:
    """The n-th free name for path: path itself, then NAME-1, NAME-2 and so on."""
    if n == 0:
        return path
    stem, suffix = values.split_name(path)
    return path.with_name(f"{stem}-{n}{suffix}")


def free_name(path: Path, taken: set[Path] | None = None) -> Path:
    """A dry run's next name; the real write still claims its name with exclusive creation."""
    n = 0
    while os.path.lexists(candidate := numbered(path, n)) or (taken is not None and candidate in taken):
        n += 1
    return candidate


@dataclass
class DayNumbers:
    """Names by the day of a run and a number in it, 20261125-0001.png. Each folder counts on from the highest
    number the day's names there hold, whatever their suffix, and the next name is one higher whether or not this
    one is written, so a dry run names copies as its run does."""

    when: datetime
    highest: dict[Path, int] = field(default_factory=dict)

    def next_in(self, folder: Path, suffix: str = ".png") -> Path:
        if folder not in self.highest:
            self.highest[folder] = max(_day_numbers(folder, f"{self.when:%Y%m%d}-"), default=0)
        self.highest[folder] += 1
        return folder / f"{self.when:%Y%m%d}-{self.highest[folder]:04d}{suffix}"


def _day_numbers(folder: Path, day: str) -> list[int]:
    try:
        names = os.listdir(folder)
    except OSError:
        # A folder not made yet holds none
        return []
    stems = (name.partition(".")[0].removeprefix(day) for name in names if name.startswith(day))
    return [int(stem) for stem in stems if len(stem) >= 4 and stem.isascii() and stem.isdigit()]


def create(path: Path, data: bytes | str, private: bool = False) -> Path:
    """Write a new file; FileExistsError when the name is taken, decided by the file system, not a check."""
    make_folder(path.parent)
    mode = 0o600 if private else 0o666
    try:
        f = open(path, "xb", opener=lambda name, flags: os.open(name, flags, mode))  # noqa: SIM115
    except PermissionError as e:
        # Windows refuses a folder of that name where POSIX reports the name taken
        found = _stat(path)
        if found is not None and stat.S_ISDIR(found.st_mode):
            raise FileExistsError(f"{path} is a folder") from e
        raise
    try:
        with f:
            f.write(_bytes(data))
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return path


def create_numbered(path: Path, data: bytes | str) -> Path:
    """Write a new file under path or the first free numbered name after it."""
    n = 0
    while True:
        try:
            return create(numbered(path, n), data)
        except FileExistsError:
            n += 1


def replace(path: Path, data: bytes | str, private: bool = False) -> Path:
    """Swap in a whole new file at path; readers never see half of it."""
    make_folder(path.parent)
    partial = create(path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"), data, private)
    try:
        os.replace(partial, path)
    finally:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
    return path


def append(path: Path, text: str) -> None:
    """Add text at the end of path, made if it is not there."""
    make_folder(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def make_folder(path: Path) -> Path:
    """NotADirectoryError when a file holds the place, so that it never reads as a name taken."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as e:
        # Windows reports a file in the way as a name taken, POSIX as a path that is not a folder
        blocking = next((p for p in (path, *path.parents) if is_file(p)), path)
        raise NotADirectoryError(f"{blocking} is a file, not a folder") from e
    return path


def try_names(folder: Path, name: Path | None = None) -> None:
    """Fail as saving into folder would, and leave nothing behind: NotADirectoryError when a file holds a folder's
    place, and the file system's own error for a folder or a name it refuses; only the folders this call made are
    removed again, so another process's stay. name is tried in a scratch folder, so a file under it stays as it is."""
    blocking = next((p for p in (folder, *folder.parents) if is_file(p)), None)
    if blocking is not None:
        raise NotADirectoryError(f"{blocking} is a file, not a folder")
    made: list[Path] = []
    try:
        for step in reversed((folder, *folder.parents)):
            if _stat(step) is None:
                with contextlib.suppress(FileExistsError):
                    step.mkdir()
                    made.append(step)
        probe = folder / f".nai-{uuid.uuid4().hex[:12]}.tmp"
        if name is not None:
            probe.mkdir()
            made.append(probe)
            probe = probe / name.name
        create(probe, b"")
        with contextlib.suppress(OSError):
            probe.unlink()
    finally:
        # Deepest first, so each is empty once those under it are gone; one another process filled stays
        for step in reversed(made):
            with contextlib.suppress(OSError):
                step.rmdir()


def exists(path: Path) -> bool:
    return _kind(path) is not None


def is_file(path: Path) -> bool:
    return _kind(path) is _Kind.FILE


def is_folder(path: Path) -> bool:
    return _kind(path) is _Kind.FOLDER


def _kind(path: Path) -> _Kind | None:
    found = _stat(path)
    if found is None:
        return None
    if stat.S_ISDIR(found.st_mode):
        return _Kind.FOLDER
    return _Kind.FILE if stat.S_ISREG(found.st_mode) else _Kind.OTHER


def _stat(path: Path) -> os.stat_result | None:
    # A name too long to look up on POSIX, or holding a NUL, names nothing rather than failing the command
    try:
        return path.stat()
    except (OSError, ValueError):
        return None


def _bytes(data: bytes | str) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else data


def remove(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def acquire(path: Path, timeout: float, poll: float) -> BinaryIO:
    """An exclusive lock on path shared by every process, held until release; TimeoutError when another process
    holds it past timeout, any other OSError when the file cannot be opened."""
    make_folder(path.parent)
    f = path.open("a+b")
    deadline = time.monotonic() + timeout
    while True:
        try:
            _lock(f)
            return f
        except OSError:
            if time.monotonic() > deadline:
                f.close()
                raise TimeoutError(f"{path} is still locked after {timeout:g} s") from None
            time.sleep(poll)


def release(f: BinaryIO) -> None:
    with contextlib.suppress(OSError):
        _unlock(f)
    f.close()


if sys.platform == "win32":
    import msvcrt

    def _lock(f: BinaryIO) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f: BinaryIO) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(f: BinaryIO) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f: BinaryIO) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
