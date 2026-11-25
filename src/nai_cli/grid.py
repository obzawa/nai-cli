"""Grids of jobs that vary settings along two axes, laid out as the official client's grid generation does."""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nai_cli import errors, import_image, models, presets, request, values
from nai_cli.batch import Job

MAX_CELLS = 40
COMMANDS = ("gen", "i2i", "inpaint")
_LABEL_LENGTH = 40
_RANGE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_RANDOM_SEEDS = re.compile(r"^r(\d*)$", re.IGNORECASE)


class Kind(Enum):
    INTEGER = "integer"
    NUMBER = "number"
    SEED = "seed"
    # One value per line
    TEXT = "text"
    # Names separated by commas
    CHOICE = "choice"


@dataclass(frozen=True)
class Param:
    name: str
    option: str
    kind: Kind


PARAMS = {
    p.name: p
    for p in (
        Param("steps", "--steps", Kind.INTEGER),
        Param("cfg", "--cfg", Kind.NUMBER),
        Param("cfg_rescale", "--cfg-rescale", Kind.NUMBER),
        Param("seed", "--seed", Kind.SEED),
        Param("size", "--size", Kind.CHOICE),
        Param("sampler", "--sampler", Kind.CHOICE),
        Param("noise_schedule", "--noise-schedule", Kind.CHOICE),
        Param("model", "--model", Kind.CHOICE),
        Param("prompt", "--prompt", Kind.TEXT),
        Param("neg", "--neg", Kind.TEXT),
        Param("ref_strength", "--ref-strength", Kind.NUMBER),
        Param("ref_fidelity", "--ref-fidelity", Kind.NUMBER),
    )
}
# The official names
ALIASES = {
    "scale": "cfg",
    "uc": "neg",
    "cfg-rescale": "cfg_rescale",
    "noise-schedule": "noise_schedule",
    "pr_strength": "ref_strength",
    "pr_fidelity": "ref_fidelity",
    "ref-strength": "ref_strength",
    "ref-fidelity": "ref_fidelity",
}


@dataclass(frozen=True)
class Axis:
    params: tuple[Param, ...]
    # One row of values per step along the axis, a value for each param
    steps: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...]

    @property
    def title(self) -> str:
        return ", ".join(p.name for p in self.params)


@dataclass(frozen=True)
class Cell:
    job: Job
    column: int
    row: int
    x: str
    y: str


@dataclass(frozen=True)
class Grid:
    x: Axis | None
    y: Axis | None
    cells: tuple[Cell, ...]

    @property
    def columns(self) -> int:
        return len(self.x.steps) if self.x else 1

    @property
    def rows(self) -> int:
        return len(self.y.steps) if self.y else 1


def parse_axis(entries: list[tuple[str, str]]) -> Axis | None:
    """PARAM=VALUES entries of one axis. Params on one axis step together, a shorter list keeping its last value."""
    params: list[Param] = []
    values: list[list[str]] = []
    for name, text in entries:
        key = ALIASES.get(name.strip().lower(), name.strip().lower())
        if key not in PARAMS:
            raise errors.NaiError(errors.BAD_ARGS, f"a grid cannot vary {name}; it varies {', '.join(PARAMS)}")
        param = PARAMS[key]
        if param in params:
            raise errors.NaiError(errors.BAD_ARGS, f"{param.name} is given twice on one axis")
        parsed = _values(param, text)
        if parsed:
            params.append(param)
            values.append(parsed)
    if not params:
        return None
    length = max(map(len, values))
    steps = tuple(tuple(v[min(i, len(v) - 1)] for v in values) for i in range(length))
    return Axis(tuple(params), steps, tuple(_label(", ".join(step)) for step in steps))


def build(argv: list[str], x: Axis | None, y: Axis | None, parse: Callable[[list[str]], Mapping[str, Any]]) -> Grid:
    """A job per cell, row by row; each runs the base command with the cell's values given last. parse reads a
    command line as the command will, so every cell's options are checked before any cell runs."""
    if not argv or argv[0] not in COMMANDS:
        raise errors.NaiError(errors.BAD_ARGS, f"a grid runs one of {', '.join(COMMANDS)}, given after the axes")
    if x is None and y is None:
        raise errors.NaiError(errors.BAD_ARGS, "a grid needs --x or --y")
    shared = set(x.params if x else ()) & set(y.params if y else ())
    if shared:
        raise errors.NaiError(errors.BAD_ARGS, f"{next(iter(shared)).name} is on both axes")
    grid = Grid(x, y, ())
    if grid.columns * grid.rows > MAX_CELLS:
        raise errors.NaiError(
            errors.BAD_ARGS, f"the grid has {grid.columns * grid.rows} cells; at most {MAX_CELLS} are allowed"
        )
    given = parse(argv)
    _check_command(given)
    # Each cell is one image, as the official grid sends
    shared = ["--n=1"]
    varies_seed = PARAMS["seed"] in (*(x.params if x else ()), *(y.params if y else ()))
    if not varies_seed and given.get("seed") is None and not _imports_seed(given):
        # Cells differ only along the axes
        shared.append(f"--seed={request.random_seed()}")
    cells = []
    for row in range(grid.rows):
        for column in range(grid.columns):
            args = [*argv, *shared, *_overrides(x, column), *_overrides(y, row)]
            parse(args)
            job_id = (f"x{column + 1}" if x else "") + (f"y{row + 1}" if y else "")
            label_x = x.labels[column] if x else ""
            label_y = y.labels[row] if y else ""
            cells.append(Cell(Job(job_id, len(cells) + 1, args), column, row, label_x, label_y))
    return Grid(x, y, tuple(cells))


def _overrides(axis: Axis | None, index: int) -> list[str]:
    if axis is None:
        return []
    return [values.option(p.option, v) for p, v in zip(axis.params, axis.steps[index], strict=True)]


def _imports_seed(given: Mapping[str, Any]) -> bool:
    if given.get("from_image") is None:
        return False
    try:
        return "seed" in import_image.parse_parts(given.get("import_parts"))
    except errors.NaiError:
        # The cell reports the bad --import itself
        return False


def _check_command(given: Mapping[str, Any]) -> None:
    """Options that cannot hold across cells: more images than one per cell, a single output file, and prompts
    drawn anew for every cell."""
    if given.get("n") not in (None, 1):
        raise errors.NaiError(errors.BAD_ARGS, "each grid cell is one image; leave out --n")
    output = given.get("output")
    if output is not None and str(output).lower().endswith(".png"):
        raise errors.NaiError(errors.BAD_ARGS, "a grid saves an image per cell; give -o a folder")
    if given.get("random_prompt") is not None:
        raise errors.NaiError(
            errors.BAD_ARGS,
            "a random prompt would differ in every cell; draw one with nai gen --random-prompt --dry-run "
            "and pass it with --prompt",
        )


def _values(param: Param, text: str) -> list[str]:
    if param.kind is Kind.TEXT:
        return [line.strip() for line in text.split("\n") if line.strip()]
    items = [item.strip() for item in text.split(",") if item.strip()]
    if param.kind is Kind.CHOICE:
        for item in items:
            _check_choice(param, item)
        return items
    values: list[str] = []
    for item in items:
        found = _RANDOM_SEEDS.match(item) if param.kind is Kind.SEED else None
        if found:
            count = int(found[1] or 1)
            if count < 1 or len(values) + count > MAX_CELLS:
                raise errors.NaiError(errors.BAD_ARGS, f"{param.name}: {item} asks for 1 to {MAX_CELLS} random seeds")
            values += [str(request.random_seed()) for _ in range(count)]
        else:
            values += _numbers(param, item, MAX_CELLS - len(values))
    return values


def _numbers(param: Param, item: str, room: int) -> list[str]:
    """A number, or a range START-END:STEP with a step of 1 by default."""
    whole = param.kind is not Kind.NUMBER
    span = _RANGE.match(item)
    if span is None:
        if not _NUMBER.match(item):
            raise errors.NaiError(errors.BAD_ARGS, f"{param.name}: {item} is not a number or a range")
        return [_text(int(float(item)) if whole else float(item))]
    start, end = (int(float(v)) if whole else float(v) for v in (span[1], span[2]))
    step = float(span[3]) if span[3] else 1.0
    if step <= 0 or end < start:
        raise errors.NaiError(
            errors.BAD_ARGS, f"{param.name}: {item} is not a range from low to high by a step above 0"
        )
    count = math.floor((end - start) / step + 1e-9) + 1
    if count > room:
        raise errors.NaiError(errors.BAD_ARGS, f"{param.name}: {item} gives more than {MAX_CELLS} values")
    # Rounded half up, as the official client rounds
    return [
        _text(math.floor(start + i * step + 0.5) if whole else float(f"{start + i * step:.12g}")) for i in range(count)
    ]


def _text(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def _check_choice(param: Param, item: str) -> None:
    if param.name == "model":
        models.resolve(item)
    elif param.name == "sampler":
        models.resolve_sampler(item)
    elif param.name == "noise_schedule" and item not in models.NOISE_SCHEDULES:
        raise errors.NaiError(
            errors.BAD_ARGS, f"noise_schedule: {item} is not one of {', '.join(models.NOISE_SCHEDULES)}"
        )
    elif param.name == "size":
        presets.resolve_size(item)


def _label(text: str) -> str:
    return text if len(text) <= _LABEL_LENGTH else text[: _LABEL_LENGTH - 1] + "…"
