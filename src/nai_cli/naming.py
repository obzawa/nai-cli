"""File names and folders for everything nai-cli saves, by the official template syntax and their sources."""

import codecs
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nai_cli import models, values
from nai_cli import prompt as P

_TEXT_FILTERS = {
    "first": "positive",
    "last": "positive",
    "words": "nonzero",
    "tags": "nonzero",
    "pre": "text",
    "suf": "text",
    "pad": "positive",
    "lower": "none",
    "upper": "none",
}
_CHARACTER_FILTERS = {"max": "positive", "at": "positive", "each": "positive", "join": "text", "full": "none"}
_TOKENS = frozenset(
    {
        "prompt",
        "seed",
        "date",
        "time",
        "year",
        "month",
        "day",
        "model",
        "modelname",
        "resolution",
        "width",
        "height",
        "index",
        "id",
        "type",
        "characters",
        "steps",
        "cfg",
        "sampler",
    }
)
# A character is named by its first tag, or its second when the first only says who it is
_GENERIC_TAGS = frozenset({"girl", "female", "boy", "male", "1girl", "1boy"})
_LENGTH_FILTERS = frozenset({"first", "last", "words", "tags"})
_INTEGER = re.compile(r"-?[0-9]+")
_DIGITS = re.compile(r"[0-9]+")
_SPACES = re.compile("[" + re.escape(P.JS_SPACE) + "]+")
_UNSAFE = re.compile(r'[\n\r"%*/:<>?\\|]')
# Folder rules split only on separators written in the template; a value's become _ like any unsafe character
_SEPARATOR = re.compile(r"[/\\]")
_PLACEHOLDER = "\0"
# Windows refuses these in file names; a browser download never meets them
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PROMPT_LENGTH = 80
_NAME_LENGTH = 180
# Linux and macOS take 255 UTF-8 bytes a name; room is kept for a number and the longest extension
_NAME_BYTES = 255 - len("-9999.naiv4vibebundle.png")
_DEFAULT_CHARACTERS = 3


@dataclass(frozen=True)
class Character:
    prompt: str
    name: str | None = None


@dataclass(frozen=True)
class Meta:
    """What a template can name an image by; anything unknown renders empty."""

    prompt: str = ""
    seed: int | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None
    # Counted from 0 and shown from 1
    index: int | None = None
    id: str = ""
    type: str = ""
    date: datetime | None = None
    characters: tuple[Character, ...] = field(default_factory=tuple)
    steps: int | None = None
    cfg: float | None = None
    sampler: str = ""


@dataclass(frozen=True)
class Destination:
    """Where a run saves images: under the config's folder by its folder rule, in a folder given for the run, or
    at a file name given for the run, which later images of a request extend with their index."""

    folder: Path
    folders: "Template | None" = None
    exact: Path | None = None

    def image_path(
        self,
        meta: Meta,
        seed: int = 0,
        index: int = 0,
        suffix: str = "",
        name: str | None = None,
    ) -> Path:
        """The name to try first; fs numbers it when taken. seed is the request's, which the default name
        HHMMSS_seed_index keeps; a name from the template replaces it, and a file name given replaces both. Every
        name is cut to fit, its suffix kept."""
        if self.exact is not None:
            return self.exact.with_name(fit(self.exact.stem, (f"_{index}" if index else "") + suffix) + ".png")
        stem = name if name is not None else default_stem(meta.date, seed, index)
        return self.folder_for(meta) / f"{fit(stem, suffix)}.png"

    def fallback_path(self, meta: Meta, seed: int = 0, index: int = 0, suffix: str = "") -> Path:
        """The default name, with no text from the prompt, the source or -o, in the folder checked before paying,
        for when the chosen name cannot be saved."""
        return self.checked_folder / f"{fit(default_stem(meta.date, seed, index), suffix)}.png"

    @property
    def checked_folder(self) -> Path:
        """The folder made and tried before anything is paid for; the folders of a folder rule come later."""
        return self.exact.parent if self.exact is not None else self.folder

    def folder_for(self, meta: Meta) -> Path:
        if self.exact is not None:
            return self.exact.parent
        if self.folders is None:
            return self.folder
        return self.folder.joinpath(*render_folders(self.folders, meta))


def default_stem(when: datetime | None, seed: int, index: int = 0) -> str:
    time = when.strftime("%H%M%S") if when else "000000"
    return f"{time}_{seed}" + (f"_{index}" if index else "")


def fit(stem: str, kept: str = "") -> str:
    """stem cut on a character boundary so that stem and kept make a name every platform takes; kept, which tells
    the file apart, is never cut."""
    # POSIX file names can carry surrogate escapes, which must survive the cut
    room = max(_NAME_BYTES - len(kept.encode("utf-8", "surrogatepass")), 0)
    data = stem.encode("utf-8", "surrogatepass")
    if len(data) <= room:
        return stem + kept
    prefix = codecs.getincrementaldecoder("utf-8")("surrogatepass").decode(data[:room])
    return P.js_trim(prefix) + kept


def fitted(path: Path) -> Path:
    """path with its name cut to fit, its suffix kept."""
    stem, suffix = values.split_name(path)
    return path.with_name(fit(stem, suffix))


def copy_path(source: Path, label: str) -> Path:
    """An image made from another, beside it as NAME_label.png."""
    return source.with_name(f"{fit(source.stem, '_' + label)}.png")


def vibe_path(first_source: Path, bundled: bool, into: Path | None, when: datetime) -> Path:
    """The official export's names: the source's for one Vibe, the date for a bundle, which the official client
    takes in UTC, and the holding image's for a bundle written into a PNG."""
    if into is not None:
        return into.with_name(f"{fit(into.name.split('.')[0])}.naiv4vibebundle.png")
    if bundled:
        return first_source.parent / f"{when.astimezone(UTC).date().isoformat()}.naiv4vibebundle"
    return first_source.with_name(f"{fit(first_source.stem)}.naiv4vibe")


@dataclass(frozen=True)
class _Token:
    name: str
    filters: tuple[tuple[str, str | None], ...]


Template = list[str | _Token]


def parse(template: str) -> Template:
    """Raises ValueError with the official client's message for a template it would refuse."""
    parts: Template = []
    text, i = "", 0
    while i < len(template):
        ch = template[i]
        if ch != "{":
            text += ch
            i += 1
            continue
        if template[i + 1 : i + 2] == "{":
            text += "{"
            i += 2
            continue
        end = template.find("}", i)
        if end < 0:
            raise ValueError(f'Unclosed "{{" at position {P.js_length(template[:i]) + 1}')
        if text:
            parts.append(text)
            text = ""
        parts.append(_token(template[i + 1 : end]))
        i = end + 1
    if text:
        parts.append(text)
    return parts


def parse_folders(template: str) -> Template:
    """A folder rule; refused when it could leave the output folder."""
    parts = parse(template)
    written = "".join(part if isinstance(part, str) else _PLACEHOLDER for part in parts)
    if _SEPARATOR.match(written) or re.match(r"[A-Za-z]:", written):
        raise ValueError("Folder rules are relative to the output folder")
    if any(P.js_trim(piece) in (".", "..") for piece in _SEPARATOR.split(written)):
        raise ValueError('Folder rules cannot use "." or ".."')
    return parts


def _token(body: str) -> _Token:
    pieces = body.split("|")
    name = P.js_trim(pieces[0])
    if name not in _TOKENS:
        raise ValueError(f'Unknown token "{{{name}}}"')
    characters = name == "characters"
    text_filter_seen = not characters
    filters = []
    for piece in pieces[1:]:
        colon = piece.find(":")
        filter_name = P.js_trim(piece if colon < 0 else piece[:colon])
        arg = None if colon < 0 else piece[colon + 1 :]
        kind = _TEXT_FILTERS.get(filter_name)
        character_kind = _CHARACTER_FILTERS.get(filter_name)
        if kind is None and character_kind is None:
            raise ValueError(f'Unknown filter "{filter_name}"')
        if kind is None:
            if not characters:
                raise ValueError(f'Filter "{filter_name}" only applies to {{characters}}')
            if text_filter_seen:
                raise ValueError(f'Filter "{filter_name}" must come before text filters like "first" or "pre"')
        else:
            text_filter_seen = True
        _check_arg(filter_name, kind or character_kind or "", arg)
        filters.append((filter_name, arg))
    return _Token(name, tuple(filters))


def _check_arg(name: str, kind: str, arg: str | None) -> None:
    number = _number(arg)
    if kind == "none" and arg is not None:
        raise ValueError(f'Filter "{name}" takes no value')
    if kind == "text" and arg is None:
        raise ValueError(f'Filter "{name}" needs a value, e.g. "{name}:-"')
    if kind == "positive" and (number is None or number <= 0):
        raise ValueError(f'Filter "{name}" needs a positive number')
    if kind == "nonzero" and not number:
        raise ValueError(f'Filter "{name}" needs a non-zero number')


def _number(arg: str | None) -> int | None:
    if arg is None or not _INTEGER.fullmatch(P.js_trim(arg)):
        return None
    return int(P.js_trim(arg))


def names_type(template: Template) -> bool:
    """Whether the template tells results by what made them, which the official client asks of a template before
    naming a tool's or an upscale's results by it."""
    return any(isinstance(part, _Token) and part.name == "type" for part in template)


def render(template: Template, meta: Meta) -> str:
    """The file name without extension; empty means the template gave nothing, and the caller names the file
    another way."""
    return _clean("".join(part if isinstance(part, str) else _render_token(part, meta) for part in template))


def render_folders(template: Template, meta: Meta) -> list[str]:
    """The folders a rule gives, outermost first; empty ones and those of dots only are left out."""
    segments = [""]
    for part in template:
        if isinstance(part, str):
            first, *rest = _SEPARATOR.split(part)
            segments[-1] += first
            segments.extend(rest)
        else:
            segments[-1] += _UNSAFE.sub("_", _render_token(part, meta))
    return [folder for folder in map(_clean, segments) if folder.strip(".")]


def _clean(name: str) -> str:
    out = P.js_trim(_CONTROL.sub("_", _UNSAFE.sub("_", name)))
    return fit(P.js_trim(out[:_NAME_LENGTH]) if len(out) > _NAME_LENGTH else out)


# The official download name: the prompt's first 80 characters, " s-" and the seed, " (N)" for the N-th image of a
# request; browsers add " (N)" again for a name already taken
_DOWNLOAD_NAME = re.compile(r" s-[0-9]+(?: \([0-9]+\))*$")


def is_download_name(stem: str) -> bool:
    return _DOWNLOAD_NAME.search(stem) is not None


def fallback(meta: Meta) -> str:
    """The official client's name when a template renders empty: the prompt's start and the seed."""
    seed = "" if meta.seed is None else str(meta.seed)
    name = f"{meta.prompt[:_PROMPT_LENGTH]} s-{seed}"
    return _CONTROL.sub("_", re.sub(r'["%*/:<>?\\|]', "_", name))


def _render_token(token: _Token, meta: Meta) -> str:
    if token.name == "characters":
        return _characters(token.filters, meta.characters)
    value = _value(token.name, meta)
    if token.name == "prompt" and not any(name in _LENGTH_FILTERS for name, _ in token.filters):
        value = value[:_PROMPT_LENGTH]
    for name, arg in token.filters:
        value = _apply(value, name, arg)
    return value


def _value(name: str, meta: Meta) -> str:
    match name:
        case "prompt":
            return meta.prompt
        case "seed":
            return "" if meta.seed is None else str(meta.seed)
        case "date":
            return meta.date.strftime("%Y-%m-%d") if meta.date else ""
        case "time":
            return meta.date.strftime("%H-%M-%S") if meta.date else ""
        case "year":
            return meta.date.strftime("%Y") if meta.date else ""
        case "month":
            return meta.date.strftime("%m") if meta.date else ""
        case "day":
            return meta.date.strftime("%d") if meta.date else ""
        case "model":
            return meta.model or ""
        case "modelname":
            if not meta.model:
                return ""
            model = models.MODELS.get(meta.model)
            return model.short_name if model else meta.model
        case "resolution":
            return f"{meta.width}x{meta.height}" if meta.width and meta.height else ""
        case "width":
            return str(meta.width) if meta.width else ""
        case "height":
            return str(meta.height) if meta.height else ""
        case "index":
            return "" if meta.index is None else str(meta.index + 1)
        case "id":
            return meta.id
        case "type":
            return meta.type
        case "steps":
            return "" if meta.steps is None else str(meta.steps)
        case "cfg":
            return "" if meta.cfg is None else f"{meta.cfg:g}"
        case "sampler":
            return meta.sampler
    return ""


def _apply(text: str, name: str, arg: str | None) -> str:
    number = _number(arg) or 0
    match name:
        case "first":
            return text[:number]
        case "last":
            return text[-number:]
        case "words":
            words = [w for w in _SPACES.split(text) if w]
            return " ".join(words[:number] if number > 0 else words[number:])
        case "tags":
            tags = [t for t in (P.js_trim(t) for t in text.split(",")) if t]
            return ", ".join(tags[:number] if number > 0 else tags[number:])
        case "pre":
            return "" if text == "" else (arg or "") + text
        case "suf":
            return "" if text == "" else text + (arg or "")
        case "pad":
            return text.rjust(number, "0") if _DIGITS.fullmatch(text) else text
        case "lower":
            return text.lower()
        case "upper":
            return text.upper()
    return text


def _characters(filters: tuple[tuple[str, str | None], ...], characters: tuple[Character, ...]) -> str:
    limit = at = each = None
    join, full = "+", False

    def listing() -> str:
        chosen = (
            characters[: _DEFAULT_CHARACTERS if limit is None else limit] if at is None else characters[at - 1 : at]
        )
        names = [
            P.js_trim(c.prompt) if full else c.name if c.name is not None else _short_name(c.prompt) for c in chosen
        ]
        if each is not None:
            names = [n[:each] for n in names]
        return join.join(n for n in names if n)

    text = None
    for name, arg in filters:
        if text is None and name not in _TEXT_FILTERS:
            match name:
                case "full":
                    full = True
                case "max":
                    limit = _number(arg)
                case "at":
                    at = _number(arg)
                case "each":
                    each = _number(arg)
                case "join":
                    join = arg or ""
            continue
        if text is None:
            text = listing()
        text = _apply(text, name, arg)
    return listing() if text is None else text


def _short_name(prompt: str) -> str:
    tags = [t for t in (P.js_trim(t) for t in prompt.split(",")) if t]
    if not tags:
        return ""
    if tags[0].lower() in _GENERIC_TAGS:
        return tags[1] if len(tags) > 1 else tags[0]
    return tags[0]
