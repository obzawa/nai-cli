"""Tags read from an image by an ONNX tagger run on this computer: WD14, CL, CL v2, EVA02 or PixAI."""

import csv
import io
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nai_cli import errors, values

DEFAULT_THRESHOLD = 0.35
DEFAULT_SIZE = 448


class Kind(Enum):
    WD14 = "wd14"
    CL = "cl"
    CL_V2 = "cl-v2"
    EVA02 = "eva02"
    PIXAI = "pixai"


class Category(Enum):
    RATING = "rating"
    GENERAL = "general"
    CHARACTER = "character"
    COPYRIGHT = "copyright"
    ARTIST = "artist"
    META = "meta"
    OTHER = "other"


DEFAULT_CATEGORIES = (Category.GENERAL, Category.CHARACTER, Category.COPYRIGHT)
CHOOSABLE = (Category.GENERAL, Category.CHARACTER, Category.COPYRIGHT, Category.ARTIST, Category.META)
# The order of the prompt: by category, then by score
_PROMPT_ORDER = (Category.ARTIST, Category.CHARACTER, Category.COPYRIGHT, Category.GENERAL, Category.META)
_CATEGORY_NAMES = {
    Category.RATING: ("9", "rating"),
    Category.GENERAL: ("0", "general", "tag", "tags"),
    Category.CHARACTER: ("4", "character", "characters"),
    Category.COPYRIGHT: ("3", "copyright", "copyrights"),
    # PixAI files its artists under style
    Category.ARTIST: ("1", "artist", "artists", "style"),
    Category.META: ("5", "meta"),
}
_LABEL_FILES = (
    "selected_tags.csv",
    "tags.csv",
    "labels.csv",
    "labels.txt",
    "classes.txt",
    "tag_mapping.json",
    "model_vocabulary.json",
    "tags.json",
)
_SIZE_IN_NAME = re.compile(r"(?:^|[^0-9])(224|256|384|448|512)(?:[^0-9]|$)")
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.2613026, 0.2757771], np.float32)


def category_of(raw: str | None, name: str = "") -> Category:
    text = (raw or "").strip().lower()
    if name.startswith("rating:"):
        return Category.RATING
    return next((c for c, names in _CATEGORY_NAMES.items() if text in names), Category.OTHER)


@dataclass(frozen=True)
class Label:
    name: str
    category: Category


@dataclass(frozen=True)
class Labels:
    labels: list[Label]
    # Thresholds a label file recommends per category
    thresholds: dict[Category, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Tag:
    name: str
    score: float
    category: Category

    @property
    def text(self) -> str:
        if self.category is Category.ARTIST and not self.name.lower().startswith("artist:"):
            return f"artist:{self.name}"
        return self.name


@dataclass(frozen=True)
class Settings:
    model: Path
    labels: Path
    kind: Kind
    threshold: float = DEFAULT_THRESHOLD
    character_threshold: float = DEFAULT_THRESHOLD
    categories: tuple[Category, ...] = DEFAULT_CATEGORIES


@dataclass(frozen=True)
class _Profile:
    prepare: Callable[[Image.Image, int], np.ndarray]
    size: int
    # Scores are logits, not probabilities
    sigmoid: bool
    channels_last: bool = False
    thresholds: dict[Category, float] = field(default_factory=dict)


def label_candidates(model: Path) -> list[Path]:
    """Where the tag list beside the model may be, first match first: one of the same name, then the usual
    file names."""
    candidates = [model.with_suffix(ext) for ext in (".csv", ".txt", ".json")]
    if "cl_tagger" in str(model).lower():
        candidates.insert(0, model.parent / "tag_mapping.json")
    return candidates + [model.parent / name for name in _LABEL_FILES]


def is_split_config(path: Path) -> bool:
    """Whether a config.json is a tagger's, which splits its tags, and so a tag list; most are not."""
    return _mentions(path, '"tags_split"')


def guess_kind(model: Path, labels: Path) -> Kind:
    path, label = str(model).lower(), labels.name.lower()
    near = f"{model.parent.name}/{model.name}".lower()
    if "cl_tagger_v2" in path or "cl-tagger-v2" in path or label == "model_vocabulary.json":
        return Kind.CL_V2
    if "eva02_large_patch14" in path:
        return Kind.EVA02
    if "pixai" in near or _is_pixai_json(labels):
        return Kind.PIXAI
    stem = model.stem.lower()
    if any(word in stem for word in ("wd14", "wd-v1-4", "wd-v1-5", "convnext", "vit", "swinv2")):
        return Kind.WD14
    if ("cl" in stem and "tagger" in stem) or label == "tag_mapping.json":
        return Kind.CL
    return Kind.WD14


def _is_pixai_json(labels: Path) -> bool:
    if labels.suffix.lower() != ".json":
        return False
    return _mentions(labels, '"tags_split"') or _mentions(labels, '"category_order"', '"offset"')


def _mentions(path: Path, *words: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # load_labels reports a file that cannot be read
        return False
    return all(word in text for word in words)


def interrogate(image: Image.Image, s: Settings, labels: Labels, model: Any) -> list[Tag]:
    """Tags of image from the tag list and model files s names, as load_labels and local_models.load read them."""
    profile = _PROFILES[s.kind]
    inputs = model.inputs()
    if not inputs:
        raise errors.NaiError(errors.BAD_ARGS, f"{s.model.name} takes no image, so it is not a tagger")
    name = "pixel_values" if s.kind is Kind.CL_V2 and any(i.name == "pixel_values" for i in inputs) else inputs[0].name
    shape = next(i.shape for i in inputs if i.name == name)
    dims = [d if values.is_whole(d) and d > 0 else -1 for d in shape]
    if len(dims) == 4 and 3 in (dims[1], dims[3]) and (dims[3] == 3) != profile.channels_last:
        raise errors.NaiError(
            errors.BAD_ARGS, f"{s.model.name} does not take {s.kind.value} input; check --kind and the tag list"
        )
    side = _square_side(dims, profile.channels_last) or _size_from_name(s.model, profile.size, s.kind)
    tensor = profile.prepare(image, side)
    batch = dims[0] if dims and dims[0] > 0 else 1
    outputs = [o.name for o in model.outputs()]
    wanted = ["logits"] if s.kind is Kind.CL_V2 and "logits" in outputs else None
    feeds = {name: np.repeat(tensor, batch, axis=0)}
    answer = model.run(wanted, feeds)
    try:
        scores = np.asarray(answer[0], np.float64)[0].reshape(-1)
    except (ValueError, TypeError, IndexError) as e:
        raise errors.NaiError(errors.BAD_ARGS, f"{s.model.name} does not give tag scores: {e}") from e
    if scores.size != len(labels.labels):
        raise errors.NaiError(
            errors.BAD_ARGS,
            f"{s.model.name} gives {scores.size} scores but {s.labels.name} names {len(labels.labels)} tags; "
            "they are not from the same model",
        )
    if profile.sigmoid or scores.min(initial=0) < 0 or scores.max(initial=0) > 1:
        scores = 1 / (1 + np.exp(-scores))
    recommended = {**profile.thresholds, **labels.thresholds}
    return select(labels.labels, scores, s, recommended)


def select(labels: list[Label], scores: np.ndarray, s: Settings, recommended: dict[Category, float]) -> list[Tag]:
    """General and character follow the settings; the others use the model's own thresholds, or fall back to
    the character threshold for copyright and artist and the general one otherwise."""

    def threshold(c: Category) -> float:
        if c is Category.GENERAL:
            return s.threshold
        if c is Category.CHARACTER:
            return s.character_threshold
        fallback = s.character_threshold if c in (Category.COPYRIGHT, Category.ARTIST) else s.threshold
        return recommended.get(c, fallback)

    tags = [
        Tag(label.name, float(score), label.category)
        for label, score in zip(labels, scores, strict=True)
        if label.category in s.categories and score >= threshold(label.category)
    ]
    rank = {c: i for i, c in enumerate(_PROMPT_ORDER)}
    return sorted(tags, key=lambda t: (rank.get(t.category, len(rank)), -t.score))


def _square_side(dims: list[int], channels_last: bool) -> int | None:
    if len(dims) != 4:
        return None
    h, w = (dims[1], dims[2]) if channels_last else (dims[2], dims[3])
    return h if h > 0 and h == w else None


def _size_from_name(model: Path, default: int, kind: Kind) -> int:
    if kind is not Kind.WD14:
        return default
    match = _SIZE_IN_NAME.search(model.name.lower())
    return int(match.group(1)) if match else default


def load_labels(path: Path) -> Labels:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"cannot read the tag list {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise errors.NaiError(errors.BAD_ARGS, f"the tag list {path.name} is not UTF-8 text: {e}") from e
    suffix = path.suffix.lower()
    try:
        found = (
            _json_labels(values.parse_json(raw))
            if suffix == ".json"
            else Labels(_csv_labels(raw) if suffix == ".csv" else _text_labels(raw))
        )
    except (ValueError, KeyError, TypeError, csv.Error) as e:
        raise errors.NaiError(errors.BAD_ARGS, f"cannot read the tag list {path.name}: {e}") from e
    if not found.labels:
        raise errors.NaiError(errors.BAD_ARGS, f"the tag list {path.name} has no tags")
    if not any(label.category in CHOOSABLE for label in found.labels):
        # A list without categories would otherwise never give a tag; rating: names alone are no categories
        promoted = [Label(x.name, Category.GENERAL) if x.category is Category.OTHER else x for x in found.labels]
        return Labels(promoted, found.thresholds)
    return found


def _label(name: Any, category: Any = None) -> Label | None:
    text = str(name).strip() if name is not None else ""
    return Label(text, category_of(None if category is None else str(category), text)) if text else None


def _csv_labels(raw: str) -> list[Label]:
    rows = [row for row in csv.reader(io.StringIO(raw)) if row]
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    has_header = any(h in header for h in ("name", "tag", "category"))
    if has_header:
        name_at = max(header.index(h) if h in header else -1 for h in ("name", "tag"))
        name_at = name_at if name_at >= 0 else (1 if len(rows[0]) > 1 else 0)
        category_at = header.index("category") if "category" in header else -1
        rows = rows[1:]
    else:
        name_at, category_at = (1 if len(rows[0]) > 1 else 0), 2
    out = []
    for row in rows:
        if len(row) > name_at:
            label = _label(row[name_at], row[category_at] if 0 <= category_at < len(row) else None)
            if label:
                out.append(label)
    return out


def _text_labels(raw: str) -> list[Label]:
    lines = (line.strip() for line in raw.splitlines())
    labels = (_label(re.split(r"[\t,]", line)[0]) for line in lines if line and not line.startswith("#"))
    return [label for label in labels if label]


def _json_labels(data: Any) -> Labels:
    if isinstance(data, list):
        return Labels(_listed(data))
    if not isinstance(data, dict):
        return Labels([])
    for read in (_vocabulary, _split_tags, _category_offsets):
        found = read(data)
        if found is not None:
            return found
    numbered = sorted(int(k) for k in data if values.is_digits(str(k)))
    if numbered:
        return Labels(_listed([data[str(i)] for i in numbered]))
    listed = data.get("labels", data.get("tags"))
    return Labels(_listed(listed) if isinstance(listed, list) else [])


def _listed(items: list[Any]) -> list[Label]:
    out = []
    for item in items:
        if isinstance(item, dict):
            label = _label(item.get("name", item.get("tag", item.get("label"))), item.get("category"))
        else:
            label = _label(item) if isinstance(item, str) else None
        if label:
            out.append(label)
    return out


def _vocabulary(data: dict[str, Any]) -> Labels | None:
    """idx_to_tag or tag_to_idx, with categories through tag_to_category."""
    by_index: dict[int, str] = {}
    names = data.get("idx_to_tag")
    if isinstance(names, dict):
        by_index = {int(k): str(v).strip() for k, v in names.items() if values.is_digits(str(k)) and str(v).strip()}
    elif isinstance(names, list):
        by_index = {i: str(v).strip() for i, v in enumerate(names) if v is not None and str(v).strip()}
    elif isinstance(data.get("tag_to_idx"), dict):
        by_index = {int(v): str(k).strip() for k, v in data["tag_to_idx"].items() if values.is_digits(str(v))}
    if not by_index:
        return None
    of_tag = data.get("tag_to_category") if isinstance(data.get("tag_to_category"), dict) else {}
    names = data.get("categories") if isinstance(data.get("categories"), dict) else {}
    # Most files map a category id to its name; CL tagger v2 lists each category's tags under its name instead
    names_of = {str(k): v for k, v in names.items() if isinstance(v, str | int | float | bool)}
    listed = {str(tag): name for name, tags in names.items() if isinstance(tags, list) for tag in tags}
    labels = []
    for i in sorted(by_index):
        raw = of_tag.get(by_index[i], listed.get(by_index[i]))
        category = None if raw is None else names_of.get(str(raw), raw)
        labels.append(Label(by_index[i], category_of(None if category is None else str(category), by_index[i])))
    return Labels(labels)


def _split_tags(data: dict[str, Any]) -> Labels | None:
    """tags in the order and counts of tags_split, with category_best_threshold."""
    tags, split = data.get("tags"), data.get("tags_split")
    if not isinstance(tags, list) or not isinstance(split, list):
        return None
    labels: list[Label] = []
    for category, count in split:
        if not values.is_whole(count) or count < 0:
            raise ValueError(f"tags_split has a count of {count!r}")
        end = len(labels) + count
        if end > len(tags):
            raise ValueError("tags_split exceeds the tag list")
        labels += [Label(str(t).strip(), category_of(str(category), str(t))) for t in tags[len(labels) : end]]
    if len(labels) != len(tags):
        raise ValueError("tags_split does not cover every tag")
    best = data.get("category_best_threshold")
    thresholds = {}
    for name, value in best.items() if isinstance(best, dict) else []:
        category = category_of(str(name))
        if values.is_number(value) and category is not Category.OTHER:
            thresholds[category] = float(value)
    return Labels(labels, thresholds)


def _category_offsets(data: dict[str, Any]) -> Labels | None:
    """categories, each with an offset, a count and its tags."""
    entries = data.get("categories")
    if not isinstance(entries, list) or not entries:
        return None
    if not all(
        isinstance(e, dict) and values.is_whole(e.get("offset")) and values.is_whole(e.get("count")) for e in entries
    ):
        return None
    # Counted from the tags listed, so a count claiming more than the file holds allocates nothing
    if not all(isinstance(e.get("tags"), list) and len(e["tags"]) == e["count"] for e in entries):
        raise ValueError("a category's count does not match its tags")
    named = sum(e["count"] for e in entries)
    total = data.get("num_classes") or named
    # More outputs than tags named would leave some unnamed
    if not values.is_whole(total) or not 0 < total <= named:
        raise ValueError(f"num_classes {total!r} does not match the {named} tags named")
    labels: list[Label | None] = [None] * total
    for e in entries:
        tags = e["tags"]
        if e["offset"] < 0 or e["offset"] + len(tags) > total:
            raise ValueError("a category does not fit the tag map")
        for i, tag in enumerate(tags):
            labels[e["offset"] + i] = Label(str(tag).strip(), category_of(e.get("name"), str(tag)))
    if any(label is None for label in labels):
        raise ValueError("the tag map leaves model outputs unnamed")
    return Labels([label for label in labels if label is not None])


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def _letterbox(image: Image.Image, side: int, resample: Image.Resampling) -> Image.Image:
    """Fitted by its longest side and centred on a white square."""
    rgb = _rgb_on_white(image)
    scale = side / max(1, *rgb.size)
    w, h = (min(side, max(1, round(v * scale))) for v in rgb.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(rgb.resize((w, h), resample), ((side - w) // 2, (side - h) // 2))
    return canvas


def _channels_first(pixels: np.ndarray) -> np.ndarray:
    return pixels.transpose(2, 0, 1)[None].astype(np.float32)


def _wd14(image: Image.Image, side: int) -> np.ndarray:
    # BGR from 0 to 255, channels last
    pixels = np.asarray(_letterbox(image, side, Image.Resampling.BICUBIC), np.float32)[..., ::-1]
    return np.ascontiguousarray(pixels)[None]


def _cl(image: Image.Image, side: int) -> np.ndarray:
    pixels = np.asarray(_letterbox(image, side, Image.Resampling.BICUBIC), np.float32)[..., ::-1] / 255
    return _channels_first(pixels)


def _cl_v2(image: Image.Image, side: int) -> np.ndarray:
    pixels = np.asarray(_letterbox(image, side, Image.Resampling.BICUBIC), np.float32) / 127.5 - 1
    return _channels_first(pixels)


def _eva02(image: Image.Image, side: int) -> np.ndarray:
    # Fitted onto a 512 white square first, then resized, as the model was trained
    padded = _letterbox(image, 512, Image.Resampling.BILINEAR).resize((side, side), Image.Resampling.BICUBIC)
    pixels = (np.asarray(padded, np.float32) / 255 - _CLIP_MEAN) / _CLIP_STD
    return _channels_first(pixels)


def _pixai(image: Image.Image, side: int) -> np.ndarray:
    """Scaled down to fit with sizes floored, centred on black, then from -1 to 1."""
    rgb = _rgb_on_white(image)
    ratio = min(side / rgb.height, side / rgb.width)
    w, h = (min(side, max(1, math.floor(v * ratio))) for v in rgb.size)
    pixels = np.full((side, side, 3), -1.0, np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    resized = np.asarray(rgb.resize((w, h), Image.Resampling.BILINEAR, reducing_gap=None), np.float32) / 255
    pixels[top : top + h, left : left + w] = resized * 2 - 1
    return _channels_first(pixels)


_PROFILES = {
    Kind.WD14: _Profile(_wd14, DEFAULT_SIZE, False, channels_last=True),
    Kind.CL: _Profile(_cl, 448, False),
    Kind.CL_V2: _Profile(_cl_v2, 384, True),
    Kind.EVA02: _Profile(_eva02, 448, True),
    Kind.PIXAI: _Profile(
        _pixai, 1008, False, thresholds={Category.COPYRIGHT: 0.24, Category.ARTIST: 0.15, Category.META: 0.17}
    ),
}
