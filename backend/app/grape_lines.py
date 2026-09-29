"""A tight crop of the grape line for series whose labels share one layout.

These vectors live under their own model name, so they never enter the
bottle search. Existing label crops stay as they are.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from PIL import Image

from .catalog import normalize, tokens
from .label_signals import SIBLING_GAP, _label_markers

GRAPE_SUFFIX = "+grape-line"
MIN_GRAPE_SCORE = 0.40
MIN_GRAPE_LEAD = 0.04


def grape_model(model: str) -> str:
    return f"{model}{GRAPE_SUFFIX}"


def grape_view_digest(file_digest: str) -> str:
    return hashlib.sha256(f"grape:{file_digest}".encode()).hexdigest()


def in_confusable_series(wine) -> bool:
    """Labels that repeat one design and differ by the grape line.

    The Mezyb estate also bottles Shishka and other lines. Those stay out:
    only a name that itself says Мезыбь shares the confused label.
    """

    name = normalize(wine.name)
    slug = normalize(str(wine.slug).replace("-", " "))
    if "шишка" in name or "shishka" in slug:
        return False
    if "красная стрелка" in name or "velvet season" in name or "вельвет сизон" in name:
        return True
    if slug.startswith("loco cimbali") or "loco cimbali" in name or "локо чимбали" in name:
        return True
    if "мезыб" in name or "русское игристое" in name:
        return True
    return False


def grape_tokens(wine) -> set:
    return {word for word in tokens(" ".join(wine.grapes or [])) if len(word) >= 4}


def index_path() -> Path:
    configured = os.getenv("GRAPE_LINE_INDEX", "").strip()
    if configured:
        return Path(configured)
    models = Path("/models/grape_line_index.json")
    if models.exists() or Path("/models").is_dir():
        return models
    return Path(__file__).with_name("grape_line_index.json")


def load_index(path: Optional[Path] = None) -> dict:
    file = path or index_path()
    try:
        payload = json.loads(file.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(slug): str(digest) for slug, digest in payload.items() if slug and digest}


def _points(box) -> list:
    if box is None:
        return []
    if hasattr(box, "tolist"):
        box = box.tolist()
    points = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _top(box) -> float:
    points = _points(box)
    return min(point[1] for point in points) if points else 0.0


def _bottom(box) -> float:
    points = _points(box)
    return max(point[1] for point in points) if points else 0.0


def line_for_grapes(lines: Sequence[Mapping], grape_words: Iterable[str]) -> Optional[dict]:
    """The line that actually names this bottle's grape."""

    wanted = set(grape_words)
    best = None
    best_hits = 0
    for line in lines:
        hits = len(tokens(line.get("text")) & wanted)
        if hits > best_hits:
            best = line
            best_hits = hits
    return best if best_hits else None


def select_query_line(lines: Sequence[Mapping], wines: Sequence) -> Optional[list]:
    """Box of the grape line when two bottles of one series are both plausible."""

    usable = [wine for wine in wines if wine is not None]
    if len(usable) < 2 or not lines:
        return None
    marker_sets = [_label_markers(wine) for wine in usable]
    for line in lines:
        seen = tokens(line.get("text"))
        owners = []
        for index, markers in enumerate(marker_sets):
            others = set().union(*(marker_sets[:index] + marker_sets[index + 1 :]))
            if (markers - others) & seen:
                owners.append(index)
        if len(owners) == 1:
            return line.get("box")
    shared = set.intersection(*marker_sets) if marker_sets else set()
    titled = [line for line in lines if tokens(line.get("text")) & shared]
    if not titled:
        return None
    title = max(titled, key=lambda line: len(tokens(line.get("text")) & shared))
    title_bottom = _bottom(title.get("box"))
    below = [
        line
        for line in lines
        if line is not title and _top(line.get("box")) >= title_bottom - 4
    ]
    if not below:
        return None
    below.sort(key=lambda line: _top(line.get("box")))
    return below[0].get("box")


def crop_quad(image: Image.Image, box, pad: float = 0.2) -> Optional[Image.Image]:
    points = _points(box)
    if len(points) < 2 or image is None:
        return None
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    left = max(0, int(left - width * pad))
    top = max(0, int(top - height * pad))
    right = min(image.width, int(right + width * pad))
    bottom = min(image.height, int(bottom + height * pad))
    if right - left < 12 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom))


def prefer_by_grape_score(
    rows: list,
    grape_scores: Mapping[str, float],
    max_gap: float = SIBLING_GAP,
) -> list:
    """Promote a close sibling whose grape-line photo matches the query line."""

    if len(rows) < 2 or not grape_scores:
        return rows
    leader = rows[0]
    leader_visual = float(leader.get("siglip") or leader.get("score") or 0.0)
    leader_grape = grape_scores.get(leader.get("slug"))
    best = None
    best_score = -1.0
    for other in rows[1:5]:
        visual = float(other.get("siglip") or other.get("score") or 0.0)
        if leader_visual - visual > max_gap:
            continue
        score = grape_scores.get(other.get("slug"))
        if score is None or score < MIN_GRAPE_SCORE:
            continue
        if leader_grape is not None and score < leader_grape + MIN_GRAPE_LEAD:
            continue
        if score > best_score:
            best = other
            best_score = score
    if best is None:
        return rows
    rest = [row for row in rows if row.get("slug") != best.get("slug")]
    return [best] + rest
