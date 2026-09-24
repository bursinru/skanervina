"""Cheap label-color and OCR extras on top of SigLIP cosine scores."""

import os
from typing import Any, Dict, Mapping, Optional, Sequence
from difflib import SequenceMatcher

import numpy as np
from PIL import Image

from .catalog import CatalogWine, normalize, tokens

COLOR_WEIGHT = 0.05
OCR_WEIGHT = 0.12
VISUAL_LOCK = 0.02
OCR_STOP = {
    "вино",
    "вина",
    "сухое",
    "сухое",
    "полусухое",
    "полусладкое",
    "сладкое",
    "красное",
    "белое",
    "розовое",
    "оранжевое",
    "the",
    "and",
    "wine",
    "estate",
    "chateau",
    "reserve",
    "резерв",
    "усадьба",
}


def wine_tone(wine: Optional[CatalogWine]) -> str:
    if wine is None:
        return "unknown"

    def from_blob(blob: str) -> str:
        if "роз" in blob or "rose" in blob:
            return "rose"
        if "оранж" in blob or "orange" in blob:
            return "orange"
        if "красн" in blob or "red" in blob:
            return "red"
        if "бел" in blob or "white" in blob:
            return "white"
        return "unknown"

    # Official colour beats a brand line that says "Orange" on a white wine.
    tone = from_blob(normalize(wine.color))
    if tone != "unknown":
        return tone
    return from_blob(normalize(f"{wine.category} {wine.name}"))


def crop_color_features(image: Image.Image) -> Dict[str, Any]:
    sample = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    pixels = np.asarray(sample, dtype=np.float32) / 255.0
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    brightness = float(pixels.mean())
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    saturation = float((channel_max - channel_min).mean())
    paper = "cream" if brightness >= 0.55 and saturation < 0.28 else ("dark" if brightness < 0.32 else "mixed")
    luminance = pixels.mean(axis=2)
    red_liquid = float(((red > 0.28) & (red > green + 0.12) & (red > blue + 0.15) & (green < 0.28) & (luminance < 0.48)).mean())
    orange_liquid = float(
        (
            (red > 0.32)
            & (green > 0.16)
            & (red > blue + 0.06)
            & (green >= blue * 0.85)
            & ((red - green) < 0.28)
            & (luminance < 0.58)
        ).mean()
    )
    pale_liquid = float(((brightness > 0.58) & (green + blue > red * 1.05) & (saturation < 0.35)).mean())
    orange_print = float(
        (
            (red > 0.35)
            & (red > green + 0.08)
            & (red > blue + 0.12)
            & (luminance > 0.22)
            & (luminance < 0.72)
        ).mean()
    )
    if paper == "cream" or (paper == "mixed" and brightness >= 0.52 and pale_liquid >= 0.12):
        bottle_tone = "unknown"
    elif orange_liquid >= 0.08 and orange_liquid >= red_liquid:
        bottle_tone = "orange"
    elif red_liquid >= 0.08 and red_liquid > pale_liquid:
        bottle_tone = "red"
    elif pale_liquid >= 0.18 and paper != "dark":
        bottle_tone = "white"
    elif 0.04 <= red_liquid < 0.08 and brightness > 0.45:
        bottle_tone = "rose"
    else:
        bottle_tone = "unknown"
    if orange_print >= 0.05:
        print_tone = "orange"
    elif orange_liquid >= 0.05:
        print_tone = "orange"
    elif red_liquid >= 0.08:
        print_tone = "red"
    else:
        print_tone = "unknown"
    return {
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "paper": paper,
        "bottle_tone": bottle_tone,
        "print_tone": print_tone,
        "red_fraction": round(red_liquid, 4),
        "pale_fraction": round(pale_liquid, 4),
        "orange_fraction": round(orange_liquid, 4),
        "orange_print_fraction": round(orange_print, 4),
    }


def color_delta(features: Mapping[str, Any], wine: Optional[CatalogWine]) -> float:
    tone = wine_tone(wine)
    bottle = str(features.get("bottle_tone") or "unknown")
    print_tone = str(features.get("print_tone") or "unknown")
    paper = str(features.get("paper") or "mixed")
    if tone == "unknown":
        return 0.0
    # Paint on a cream label (orange dress, ochre type) is about the wine, not the glass.
    if print_tone == "orange":
        if tone == "orange":
            return COLOR_WEIGHT
        if tone in {"white", "red"}:
            return -COLOR_WEIGHT
    return 0.0


def ocr_delta(text: str, wine: Optional[CatalogWine], reference_text: str = "") -> float:
    if wine is None:
        return 0.0
    useful = {word for word in tokens(text) if len(word) >= 3 and word not in OCR_STOP}
    catalog = {word for word in tokens(f"{wine.name} {wine.winery} {reference_text}") if len(word) >= 3 and word not in OCR_STOP}
    if len(useful) < 2 or not catalog:
        return 0.0
    matched = {word for word in useful if word in catalog or (
        len(word) >= 6 and word.isalpha() and any(
            target.isalpha() and len(target) >= 6 and SequenceMatcher(None, word, target).ratio() >= .88
            for target in catalog))}
    # A shared vintage or grape alone is insufficient evidence for a bonus.
    if len([word for word in matched if word.isalpha()]) < 2:
        return 0.0
    overlap = len(matched) / min(8, len(useful))
    return round(min(OCR_WEIGHT, overlap * OCR_WEIGHT), 4)


def _visual_score(item: Mapping[str, Any]) -> float:
    scores = []
    for key in ("crop_score", "full_score", "siglip", "score"):
        value = item.get(key)
        if value is not None:
            scores.append(float(value))
    return max(scores) if scores else 0.0


def _lock_visual_leader(rows: list) -> list:
    """Color may break ties; it must not bury a clear label/bottle leader."""

    if len(rows) < 2:
        return rows
    leader = max(rows, key=_visual_score)
    others = [row for row in rows if row.get("slug") != leader.get("slug")]
    if not others:
        return rows
    runner = max(others, key=_visual_score)
    if _visual_score(leader) < _visual_score(runner) + VISUAL_LOCK:
        return rows
    ocr_best = max(rows, key=lambda row: float(row.get("ocr_delta") or 0.0))
    if float(ocr_best.get("ocr_delta") or 0.0) >= 0.06 and ocr_best.get("slug") != leader.get("slug"):
        if _visual_score(leader) - _visual_score(ocr_best) <= 0.05:
            return rows
    color_best = max(rows, key=lambda row: float(row.get("color_delta") or 0.0))
    if (
        float(leader.get("color_delta") or 0.0) < 0
        and float(color_best.get("color_delta") or 0.0) > 0
        and _visual_score(leader) - _visual_score(color_best) <= 0.10
    ):
        return rows
    rest = sorted(others, key=lambda row: row["score"], reverse=True)
    return [leader] + rest


def _distinguishing_tokens(wine: CatalogWine, family: Sequence[CatalogWine]) -> set:
    """Name/category/grape words of one family member that its siblings lack."""

    own = {word for word in tokens(f"{wine.name} {wine.category} {' '.join(wine.grapes)}") if len(word) >= 4 and word.isalpha()}
    others = set()
    for sibling in family:
        if sibling.slug != wine.slug:
            others |= tokens(f"{sibling.name} {sibling.category} {' '.join(sibling.grapes)}")
    return own - others


def _ocr_hit(word: str, text_tokens: set) -> bool:
    if word in text_tokens:
        return True
    return len(word) >= 6 and any(
        len(token) >= 6 and SequenceMatcher(None, word, token).ratio() >= 0.85 for token in text_tokens
    )


def family_tiebreak(rows: list, catalog, ocr_text: str, window: float = 0.08) -> list:
    """Within one producer's line, let OCR of sweetness/colour/grape words pick the SKU.

    OCR_STOP drops these words for cross-producer scoring because every label has
    them; between siblings they are the only thing that differs.
    """

    from .ranking import same_label_family

    if len(rows) < 2 or not ocr_text:
        return rows
    leader = catalog.get(rows[0]["slug"])
    if leader is None:
        return rows
    top_score = float(rows[0]["score"])
    family = [leader] + [
        catalog.get(row["slug"]) for row in rows[1:]
        if top_score - float(row["score"]) <= window and same_label_family(leader, catalog.get(row["slug"]))
    ]
    family = [wine for wine in family if wine is not None]
    if len(family) < 2:
        return rows
    text_tokens = tokens(ocr_text)
    hits = {wine.slug: sum(_ocr_hit(word, text_tokens) for word in _distinguishing_tokens(wine, family)) for wine in family}
    best = max(hits, key=hits.get)
    if hits[best] == 0 or list(hits.values()).count(hits[best]) > 1 or best == leader.slug:
        return rows
    chosen = next(row for row in rows if row["slug"] == best)
    return [{**chosen, "family_ocr": hits[best]}] + [row for row in rows if row["slug"] != best]


def blend_candidates(
    candidates: Sequence[Mapping[str, Any]],
    catalog,
    features: Mapping[str, Any],
    ocr_text: str,
    ocr_enabled: bool,
    ocr_references=None,
) -> list:
    blended = []
    for item in candidates:
        slug = item.get("slug")
        siglip = float(item.get("score") or 0.0)
        wine = catalog.get(slug) if slug else None
        color = color_delta(features, wine)
        reference = (ocr_references or {}).get(slug, "")
        ocr = ocr_delta(ocr_text, wine, reference) if ocr_enabled else 0.0
        score = max(0.0, min(1.0, siglip + color + ocr))
        blended.append(
            {
                "slug": slug,
                "score": score,
                "siglip": round(siglip, 4),
                "color_delta": round(color, 4),
                "ocr_delta": round(ocr, 4),
                "ocr_reference_text": reference[:500],
                "ocr_catalog_text": f"{wine.name} {wine.winery}" if wine else "",
                "image_hash": item.get("image_hash"),
                "full_score": item.get("full_score"),
                "crop_score": item.get("crop_score"),
                "best_view": item.get("best_view"),
            }
        )
    blended.sort(key=lambda row: row["score"], reverse=True)
    ranked = _lock_visual_leader(blended)
    if ocr_enabled and os.getenv("OCR_FAMILY", "false").lower() == "true":
        ranked = family_tiebreak(ranked, catalog, ocr_text)
    return ranked
