"""Cheap label-color and OCR extras on top of SigLIP cosine scores."""

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .catalog import CatalogWine, normalize, tokens

COLOR_WEIGHT = 0.08
OCR_WEIGHT = 0.12
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
    blob = normalize(f"{wine.color} {wine.category}")
    if "роз" in blob or "rose" in blob:
        return "rose"
    if "оранж" in blob:
        return "orange"
    if "красн" in blob or "red" in blob:
        return "red"
    if "бел" in blob or "white" in blob:
        return "white"
    return "unknown"


def crop_color_features(image: Image.Image) -> Dict[str, Any]:
    sample = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    pixels = np.asarray(sample, dtype=np.float32) / 255.0
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    brightness = float(pixels.mean())
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    saturation = float((channel_max - channel_min).mean())
    paper = "cream" if brightness >= 0.55 and saturation < 0.28 else ("dark" if brightness < 0.32 else "mixed")
    red_liquid = float(((red > 0.28) & (red > green + 0.07) & (red > blue + 0.07) & (brightness < 0.62)).mean())
    pale_liquid = float(((brightness > 0.58) & (green + blue > red * 1.05) & (saturation < 0.35)).mean())
    if red_liquid >= 0.08 and red_liquid > pale_liquid:
        bottle_tone = "red"
    elif pale_liquid >= 0.18 and paper != "dark":
        bottle_tone = "white"
    elif 0.04 <= red_liquid < 0.08 and brightness > 0.45:
        bottle_tone = "rose"
    else:
        bottle_tone = "unknown"
    return {
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "paper": paper,
        "bottle_tone": bottle_tone,
        "red_fraction": round(red_liquid, 4),
        "pale_fraction": round(pale_liquid, 4),
    }


def color_delta(features: Mapping[str, Any], wine: Optional[CatalogWine]) -> float:
    tone = wine_tone(wine)
    bottle = str(features.get("bottle_tone") or "unknown")
    paper = str(features.get("paper") or "mixed")
    if tone == "unknown":
        return 0.0
    if bottle == tone or (bottle == "white" and tone == "orange"):
        return COLOR_WEIGHT
    if bottle == "unknown":
        if paper == "cream" and tone in {"white", "orange", "rose"}:
            return COLOR_WEIGHT * 0.35
        if paper == "dark" and tone == "red":
            return COLOR_WEIGHT * 0.2
        return 0.0
    if {bottle, tone} == {"white", "rose"}:
        return 0.0
    return -COLOR_WEIGHT


def ocr_delta(text: str, wine: Optional[CatalogWine]) -> float:
    if wine is None:
        return 0.0
    useful = {word for word in tokens(text) if len(word) >= 3 and word not in OCR_STOP}
    catalog = {word for word in tokens(f"{wine.name} {wine.winery}") if len(word) >= 3 and word not in OCR_STOP}
    if len(useful) < 2 or not catalog:
        return 0.0
    overlap = len(useful & catalog) / min(8, len(useful))
    return round(min(OCR_WEIGHT, overlap * OCR_WEIGHT), 4)


def blend_candidates(
    candidates: Sequence[Mapping[str, Any]],
    catalog,
    features: Mapping[str, Any],
    ocr_text: str,
    ocr_enabled: bool,
) -> list:
    blended = []
    for item in candidates:
        slug = item.get("slug")
        siglip = float(item.get("score") or 0.0)
        wine = catalog.get(slug) if slug else None
        color = color_delta(features, wine)
        ocr = ocr_delta(ocr_text, wine) if ocr_enabled else 0.0
        score = max(0.0, min(1.0, siglip + color + ocr))
        blended.append(
            {
                "slug": slug,
                "score": score,
                "siglip": round(siglip, 4),
                "color_delta": round(color, 4),
                "ocr_delta": round(ocr, 4),
            }
        )
    blended.sort(key=lambda row: row["score"], reverse=True)
    return blended
