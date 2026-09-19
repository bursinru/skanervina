"""Fast, dependency-light label localization for user photos.

This is intentionally a conservative detector.  It finds the most prominent
central rectangular panel and keeps a small margin around it, which is safer
for OCR than an overly tight contour.  The detector returns a bbox for
diagnostics, but user photos are never written to disk.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


NormalizedBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LabelDetection:
    bbox: NormalizedBox
    confidence: float
    method: str = "label_panel"


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _mean(integral: np.ndarray, left: int, top: int, right: int, bottom: int) -> float:
    area = max(1, (right - left) * (bottom - top))
    return float(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    ) / area


def _candidate_score(
    edge: np.ndarray,
    gray_integral: np.ndarray,
    edge_integral: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> float:
    width = right - left
    height = bottom - top
    margin_x = max(1, round(width * 0.035))
    margin_y = max(1, round(height * 0.035))

    border = (
        _mean(edge_integral, left, top, right, min(bottom, top + margin_y))
        + _mean(edge_integral, left, max(top, bottom - margin_y), right, bottom)
        + _mean(edge_integral, left, top, min(right, left + margin_x), bottom)
        + _mean(edge_integral, max(left, right - margin_x), top, right, bottom)
    ) / 4

    inner_left = min(right - 1, left + margin_x * 3)
    inner_top = min(bottom - 1, top + margin_y * 3)
    inner_right = max(inner_left + 1, right - margin_x * 3)
    inner_bottom = max(inner_top + 1, bottom - margin_y * 3)
    interior_edge = _mean(edge_integral, inner_left, inner_top, inner_right, inner_bottom)

    outer_left = max(0, left - margin_x * 2)
    outer_top = max(0, top - margin_y * 2)
    outer_right = min(edge.shape[1], right + margin_x * 2)
    outer_bottom = min(edge.shape[0], bottom + margin_y * 2)
    outer_area = max(1, (outer_right - outer_left) * (outer_bottom - outer_top))
    candidate_area = max(1, width * height)
    outside_area = max(1, outer_area - candidate_area)
    outer_mean = (
        _mean(gray_integral, outer_left, outer_top, outer_right, outer_bottom) * outer_area
        - _mean(gray_integral, left, top, right, bottom) * candidate_area
    ) / outside_area
    inside_mean = _mean(gray_integral, inner_left, inner_top, inner_right, inner_bottom)
    contrast = min(1.0, abs(inside_mean - outer_mean) * 2.2)

    # Labels are normally near the visual centre of a photo.  This keeps a
    # shelf edge or a neighbouring bottle from winning on texture alone.
    center_x = (left + right) / 2 / edge.shape[1]
    center_y = (top + bottom) / 2 / edge.shape[0]
    center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.52) ** 2) ** 0.5
    centrality = max(0.0, 1.0 - center_distance / 0.55)

    width_frac = width / edge.shape[1]
    height_frac = height / edge.shape[0]
    area = width_frac * height_frac
    aspect = height_frac / max(width_frac, 1e-6)
    # Front labels are portrait paper, not a landscape laptop screen.
    if 1.05 <= aspect <= 2.55:
        aspect_score = 1.0
    elif 0.78 <= aspect < 1.05:
        aspect_score = 0.72
    elif aspect < 0.62:
        aspect_score = 0.08
    else:
        aspect_score = 0.4
    if 0.07 <= area <= 0.38:
        area_score = 1.0
    elif area <= 0.52:
        area_score = 0.55
    elif area >= 0.68:
        area_score = 0.05
    else:
        area_score = 0.28
    # Cream/white paper sits in a mid-bright band; blown UI or black bezels lose.
    paper = 1.0 - min(1.0, abs(inside_mean - 0.62) / 0.62)

    score = (
        min(1.0, border * 3.1) * 0.34
        + contrast * 0.18
        + min(1.0, interior_edge * 3.0) * 0.08
        + centrality * 0.10
        + aspect_score * 0.16
        + area_score * 0.10
        + paper * 0.04
    )
    return float(score)


def detect_label(image: Image.Image) -> LabelDetection:
    """Return a conservative normalized bbox for the likely front label."""

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((720, 960), Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    gray = pixels.mean(axis=2)
    edge = np.zeros_like(gray)
    edge[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    edge[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    # A tiny blur suppresses JPEG grain while retaining label borders and text.
    edge = np.asarray(
        Image.fromarray(np.clip(edge * 255, 0, 255).astype("uint8"))
        .filter(ImageFilter.GaussianBlur(radius=0.7)),
        dtype=np.float32,
    ) / 255.0
    height, width = gray.shape
    gray_integral = _integral(gray)
    edge_integral = _integral(edge)

    # The grid is intentionally small: this runs before both visual retrieval
    # and OCR and should add only a few milliseconds on a phone photo.
    widths = (0.22, 0.30, 0.38, 0.48, 0.58, 0.70)
    heights = (0.30, 0.40, 0.50, 0.60, 0.72)
    centres_x = (0.32, 0.42, 0.50, 0.58, 0.68)
    # The front label is normally in the middle half of a bottle photo.  Do
    # not let a shelf rail or the table below the bottle win on edge density.
    centres_y = (0.30, 0.40, 0.50, 0.58, 0.66)
    best_score = -1.0
    best_box = (0.28, 0.22, 0.72, 0.78)
    for box_width in widths:
        for box_height in heights:
            for centre_x in centres_x:
                for centre_y in centres_y:
                    left = max(0, round((centre_x - box_width / 2) * width))
                    top = max(0, round((centre_y - box_height / 2) * height))
                    right = min(width, round((centre_x + box_width / 2) * width))
                    bottom = min(height, round((centre_y + box_height / 2) * height))
                    if right - left < width * 0.16 or bottom - top < height * 0.22:
                        continue
                    score = _candidate_score(
                        edge, gray_integral, edge_integral, left, top, right, bottom
                    )
                    if score > best_score:
                        best_score = score
                        best_box = (left / width, top / height, right / width, bottom / height)

    # Prefer a tighter nested box when the paper panel is smaller than the first hit.
    left, top, right, bottom = best_box
    for shrink in (0.12, 0.22):
        inner = (
            left + (right - left) * shrink,
            top + (bottom - top) * shrink,
            right - (right - left) * shrink,
            bottom - (bottom - top) * shrink,
        )
        pixel = (
            max(0, round(inner[0] * width)),
            max(0, round(inner[1] * height)),
            min(width, round(inner[2] * width)),
            min(height, round(inner[3] * height)),
        )
        if pixel[2] - pixel[0] < width * 0.16 or pixel[3] - pixel[1] < height * 0.22:
            continue
        score = _candidate_score(edge, gray_integral, edge_integral, *pixel)
        if score > best_score + 0.015:
            best_score = score
            best_box = (pixel[0] / width, pixel[1] / height, pixel[2] / width, pixel[3] / height)

    # Add margin so small text at the edge of a label is not clipped.
    left, top, right, bottom = best_box
    pad_x = max(0.02, (right - left) * 0.06)
    pad_y = max(0.02, (bottom - top) * 0.06)
    result = (
        max(0.0, left - pad_x),
        max(0.0, top - pad_y),
        min(1.0, right + pad_x),
        min(1.0, bottom + pad_y),
    )
    return LabelDetection(result, round(max(0.0, min(1.0, best_score)), 4))


def crop_label(image: Image.Image, detection: LabelDetection) -> Image.Image:
    """Crop a detected label from the original-resolution image."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    left, top, right, bottom = detection.bbox
    return image.crop(
        (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )
    )


def enhance_label(image: Image.Image) -> Image.Image:
    """Create a mild OCR/visual variant without changing the crop geometry."""

    image = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=105, threshold=3))
