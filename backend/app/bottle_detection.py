"""Detect bottle instances and rank the likely foreground target."""

from math import sqrt
from threading import Lock
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image


MODEL_ID = "PekingU/rtdetr_r18vd"
MODEL_REVISION = "88f462f2350473029c7f938f14c9d2a565933e3f"
DETECTION_THRESHOLD = 0.25


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rank_bottle_candidates(
    detections: Iterable[Dict[str, Any]], image_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """Normalize bottle boxes and rank by size, centrality, and image position."""

    image_width, image_height = image_size
    if image_width < 1 or image_height < 1:
        return []

    candidates = []
    for detection in detections:
        if str(detection.get("label", "")).strip().lower() != "bottle":
            continue
        confidence = float(detection.get("score", 0.0))
        if confidence < DETECTION_THRESHOLD:
            continue
        box = detection.get("box")
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        left, top, right, bottom = (float(value) for value in box)
        left = _clamp(left / image_width, 0.0, 1.0)
        top = _clamp(top / image_height, 0.0, 1.0)
        right = _clamp(right / image_width, 0.0, 1.0)
        bottom = _clamp(bottom / image_height, 0.0, 1.0)
        box_width, box_height = right - left, bottom - top
        area = box_width * box_height
        if (
            box_width < 0.045
            or box_height < 0.18
            or area < 0.015
            or area > 0.92
            or box_height / max(box_width, 1e-6) < 1.12
        ):
            continue

        centre_x = (left + right) / 2
        size_score = min(1.0, sqrt(area) / 0.62)
        centre_score = 1.0 - min(1.0, abs(centre_x - 0.5) / 0.5)
        lower_score = _clamp((bottom - 0.35) / 0.60, 0.0, 1.0)
        edge_penalty = 0.10 if left < 0.015 or right > 0.985 else 0.0
        priority = (
            0.48 * size_score
            + 0.24 * centre_score
            + 0.16 * lower_score
            + 0.12 * confidence
            - edge_penalty
        )
        candidates.append(
            {
                "bbox": [round(left, 5), round(top, 5), round(right, 5), round(bottom, 5)],
                "confidence": round(confidence, 4),
                "priority": round(priority, 4),
                "area": area,
            }
        )

    candidates.sort(key=lambda item: item["priority"], reverse=True)

    # DETR-family models are set-based, but discard any remaining near-duplicate
    # boxes in case a future checkpoint emits overlapping instances.
    unique = []
    for candidate in candidates:
        left, top, right, bottom = candidate["bbox"]
        duplicate = False
        for kept in unique:
            kl, kt, kr, kb = kept["bbox"]
            iw = max(0.0, min(right, kr) - max(left, kl))
            ih = max(0.0, min(bottom, kb) - max(top, kt))
            intersection = iw * ih
            union = candidate["area"] + kept["area"] - intersection
            if intersection / max(union, 1e-8) > 0.72:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    for candidate in unique:
        candidate.pop("area")
    return unique


def selection_is_ambiguous(candidates: Sequence[Dict[str, Any]]) -> bool:
    """Ask a web user only when two foreground bottle candidates are comparable."""

    if len(candidates) < 2:
        return False
    first, second = candidates[:2]
    first_box, second_box = first["bbox"], second["bbox"]
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    return (
        second_area >= first_area * 0.68
        and first["priority"] - second["priority"] <= 0.11
        and abs(first["confidence"] - second["confidence"]) <= 0.22
    )


class BottleDetector:
    """Lazy CPU/GPU object detector; weights are cached by Hugging Face Hub."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._lock = Lock()
        self._torch = None
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_fast=False
        )
        model = AutoModelForObjectDetection.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        )
        model.to(self.device).eval()
        self._torch = torch
        self._processor = processor
        self._model = model

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        with self._lock:
            self._load()
            rgb = image.convert("RGB")
            inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            target_sizes = self._torch.tensor([rgb.size[::-1]], device=self.device)
            result = self._processor.post_process_object_detection(
                outputs,
                threshold=DETECTION_THRESHOLD,
                target_sizes=target_sizes,
            )[0]
            labels = self._model.config.id2label
            detections = [
                {
                    "label": labels.get(int(label), labels.get(str(int(label)), "")),
                    "score": float(score),
                    "box": [float(value) for value in box],
                }
                for score, label, box in zip(
                    result["scores"].tolist(),
                    result["labels"].tolist(),
                    result["boxes"].tolist(),
                )
            ]
            candidates = rank_bottle_candidates(detections, rgb.size)

        return {
            "available": True,
            "count": len(candidates),
            "candidates": candidates,
            "primary_box": candidates[0]["bbox"] if candidates else None,
            "needs_selection": selection_is_ambiguous(candidates),
        }
