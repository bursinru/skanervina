"""Shared, local OCR for query labels and precomputed catalog references."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

OCR_VERSION = "label-rgb-upscale-psm6-v2"


def pipeline_version():
    detector = Path(__file__).with_name("label_detection.py").read_bytes()
    return OCR_VERSION + ":" + hashlib.sha256(detector).hexdigest()[:16]


def read_label(image, languages="rus+eng", psm=6):
    import pytesseract

    image = ImageOps.exif_transpose(image).convert("RGB")
    # Never shrink small print to the embedding model's 224px input.
    scale = min(2., 1600 / max(image.size), 900 / image.width)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    data = pytesseract.image_to_data(image, lang=languages, config=f"--psm {psm}",
                                    timeout=20, output_type=pytesseract.Output.DICT)
    words = [word.strip() for word, confidence in zip(data["text"], data["conf"])
             if float(confidence) >= 25 and sum(c.isalnum() for c in word) >= 3]
    return " ".join(words)


def load_references(path, languages="rus+eng"):
    try:
        payload = json.loads(Path(path).read_text())
        if (payload.get("pipeline_version") != pipeline_version()
                or payload.get("language_version") != pipeline_version() + ':' + languages):
            return {}
        return {slug: entry["text"] for slug, entry in payload["entries"].items()
                if entry.get("status") == "ok" and entry.get("text", "").strip()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
