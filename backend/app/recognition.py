import shutil
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from .catalog import WineCatalog
from .settings import Settings


class Recognizer:
    """First recognition baseline: OCR text followed by catalog matching.

    The visual embedding stage will be added behind this same interface. The
    HTTP contract therefore remains stable while recognition quality improves.
    """

    MATCH_THRESHOLD = 0.55
    UNCERTAIN_THRESHOLD = 0.30

    def __init__(self, catalog: WineCatalog, settings: Settings):
        self.catalog = catalog
        self.settings = settings

    def _ocr(self, image_bytes: bytes) -> Tuple[str, str]:
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            return "", "dependency_missing"
        if not shutil.which("tesseract"):
            return "", "binary_missing"

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                text = pytesseract.image_to_string(
                    image.convert("RGB"),
                    lang=self.settings.ocr_languages,
                    config="--psm 6",
                )
            return text.strip(), "ok"
        except Exception:
            return "", "failed"

    def recognize(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except Exception:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "input_validation", "reason": "invalid_image"},
            }

        text, ocr_status = self._ocr(image_bytes)
        if not text:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "ocr", "reason": ocr_status},
            }

        matches = self.catalog.search(text)
        if not matches:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "ocr+catalog", "reason": "no_candidates"},
            }

        best = matches[0]
        if best.score >= self.MATCH_THRESHOLD:
            return {
                "status": "matched",
                "slug": best.wine.slug,
                "wine": best.wine.to_card(),
                "confidence": round(best.score, 3),
                "recognition": {"method": "ocr+catalog", "ocr": ocr_status},
            }
        if best.score >= self.UNCERTAIN_THRESHOLD:
            return {
                "status": "uncertain",
                "confidence": round(best.score, 3),
                "recognition": {"method": "ocr+catalog", "reason": "low_confidence"},
            }
        return {
            "status": "unknown",
            "confidence": round(best.score, 3),
            "recognition": {"method": "ocr+catalog", "reason": "no_confident_match"},
        }
