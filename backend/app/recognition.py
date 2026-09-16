import shutil
import os
import logging
from time import perf_counter
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from .catalog import WineCatalog, normalize
from .settings import Settings


class Recognizer:
    """Visual retrieval with OCR corroboration and explicit uncertainty."""

    MATCH_THRESHOLD = 0.55
    UNCERTAIN_THRESHOLD = 0.30

    def __init__(self, catalog: WineCatalog, settings: Settings):
        self.catalog = catalog
        self.settings = settings
        self.visual = None
        self.visual_status = 'disabled'
        if os.getenv('CV_ENABLED', 'false').lower() == 'true':
            try:
                from .vision import VisualSearch
                self.visual = VisualSearch()
                self.visual_status = 'ready'
            except Exception:
                logging.exception('Could not initialize visual search')
                self.visual_status = 'unavailable'


    def _text_matches(self, text):
        # A short OCR artefact such as "ii" can otherwise get a high overlap score.
        useful = [word for word in normalize(text).split() if len(word) >= 3 and any(c.isalpha() for c in word)]
        return self.catalog.search(' '.join(useful)) if useful else []

    @staticmethod
    def _text_evidence(match):
        return len([word for word in match.matched_tokens if len(word) >= 3 and any(c.isalpha() for c in word)]) >= 2

    def _ocr(self, image_bytes: bytes) -> Tuple[str, str]:
        try:
            from PIL import Image, ImageOps
            import pytesseract
        except ImportError:
            return "", "dependency_missing"
        if not shutil.which("tesseract"):
            return "", "binary_missing"

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                image.thumbnail((1000, 1000))
                text = pytesseract.image_to_string(
                    image.convert("RGB"),
                    lang=self.settings.ocr_languages,
                    config="--psm 6",
                    timeout=20,
                )
            return text.strip(), "ok"
        except Exception:
            return "", "failed"

    def recognize(self, image_bytes: bytes) -> Dict[str, Any]:
        started = perf_counter()
        result = self._recognize(image_bytes)
        result.setdefault('recognition', {}).setdefault('timings_ms', {})['total'] = round((perf_counter() - started) * 1000, 1)
        return result

    def _recognize(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as image:
                if image.width * image.height > 24_000_000:
                    return {'status': 'unknown', 'recognition': {'method': 'input_validation', 'reason': 'image_dimensions_too_large'}}
                image.verify()
        except Exception:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "input_validation", "reason": "invalid_image"},
            }

        if self.visual:
            visual_started = perf_counter()
            with Image.open(BytesIO(image_bytes)) as image:
                candidates = self.visual.search(image)
            visual_ms = (perf_counter() - visual_started) * 1000
            if not candidates:
                return {'status': 'unknown', 'recognition': {'method': 'siglip2+pgvector', 'reason': 'index_empty'}}
            # Similarity is not a calibrated probability. Require separation from the runner-up.
            best = candidates[0]
            margin = best['score'] - candidates[1]['score'] if len(candidates) > 1 else 0
            wine = self.catalog.get(best['slug'])
            threshold = float(os.getenv('CV_MATCH_THRESHOLD', '0.88'))
            min_margin = float(os.getenv('CV_MATCH_MARGIN', '0.04'))
            ocr_started = perf_counter()
            text, ocr_status = self._ocr(image_bytes)
            ocr_ms = (perf_counter() - ocr_started) * 1000
            text_matches = self._text_matches(text)
            corroborated = bool(text_matches and text_matches[0].wine.slug == best['slug'] and text_matches[0].score >= self.MATCH_THRESHOLD and self._text_evidence(text_matches[0]))
            matched = wine and best['score'] >= threshold and (margin >= min_margin or corroborated)
            result = {
                'status': 'matched' if matched else ('uncertain' if best['score'] >= 0.65 else 'unknown'),
                'confidence': round(best['score'], 4),
                'recognition': {'method': 'siglip2+pgvector+ocr', 'similarity': round(best['score'], 4),
                                'margin': round(margin, 4), 'ocr': ocr_status,
                                'ocr_characters': len(text),
                                'ocr_best': {'slug': text_matches[0].wine.slug, 'score': round(text_matches[0].score, 4)} if text_matches else None,
                                'ocr_corroborated': corroborated,
                                'timings_ms': {'visual': round(visual_ms, 1), 'ocr': round(ocr_ms, 1)},
                                'threshold': threshold, 'min_margin': min_margin},
            }
            if wine and result['status'] != 'unknown':
                result.update(slug=wine.slug, wine=wine.to_card())
            return result
        if os.getenv('CV_ENABLED', 'false').lower() == 'true':
            return {'status': 'unknown', 'recognition': {'method': 'siglip2', 'reason': 'visual_unavailable'}}
        text, ocr_status = self._ocr(image_bytes)
        if not text:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "ocr", "reason": ocr_status},
            }

        matches = self._text_matches(text)
        if not matches:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "ocr+catalog", "reason": "no_candidates"},
            }

        best = matches[0]
        if best.score >= self.MATCH_THRESHOLD and self._text_evidence(best):
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
                "slug": best.wine.slug,
                "wine": best.wine.to_card(),
                "confidence": round(best.score, 3),
                "recognition": {"method": "ocr+catalog", "reason": "low_confidence"},
            }
        return {
            "status": "unknown",
            "confidence": round(best.score, 3),
            "recognition": {"method": "ocr+catalog", "reason": "no_confident_match"},
        }
