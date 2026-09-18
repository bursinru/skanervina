import shutil
import os
import logging
from time import perf_counter
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from .catalog import WineCatalog, normalize
from .label_detection import crop_label, detect_label, enhance_label
from .ranking import ranking_metrics
from .recommend import alternatives as recommend_alternatives
from .settings import Settings


class Recognizer:
    """Visual retrieval with OCR corroboration and explicit uncertainty."""

    MATCH_THRESHOLD = 0.55
    UNCERTAIN_THRESHOLD = 0.30
    MODES = {
        "combined",
        "image_full",
        "image_auto",
        "image_auto_enhanced",
        "ocr_full",
        "ocr_auto",
        "ocr_auto_enhanced",
    }

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

    def _ocr(self, image_bytes: bytes, psm: int = 6) -> Tuple[str, str]:
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
                    config=f"--psm {psm}",
                    timeout=20,
                )
            return text.strip(), "ok"
        except Exception:
            return "", "failed"

    @staticmethod
    def _decode_image(image_bytes: bytes):
        from PIL import Image, ImageOps

        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > 24_000_000:
                return None, "image_dimensions_too_large"
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB"), None

    @staticmethod
    def _image_bytes(image) -> bytes:
        output = BytesIO()
        image.save(output, format="JPEG", quality=94, optimize=True)
        return output.getvalue()

    @staticmethod
    def _mode_kind(mode: str) -> str:
        if mode.startswith("image_"):
            return "image"
        if mode.startswith("ocr_"):
            return "ocr"
        return "combined"

    def recognize(self, image_bytes: bytes, mode: str = "combined", include_candidates: bool = False) -> Dict[str, Any]:
        started = perf_counter()
        result = self._recognize(image_bytes, mode, include_candidates)
        result.setdefault('recognition', {}).setdefault('timings_ms', {})['total'] = round((perf_counter() - started) * 1000, 1)
        return result

    def _recognize(self, image_bytes: bytes, mode: str = "combined", include_candidates: bool = False) -> Dict[str, Any]:
        if mode not in self.MODES:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "input_validation", "reason": "unsupported_mode"},
            }
        decode_started = perf_counter()
        try:
            image, validation_error = self._decode_image(image_bytes)
            if validation_error:
                return {
                    'status': 'unknown',
                    'recognition': {
                        'method': 'input_validation',
                        'reason': validation_error,
                        'timings_ms': {'decode': round((perf_counter() - decode_started) * 1000, 1)},
                    },
                }
        except Exception:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {"method": "input_validation", "reason": "invalid_image"},
            }
        decode_ms = (perf_counter() - decode_started) * 1000

        detection_started = perf_counter()
        detection = detect_label(image)
        detection_ms = (perf_counter() - detection_started) * 1000
        crop_started = perf_counter()
        label = crop_label(image, detection)
        crop_ms = (perf_counter() - crop_started) * 1000
        kind = self._mode_kind(mode)
        if mode in {"image_auto_enhanced", "ocr_auto_enhanced"}:
            enhancement_started = perf_counter()
            enhanced = enhance_label(label)
            enhancement_ms = (perf_counter() - enhancement_started) * 1000
        else:
            enhanced = label
            enhancement_ms = 0.0
        if mode in {"image_full", "ocr_full"}:
            work_image = image
        elif mode in {"image_auto_enhanced", "ocr_auto_enhanced"}:
            work_image = enhanced
        else:
            work_image = label

        def base_metrics(method: str) -> Dict[str, Any]:
            return {
                "method": method,
                "label_detection": {
                    "bbox": [round(value, 4) for value in detection.bbox],
                    "confidence": detection.confidence,
                    "method": detection.method,
                },
                "timings_ms": {
                    "decode": round(decode_ms, 1),
                    "label_detection": round(detection_ms, 1),
                    "label_crop": round(crop_ms, 1),
                    "enhancement": round(enhancement_ms, 1),
                },
            }

        if kind == "ocr":
            ocr_started = perf_counter()
            text, ocr_status = self._ocr(self._image_bytes(work_image), psm=6)
            ocr_ms = (perf_counter() - ocr_started) * 1000
            return self._ocr_result(
                text,
                ocr_status,
                base_metrics("ocr+catalog"),
                ocr_ms,
                include_candidates,
            )

        if self.visual:
            visual_started = perf_counter()
            candidates = self.visual.search(work_image)
            visual_ms = (perf_counter() - visual_started) * 1000
            if not candidates:
                metrics = base_metrics("siglip2+pgvector")
                metrics["reason"] = "index_empty"
                metrics["timings_ms"]["visual"] = round(visual_ms, 1)
                return {'status': 'unknown', 'recognition': metrics}
            # Similarity is not a calibrated probability. Require separation from the runner-up.
            best = candidates[0]
            margin = best['score'] - candidates[1]['score'] if len(candidates) > 1 else 0
            wine = self.catalog.get(best['slug'])
            threshold = float(os.getenv('CV_MATCH_THRESHOLD', '0.88'))
            min_margin = float(os.getenv('CV_MATCH_MARGIN', '0.04'))
            text, ocr_status, text_matches, ocr_ms = "", "skipped", [], 0.0
            corroborated = False
            ocr_enabled = os.getenv('CV_OCR_ENABLED', 'false').lower() == 'true'
            if mode == "combined" and ocr_enabled:
                ocr_started = perf_counter()
                text, ocr_status = self._ocr(self._image_bytes(work_image), psm=6)
                ocr_ms = (perf_counter() - ocr_started) * 1000
                text_matches = self._text_matches(text)
                corroborated = bool(text_matches and text_matches[0].wine.slug == best['slug'] and text_matches[0].score >= self.MATCH_THRESHOLD and self._text_evidence(text_matches[0]))
            matched = wine and best['score'] >= threshold and (margin >= min_margin or corroborated)
            method = "siglip2+pgvector+ocr" if (mode == "combined" and ocr_enabled) else "siglip2+pgvector"
            metrics = base_metrics(method)
            metrics.update(
                similarity=round(best['score'], 4),
                margin=round(margin, 4),
                ocr=ocr_status,
                ocr_characters=len(text),
                ocr_best={'slug': text_matches[0].wine.slug, 'score': round(text_matches[0].score, 4)} if text_matches else None,
                ocr_corroborated=corroborated,
                ocr_text=text[:500],
                threshold=threshold,
                min_margin=min_margin,
            )
            ranking = ranking_metrics(candidates)
            metrics["candidates"] = ranking["top5"]
            metrics["timings_ms"].update(visual=round(visual_ms, 1), ocr=round(ocr_ms, 1))
            result = {
                'status': 'matched' if matched else ('uncertain' if best['score'] >= 0.65 else 'unknown'),
                'confidence': round(best['score'], 4),
                'recognition': metrics,
                'ranking': ranking,
            }
            if wine:
                result['slug'] = wine.slug
                result['alternatives'] = recommend_alternatives(self.catalog, wine)
                if result['status'] != 'unknown':
                    result['wine'] = wine.to_card()
            return result
        if kind == "image" or os.getenv('CV_ENABLED', 'false').lower() == 'true':
            metrics = base_metrics('siglip2')
            metrics['reason'] = 'visual_unavailable'
            return {'status': 'unknown', 'recognition': metrics}
        text, ocr_status = self._ocr(self._image_bytes(work_image), psm=6)
        return self._ocr_result(text, ocr_status, base_metrics('ocr+catalog'), 0.0, include_candidates)

    def _ocr_result(self, text, ocr_status, metrics, ocr_ms=0.0, include_candidates=False):
        metrics['ocr'] = ocr_status
        metrics['ocr_characters'] = len(text)
        metrics['timings_ms']['ocr'] = round(ocr_ms, 1)
        if not text:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {**metrics, "method": "ocr", "reason": ocr_status},
            }

        matches = self._text_matches(text)
        if include_candidates:
            metrics["candidates"] = [
                {"slug": match.wine.slug, "score": round(match.score, 4)}
                for match in matches[:5]
            ]
        if not matches:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "recognition": {**metrics, "reason": "no_candidates", "ocr_text": text[:500]},
            }

        best = matches[0]
        metrics["ocr_best"] = {"slug": best.wine.slug, "score": round(best.score, 4)}
        metrics["ocr_text"] = text[:500]
        if best.score >= self.MATCH_THRESHOLD and self._text_evidence(best):
            return {
                "status": "matched",
                "slug": best.wine.slug,
                "wine": best.wine.to_card(),
                "confidence": round(best.score, 3),
                "recognition": metrics,
            }
        if best.score >= self.UNCERTAIN_THRESHOLD:
            return {
                "status": "uncertain",
                "slug": best.wine.slug,
                "wine": best.wine.to_card(),
                "confidence": round(best.score, 3),
                "recognition": {**metrics, "reason": "low_confidence"},
            }
        return {
            "status": "unknown",
            "confidence": round(best.score, 3),
            "recognition": {**metrics, "reason": "no_confident_match"},
        }
