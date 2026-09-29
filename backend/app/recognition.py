import shutil
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from io import BytesIO
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .catalog import WineCatalog, normalize
from .label_detection import LabelDetection, crop_front_design, crop_label, crop_quality, detect_label, enhance_label, label_rgb
from .label_signals import blend_candidates, crop_color_features, prefer_distinctive_label
from .ranking import distinct_margin, is_visual_match, ranking_metrics, same_label_family
from .recommend import alternatives as recommend_alternatives
from .settings import Settings
from .label_ocr import read_label, load_references
from .vision import merge_query_views
from pathlib import Path

# PP-OCR of a query runs next to SigLIP encoding.
_OCR_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix='label-ocr')
# PP-OCR shares 4 cores with SigLIP; a 12 MP frame took 6-9 s to read.
OCR_FRAME_SIDE = int(os.getenv('CV_OCR_FRAME_SIDE', '1280'))
# Cropping to the detected bottle before search: on 107 shop and live photos it
# fixed 7, broke 12 and cost ~0.6 s. The user-picked box (/v1/bottles) still works.
BOTTLE_AUTO = os.getenv('CV_BOTTLE_AUTO', 'false').lower() == 'true'
LEAD_MATCH = float(os.getenv('CV_FUSION_LEAD_MATCH', '0.6'))
VISUAL_LEAD = float(os.getenv('CV_FUSION_VISUAL_LEAD', '0.06'))
# A majority share with a clear gap opens the card. The percents on screen are
# one distribution, so 57% against 19% is already a decided leader.
SHARE_MATCH = float(os.getenv('CV_FUSION_SHARE_MATCH', '0.5'))
SHARE_GAP = float(os.getenv('CV_FUSION_SHARE_GAP', '0.1'))
TEXT_LEAD_MATCH = float(os.getenv('CV_FUSION_TEXT_MATCH', '0.4'))
TEXT_LEAD = float(os.getenv('CV_FUSION_TEXT_LEAD', '0.25'))
TEXT_COVERAGE = float(os.getenv('CV_FUSION_TEXT_COVERAGE', '0.2'))


def share_leads(probability: float, runner: float, minimum: float = SHARE_MATCH, gap: float = SHARE_GAP) -> bool:
    return probability + 1e-9 >= minimum and probability - runner + 1e-9 >= gap


# A narrow panel on a wide frame is one bottle on a shelf. Text outside that
# panel belongs to the neighbour and must not name the wine.
SHELF_MAX_WIDTH = 0.48
SHELF_MAX_AREA = 0.22


def shelf_region(bbox) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or width >= SHELF_MAX_WIDTH or width * height >= SHELF_MAX_AREA:
        return None
    return (float(left), float(top), float(right), float(bottom))


def _style_band_lines(label):
    """Upscaled bottom of a small label: the gold sweetness line is otherwise unread."""

    from PIL import Image, ImageOps

    from .pp_ocr import read_lines

    width, height = label.size
    if width < 20 or height < 40:
        return [], (1, 1)
    band = label.crop((0, int(height * 0.58), width, height))
    long_side = max(band.size)
    if 0 < long_side < 1000:
        scale = 1000 / long_side
        band = band.resize(
            (max(1, round(band.width * scale)), max(1, round(band.height * scale))),
            Image.Resampling.LANCZOS,
        )
    band = ImageOps.autocontrast(band.convert("L")).convert("RGB")
    return read_lines(band), band.size


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
        self.ocr_references = load_references(os.getenv('CATALOG_OCR_INDEX', str(Path(__file__).resolve().parents[1] / 'data/catalog-ocr.json')), settings.ocr_languages)
        self.visual = None
        self.visual_status = 'disabled'
        self._catalog_crop_cache: Dict[str, Optional[str]] = {}
        self._bottle_detector = None
        self._bottle_detector_error = None
        self._bottle_detector_lock = Lock()
        if os.getenv('CV_ENABLED', 'false').lower() == 'true':
            try:
                from .vision import VisualSearch
                self.visual = VisualSearch()
                self.visual_status = 'ready'
            except Exception:
                logging.exception('Could not initialize visual search')
                self.visual_status = 'unavailable'
        self.fusion = None
        self.text_index = None
        self.photo_index = None
        if os.getenv('CV_FUSION', 'true').lower() == 'true':
            self._load_fusion()

    def _load_fusion(self):
        """Visual + label-text ranker. Missing weights keep the older cascade."""

        from .fusion import PHOTO_TEXT_PATH, load_weights
        from .label_text import TextIndex, catalog_documents

        weights = load_weights()
        if weights is None:
            logging.warning('fusion.json missing or stale; label text fusion is off')
            return
        wines = list(self.catalog)
        self.text_index = TextIndex(catalog_documents(wines), wines)
        try:
            photo = json.loads(PHOTO_TEXT_PATH.read_text())
            self.photo_index = TextIndex({slug: [tuple(word) for word in words] for slug, words in photo.items()})
        except (OSError, ValueError):
            logging.warning('Catalog photo text missing; fusion reads card fields only')
        self.fusion = weights


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
            from .image_io import register_decoders

            register_decoders()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                text = read_label(image, self.settings.ocr_languages, psm)
            return text.strip(), "ok"
        except Exception:
            return "", "failed"

    def _read_sibling(self, image) -> Tuple[str, str, list]:
        """PP-OCRv5 for a close series pair. Tesseract covers a missing model."""

        from .catalog import tokens
        from .pp_ocr import line_text, read_lines

        lines = read_lines(image)
        text = line_text(lines)
        cyrillic = [
            word for word in tokens(text)
            if len(word) >= 4 and any("а" <= character <= "я" or character == "ё" for character in word)
        ]
        if cyrillic:
            return text, "ppocr", lines
        fallback, status = self._ocr(self._image_bytes(image), psm=6)
        return fallback, status, lines

    def _prefer_grape_line(self, ranked, image, lines):
        from .grape_lines import crop_quad, load_index, prefer_by_grape_score, select_query_line

        index = load_index()
        if not index or self.visual is None or not hasattr(self.visual, "score_grape_views"):
            return ranked
        slugs = []
        leader_visual = float(ranked[0].get("siglip") or ranked[0].get("score") or 0.0) if ranked else 0.0
        for row in ranked[:5]:
            visual = float(row.get("siglip") or row.get("score") or 0.0)
            if leader_visual - visual <= 0.06 and row.get("slug") in index:
                slugs.append(row["slug"])
        if len(slugs) < 2:
            return ranked
        wines = [self.catalog.get(slug) for slug in slugs]
        box = select_query_line(lines, wines)
        crop = crop_quad(image, box)
        if crop is None:
            return ranked
        scores = self.visual.score_grape_views(crop, {slug: index[slug] for slug in slugs})
        return prefer_by_grape_score(ranked, scores)

    @staticmethod
    def _decode_image(image_bytes: bytes):
        from PIL import Image
        from .image_io import register_decoders

        register_decoders()
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > 24_000_000:
                return None, "image_dimensions_too_large"
            image.load()
            return label_rgb(image), None

    @staticmethod
    def _image_bytes(image) -> bytes:
        output = BytesIO()
        image.save(output, format="JPEG", quality=94, optimize=True)
        return output.getvalue()

    @staticmethod
    def _preview_jpeg(image, max_side: int = 360) -> str:
        import base64

        preview = image.convert("RGB")
        preview.thumbnail((max_side, max_side))
        output = BytesIO()
        preview.save(output, format="JPEG", quality=62, optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _mode_kind(mode: str) -> str:
        if mode.startswith("image_"):
            return "image"
        if mode.startswith("ocr_"):
            return "ocr"
        return "combined"

    def recognize(self, image_bytes: bytes, mode: str = "combined", include_candidates: bool = False, ocr_enabled=None, compare_slug=None, bottle_box=None) -> Dict[str, Any]:
        started = perf_counter()
        result = self._recognize(image_bytes, mode, include_candidates, ocr_enabled, compare_slug, bottle_box)
        result.setdefault('recognition', {}).setdefault('timings_ms', {})['total'] = round((perf_counter() - started) * 1000, 1)
        return result

    def detect_bottles(self, image_bytes: bytes) -> Dict[str, Any]:
        """Return bottle-instance proposals for the web's optional picker."""

        try:
            image, validation_error = self._decode_image(image_bytes)
        except Exception:
            image, validation_error = None, "invalid_image"
        if validation_error:
            return {"available": False, "count": 0, "candidates": [], "reason": validation_error}
        result = self._detect_bottles_in_image(image)
        return result or {"available": False, "count": 0, "candidates": [], "reason": "unavailable"}

    def _detect_bottles_in_image(self, image):
        if os.getenv('CV_ENABLED', 'false').lower() != 'true' or self._bottle_detector_error:
            return None
        with self._bottle_detector_lock:
            if self._bottle_detector is None:
                try:
                    from .bottle_detection import BottleDetector

                    self._bottle_detector = BottleDetector(os.getenv('CV_DEVICE', 'cpu'))
                except Exception as error:
                    self._bottle_detector_error = error
                    logging.exception('Could not initialize bottle detector')
                    return None
        try:
            return self._bottle_detector.detect(image)
        except Exception as error:
            self._bottle_detector_error = error
            logging.exception('Bottle detection is unavailable; using label-only fallback')
            return None

    @staticmethod
    def _crop_bottle(image, box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return image, None
        try:
            left, top, right, bottom = (float(value) for value in box)
        except (TypeError, ValueError):
            return image, None
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            return image, None
        width, height = image.size
        pixel_left, pixel_top = round(left * width), round(top * height)
        pixel_right, pixel_bottom = round(right * width), round(bottom * height)
        pad_x = max(2, round((pixel_right - pixel_left) * 0.035))
        pad_y = max(2, round((pixel_bottom - pixel_top) * 0.02))
        bounds = (
            max(0, pixel_left - pad_x),
            max(0, pixel_top - pad_y),
            min(width, pixel_right + pad_x),
            min(height, pixel_bottom + pad_y),
        )
        if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) > width * height * 0.94:
            return image, None
        return image.crop(bounds), bounds

    @staticmethod
    def _map_detection_to_image(detection: LabelDetection, crop_bounds, image_size):
        if crop_bounds is None:
            return detection
        left, top, right, bottom = crop_bounds
        image_width, image_height = image_size
        crop_width, crop_height = right - left, bottom - top

        def point(x, y):
            return ((left + x * crop_width) / image_width, (top + y * crop_height) / image_height)

        box = detection.bbox
        mapped_box = (
            (left + box[0] * crop_width) / image_width,
            (top + box[1] * crop_height) / image_height,
            (left + box[2] * crop_width) / image_width,
            (top + box[3] * crop_height) / image_height,
        )
        mapped_quad = tuple(point(x, y) for x, y in detection.quad) if detection.quad else None
        mapped_contour = tuple(point(x, y) for x, y in detection.contour) if detection.contour else None
        return LabelDetection(mapped_box, detection.confidence, detection.method, mapped_quad, mapped_contour)

    def _recognize(self, image_bytes: bytes, mode: str = "combined", include_candidates: bool = False, ocr_enabled=None, compare_slug=None, bottle_box=None) -> Dict[str, Any]:
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
        bottle_result = None
        selected_box = bottle_box
        selection_method = 'user' if bottle_box is not None else 'automatic'
        bottle_detection_ms = 0.0
        if selected_box is None and not BOTTLE_AUTO:
            selection_method = 'label_fallback'
        elif selected_box is None:
            bottle_started = perf_counter()
            bottle_result = self._detect_bottles_in_image(image)
            bottle_detection_ms = (perf_counter() - bottle_started) * 1000
            if bottle_result and bottle_result.get('available'):
                selected_box = bottle_result.get('primary_box')
            if selected_box is None:
                selection_method = 'label_fallback'
        search_image, bottle_crop_bounds = self._crop_bottle(image, selected_box)
        local_detection = detect_label(search_image)
        display_detection = self._map_detection_to_image(local_detection, bottle_crop_bounds, image.size)
        detection_ms = (perf_counter() - detection_started) * 1000
        crop_started = perf_counter()
        quality = crop_quality(search_image, local_detection)
        if quality['usable']:
            label = crop_label(search_image, local_detection, preserve_pixels=True)
            query_view = 'label'
        elif 'small_upper_fragment' in quality['reasons']:
            label = crop_front_design(search_image, local_detection)
            query_view = 'front_design'
        else:
            label = search_image
            query_view = 'full_image'
        crop_ms = (perf_counter() - crop_started) * 1000
        kind = self._mode_kind(mode)
        if mode in {"image_auto_enhanced", "ocr_auto_enhanced"} and quality['usable']:
            enhancement_started = perf_counter()
            enhanced = enhance_label(label)
            enhancement_ms = (perf_counter() - enhancement_started) * 1000
        else:
            enhanced = label
            enhancement_ms = 0.0
        if mode in {"image_full", "ocr_full"}:
            work_image = search_image
        elif mode in {"image_auto_enhanced", "ocr_auto_enhanced"}:
            work_image = enhanced
        else:
            work_image = label

        def base_metrics(method: str) -> Dict[str, Any]:
            return {
                "method": method,
                "crop_quality": quality,
                "query_view": 'full_image' if mode in {'image_full', 'ocr_full'} else query_view,
                "bottle_detection": {
                    "count": bottle_result.get('count') if bottle_result else None,
                    "selected_bbox": list(selected_box) if selected_box is not None else None,
                    "selection": selection_method,
                },
                "label_detection": {
                    "bbox": [round(value, 4) for value in display_detection.bbox],
                    "confidence": display_detection.confidence,
                    "contour": [[round(x, 5), round(y, 5)] for x, y in display_detection.contour] if display_detection.contour else None,
                    "method": display_detection.method,
                    "quad": [[round(point[0], 4), round(point[1], 4)] for point in display_detection.quad] if display_detection.quad else None,
                },
                "timings_ms": {
                    "decode": round(decode_ms, 1),
                    "label_detection": round(detection_ms, 1),
                    "bottle_detection": round(bottle_detection_ms, 1),
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

        fusion_ready = (
            self.visual is not None
            and self.fusion is not None
            and mode == "combined"
            and getattr(self.visual, 'secondary_ready', False) is True
            and ocr_enabled is not False
        )
        if fusion_ready:
            region = shelf_region(local_detection.bbox) if query_view == "label" else None
            return self._fused_result(work_image, search_image, base_metrics, include_candidates, compare_slug, region)

        if self.visual:
            visual_started = perf_counter()
            if mode == "combined" and getattr(self.visual, 'secondary_ready', False) is True:
                candidates = self.visual.search_combined(work_image, search_image, limit=8)
            else:
                candidates = self.visual.search(work_image, limit=8)
                if mode == "combined" and work_image is not search_image:
                    candidates = merge_query_views(candidates, self.visual.search(search_image, limit=8), limit=8)
            visual_ms = (perf_counter() - visual_started) * 1000
            if not candidates:
                metrics = base_metrics("siglip2+pgvector")
                metrics["reason"] = "index_empty"
                metrics["timings_ms"]["visual"] = round(visual_ms, 1)
                return {'status': 'unknown', 'recognition': metrics}
            color_features = crop_color_features(work_image)
            threshold = float(os.getenv('CV_MATCH_THRESHOLD', '0.75'))
            min_margin = float(os.getenv('CV_MATCH_MARGIN', '0.04'))
            clear_match = float(os.getenv('CV_CLEAR_MATCH', '0.85'))
            text, ocr_status, text_matches, ocr_ms = "", "skipped", [], 0.0
            explicit_ocr = ocr_enabled
            if ocr_enabled is None:
                ocr_enabled = os.getenv('CV_OCR_ENABLED', 'false').lower() == 'true'
            ocr_allowed = bool(ocr_enabled) and (mode == "combined" or explicit_ocr is True)
            ranked = blend_candidates(candidates, self.catalog, color_features, "", False, self.ocr_references)
            best = ranked[0]
            wine = self.catalog.get(best['slug'])
            runner = self.catalog.get(ranked[1]['slug']) if len(ranked) > 1 else None
            family_tie = same_label_family(wine, runner)
            margin = distinct_margin(ranked, self.catalog)
            matched = bool(wine) and is_visual_match(
                best['score'],
                margin,
                threshold=threshold,
                min_margin=min_margin,
                clear_match=clear_match,
                family_tie=family_tie,
            )
            # A high embedding score cannot validate a failed label detection.
            if mode != "image_full" and not quality['usable']:
                matched = False
            # A close sibling (Rubin vs Pinot on one series) is already a visual
            # winner, so the grape word on the label would otherwise never be read.
            sibling_gap = float(best["score"]) - float(ranked[1]["score"]) if len(ranked) > 1 else 1.0
            close_sibling = family_tie and sibling_gap <= 0.06
            run_ocr = ocr_allowed and (not matched or close_sibling)
            corroborated = False
            if run_ocr:
                ocr_started = perf_counter()
                if close_sibling:
                    text, ocr_status, lines = self._read_sibling(work_image)
                else:
                    text, ocr_status = self._ocr(self._image_bytes(work_image), psm=6)
                    lines = []
                ocr_ms = (perf_counter() - ocr_started) * 1000
                text_matches = self._text_matches(text)
                if matched and close_sibling:
                    # The visual winner is already a card. OCR may name the
                    # sibling grape, and must not reshuffle the rest of the list.
                    ranked = prefer_distinctive_label(ranked, self.catalog, text)
                    ranked = self._prefer_grape_line(ranked, work_image, lines)
                else:
                    ranked = blend_candidates(candidates, self.catalog, color_features, text, True, self.ocr_references)
                    ranked = prefer_distinctive_label(ranked, self.catalog, text)
                best = ranked[0]
                wine = self.catalog.get(best['slug'])
                runner = self.catalog.get(ranked[1]['slug']) if len(ranked) > 1 else None
                family_tie = same_label_family(wine, runner)
                margin = distinct_margin(ranked, self.catalog)
                corroborated = bool(
                    text_matches
                    and wine
                    and text_matches[0].wine.slug == best['slug']
                    and text_matches[0].score >= self.MATCH_THRESHOLD
                    and self._text_evidence(text_matches[0])
                )
                matched = corroborated or (
                    bool(wine) and is_visual_match(
                        best['score'],
                        margin,
                        threshold=threshold,
                        min_margin=min_margin,
                        clear_match=clear_match,
                        family_tie=family_tie,
                        corroborated=corroborated,
                    )
                )
                if mode != "image_full" and not quality['usable'] and not corroborated:
                    matched = False
            method = "siglip2+pgvector+ocr" if run_ocr else "siglip2+pgvector"
            metrics = base_metrics(method)
            metrics.update(
                similarity=round(best['score'], 4),
                siglip=best.get('siglip'),
                color_delta=best.get('color_delta'),
                ocr_delta=best.get('ocr_delta'),
                color=color_features,
                crop_jpeg_base64=self._preview_jpeg(work_image),
                compared_with=self._compare_views(ranked[:2], with_catalog_crops=include_candidates),
                margin=round(margin, 4),
                family_tie=family_tie,
                ocr=ocr_status,
                ocr_characters=len(text),
                ocr_best={'slug': text_matches[0].wine.slug, 'score': round(text_matches[0].score, 4)} if text_matches else None,
                ocr_corroborated=corroborated,
                ocr_text=text[:500],
                ocr_enabled=bool(ocr_enabled),
                color_enabled=False,
                ocr_reference_count=len(self.ocr_references),
                ocr_comparison='name + winery + precomputed label text',
                threshold=threshold,
                min_margin=min_margin,
                clear_match=clear_match,
            )
            ranking = ranking_metrics(ranked)
            ranking["top5"] = [
                {
                    "slug": item["slug"],
                    "name": (self.catalog.get(item["slug"]).name if self.catalog.get(item["slug"]) else item["slug"]),
                    "winery": (self.catalog.get(item["slug"]).winery if self.catalog.get(item["slug"]) else ""),
                    "score": round(item["score"], 4),
                    "siglip": item.get("siglip"),
                    "color_delta": item.get("color_delta"),
                    "ocr_delta": item.get("ocr_delta"),
                    "ocr_reference_text": item.get("ocr_reference_text") if include_candidates else None,
                    "ocr_catalog_text": item.get("ocr_catalog_text") if include_candidates else None,
                    "image_hash": item.get("image_hash"),
                    "image_url": (self.catalog.get(item["slug"]).image_url if self.catalog.get(item["slug"]) else None),
                    "full_score": item.get("full_score"),
                    "crop_score": item.get("crop_score"),
                    "best_view": item.get("best_view"),
                    "primary_score": item.get("primary_score"),
                    "secondary_score": item.get("secondary_score"),
                }
                for item in ranked[:5]
            ]
            metrics["candidates"] = ranking["top5"]
            metrics["timings_ms"].update(visual=round(visual_ms, 1), ocr=round(ocr_ms, 1))
            probe = self._probe_slug(work_image, compare_slug, color_features, text, ocr_enabled, include_candidates)
            if probe is not None:
                metrics["probe"] = probe
            status = 'matched' if matched else ('uncertain' if best['score'] >= 0.65 else 'unknown')
            lookalike_items = [
                item for item in ranked
                if not (status == 'matched' and wine and item['slug'] == wine.slug)
            ]
            result = {
                'status': status,
                'confidence': round(best['score'], 4),
                'recognition': metrics,
                'ranking': ranking,
                'lookalikes': self._lookalike_cards(lookalike_items),
            }
            if wine:
                result['slug'] = wine.slug
                result['alternatives'] = recommend_alternatives(self.catalog, wine)
                if status == 'matched':
                    result['wine'] = wine.to_card()
            return result
        if kind == "image" or os.getenv('CV_ENABLED', 'false').lower() == 'true':
            metrics = base_metrics('siglip2')
            metrics['reason'] = 'visual_unavailable'
            return {'status': 'unknown', 'recognition': metrics}
        text, ocr_status = self._ocr(self._image_bytes(work_image), psm=6)
        return self._ocr_result(text, ocr_status, base_metrics('ocr+catalog'), 0.0, include_candidates)

    def _read_words(self, label, frame, region=None):
        from .label_text import lines_on_label, ocr_words
        from .pp_ocr import line_text, read_lines

        started = perf_counter()
        # The frame alone gave the same 157/205 as frame + label crop (the crop
        # alone 144), so a close-up still reads only the frame.
        image = frame
        if max(image.size) > OCR_FRAME_SIDE:
            image = image.copy()
            image.thumbnail((OCR_FRAME_SIDE, OCR_FRAME_SIDE))
        lines = read_lines(image)
        extra, extra_size = [], image.size
        if region is not None:
            kept = lines_on_label(lines, image.size, region)
            if kept:
                lines = kept
            extra, extra_size = _style_band_lines(label)
        words = ocr_words(lines, image.size) + ocr_words(extra, extra_size)
        text = " ".join(part for part in (line_text(lines), line_text(extra)) if part)
        return words, text, (perf_counter() - started) * 1000

    def _fused_result(self, label, frame, base_metrics, include_candidates, compare_slug, region=None):
        """SigLIP and PP-OCR in parallel, then the trained softmax over candidates."""

        from . import fusion

        started = perf_counter()
        reading = _OCR_POOL.submit(self._read_words, label, frame, region)
        state = {}

        def text_candidates():
            words, text, ocr_ms = reading.result()
            state.update(
                words=words, text=text, ocr_ms=ocr_ms,
                card=self.text_index.score(words),
                photo=self.photo_index.score(words) if self.photo_index is not None else None,
            )
            extra = state['card'].top(self.fusion['text_n'])
            if state['photo'] is not None:
                extra += state['photo'].top(self.fusion['text_n'])
            return extra

        rows = self.visual.search_combined(label, frame, limit=self.fusion['visual_k'], extra_slugs=text_candidates)
        if 'card' not in state:
            text_candidates()
        visual_ms = (perf_counter() - started) * 1000
        if not rows:
            metrics = base_metrics('siglip2+pgvector')
            metrics['reason'] = 'index_empty'
            return {'status': 'unknown', 'recognition': metrics}
        visual = {row['slug']: float(row['score']) for row in rows}
        details = {row['slug']: row for row in rows}
        ranked = []
        for item in fusion.rank(visual, state['card'], self.fusion, state['photo']):
            row = details.get(item['slug'], {})
            ranked.append({
                **item,
                'score': item['probability'],
                'siglip': round(item['visual'], 4),
                'image_hash': row.get('image_hash'),
                'full_score': row.get('full_score'),
                'crop_score': row.get('crop_score'),
                'primary_score': row.get('primary_score'),
                'secondary_score': row.get('secondary_score'),
            })
        best = ranked[0]
        wine = self.catalog.get(best['slug'])
        # Product decision: a leader with 40% or more opens its card. Cross-validation
        # at 0.5 gave a card on 72% of photos with 88% of them right; 0.4 trades a
        # little more precision for fewer «not found» screens.
        match_at = float(os.getenv('CV_FUSION_MATCH', '0.4'))
        # A clear picture lead also earns a card: p >= 0.6 with the top wine 0.06
        # above every other candidate by SigLIP. Cross-validated: card on 46% of
        # photos instead of 42%, precision 96.3% instead of 97.1%.
        visual_lead = best['visual'] - max((item['visual'] for item in ranked[1:]), default=0.0)
        clear_picture = best['probability'] >= LEAD_MATCH and visual_lead >= VISUAL_LEAD
        runner_probability = ranked[1]['probability'] if len(ranked) > 1 else 0.0
        # A clear text leader too: «МУСКАТЕЛЬ … БЕЛЫЙ» names one wine even when
        # the catalog photo looks different. p >= 0.4, text 25% above every other
        # candidate, 20% of its catalog identity read. Cross-validated: card on
        # 49% of photos instead of 46%, precision 95.5% instead of 96.3%.
        other_text = max((item['text_score'] for item in ranked[1:]), default=0.0)
        text_lead = (best['text_score'] - other_text) / best['text_score'] if best['text_score'] > 0 else 0.0
        clear_text = (
            best['probability'] >= TEXT_LEAD_MATCH
            and text_lead >= TEXT_LEAD
            and best['text_coverage'] >= TEXT_COVERAGE
        )
        if wine and (best['probability'] >= match_at or clear_picture or clear_text or share_leads(best['probability'], runner_probability)):
            status = 'matched'
        elif wine and (best['probability'] >= 0.35 or best['visual'] >= 0.65):
            status = 'uncertain'
        else:
            status = 'unknown'
        metrics = base_metrics('siglip2+ppocr+fusion')
        metrics.update(
            similarity=round(best['visual'], 4),
            siglip=round(best['visual'], 4),
            probability=round(best['probability'], 4),
            crop_jpeg_base64=self._preview_jpeg(label),
            compared_with=self._compare_views(ranked[:2], with_catalog_crops=include_candidates),
            margin=round(best['probability'] - (ranked[1]['probability'] if len(ranked) > 1 else 0.0), 4),
            ocr='ppocr',
            ocr_characters=len(state['text']),
            ocr_text=state['text'][:500],
            ocr_enabled=True,
            text_matched=best['text_matched'],
            match_probability=match_at,
            visual_lead=round(visual_lead, 4),
            text_lead=round(text_lead, 4),
            text_coverage=round(best['text_coverage'], 4),
        )
        metrics['timings_ms'].update(visual=round(visual_ms, 1), ocr=round(state['ocr_ms'], 1))
        ranking = ranking_metrics(ranked)
        ranking['top5'] = [
            {
                'slug': item['slug'],
                'name': self.catalog.get(item['slug']).name if self.catalog.get(item['slug']) else item['slug'],
                'winery': self.catalog.get(item['slug']).winery if self.catalog.get(item['slug']) else '',
                'score': round(item['probability'], 4),
                'probability': round(item['probability'], 4),
                'siglip': item['siglip'],
                'text_score': item['text_score'],
                'text_matched': item['text_matched'],
                'image_hash': item.get('image_hash'),
                'image_url': self.catalog.get(item['slug']).image_url if self.catalog.get(item['slug']) else None,
                'full_score': item.get('full_score'),
                'crop_score': item.get('crop_score'),
                'primary_score': item.get('primary_score'),
                'secondary_score': item.get('secondary_score'),
            }
            for item in ranked[:5]
        ]
        metrics['candidates'] = ranking['top5']
        probe = self._probe_slug(label, compare_slug, crop_color_features(label), state['text'], True, include_candidates)
        if probe is not None:
            metrics['probe'] = probe
        result = {
            'status': status,
            'confidence': round(best['probability'], 4),
            'recognition': metrics,
            'ranking': ranking,
            'lookalikes': self._lookalike_cards(
                item for item in ranked if not (status == 'matched' and item['slug'] == best['slug'])
            ),
        }
        if wine:
            result['slug'] = wine.slug
            result['alternatives'] = recommend_alternatives(self.catalog, wine)
            if status == 'matched':
                result['wine'] = wine.to_card()
        return result

    def _lookalike_cards(self, items: Iterable[Any], limit: int = 5) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            if isinstance(item, dict):
                slug = item.get('slug')
                score = item.get('score')
            elif hasattr(item, 'wine'):
                slug = item.wine.slug
                score = getattr(item, 'score', None)
            else:
                slug = item
                score = None
            if not slug or slug in seen:
                continue
            wine = self.catalog.get(slug)
            if not wine:
                continue
            card = wine.to_card()
            if score is not None:
                card['label_score'] = round(float(score), 4)
            cards.append(card)
            seen.add(slug)
            if len(cards) >= limit:
                break
        return cards

    def _probe_slug(self, work_image, slug, color_features, text, ocr_enabled, include_candidates):
        if not slug or not include_candidates or not self.visual:
            return None
        wine = self.catalog.get(slug)
        if not wine:
            return {"slug": slug, "missing": True}
        scored = self.visual.score_slug(work_image, slug)
        if not scored:
            return {"slug": slug, "name": wine.name, "winery": wine.winery, "image_url": wine.image_url, "missing": True}
        blended = blend_candidates([scored], self.catalog, color_features, text, ocr_enabled, self.ocr_references)
        views = self._compare_views(blended, with_catalog_crops=True)
        return views[0] if views else None

    def _compare_views(self, items: Iterable[Any], with_catalog_crops: bool = False) -> List[Dict[str, Any]]:
        views: List[Dict[str, Any]] = []
        for item in items:
            wine = self.catalog.get(item.get("slug")) if isinstance(item, dict) else None
            if not wine:
                continue
            view = {
                "slug": wine.slug,
                "name": wine.name,
                "winery": wine.winery,
                "image_url": wine.image_url,
                "label_jpeg_base64": self._catalog_label_preview(wine) if with_catalog_crops else None,
                "indexed_views": ["full", "crop"],
                "image_hash": item.get("image_hash"),
                "score": round(float(item.get("score") or 0), 4),
                "siglip": item.get("siglip"),
                "full_score": item.get("full_score"),
                "crop_score": item.get("crop_score"),
                "best_view": item.get("best_view"),
            }
            views.append(view)
        return views

    def _catalog_label_preview(self, wine) -> Optional[str]:
        cached = self._catalog_crop_cache.get(wine.slug, "")
        if cached != "":
            return cached
        preview = None
        try:
            stored = self._load_stored_label_crop(wine)
            if stored is not None:
                preview = self._preview_jpeg(stored)
            else:
                image = self._load_catalog_image(wine)
                if image is not None:
                    preview = self._preview_jpeg(crop_label(image, detect_label(image, catalog=True)))
        except Exception:
            logging.exception("Could not build catalog label preview for %s", wine.slug)
        self._catalog_crop_cache[wine.slug] = preview
        return preview

    def _load_stored_label_crop(self, wine):
        from pathlib import Path

        from PIL import Image, ImageOps

        folder = Path(os.getenv("CATALOG_LABEL_CROPS", "/catalog/label-crops"))
        path = folder / f"{wine.slug}.jpg"
        if not path.is_file():
            return None
        with Image.open(path) as opened:
            return label_rgb(opened)

    def _catalog_file_names(self, wine) -> List[str]:
        from urllib.parse import unquote, urlparse
        from pathlib import Path

        names = []
        image_name = str(getattr(wine, "image_name", "") or "").strip()
        if image_name:
            names.append(Path(image_name).name)
        url = str(getattr(wine, "image_url", "") or getattr(wine, "direct_image_url", "") or "")
        if url:
            names.append(unquote(Path(urlparse(url).path).name))
        seen = set()
        unique = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _load_catalog_image(self, wine):
        from pathlib import Path
        from urllib.request import Request, urlopen

        from PIL import Image, ImageOps

        names = self._catalog_file_names(wine)
        folders = []
        extra = os.getenv("CATALOG_IMAGES", "").strip()
        if extra:
            folders.append(Path(extra))
        catalog = os.getenv("CATALOG_CSV", "").strip()
        if catalog:
            parent = Path(catalog).parent
            folders.extend((parent, parent / "uploads", parent / "prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"))
        for folder in folders:
            if not folder.is_dir():
                continue
            for name in names:
                path = folder / name
                if path.is_file():
                    with Image.open(path) as opened:
                        return label_rgb(opened)
        root = Path("/catalog")
        if names and root.is_dir():
            for name in names:
                match = next(root.rglob(name), None)
                if match and match.is_file():
                    with Image.open(match) as opened:
                        return label_rgb(opened)
        url = wine.image_url
        if not url or not url.startswith("http"):
            return None
        request = Request(url, headers={"User-Agent": "skanervina-debug/1"})
        with urlopen(request, timeout=3) as response:
            payload = response.read(self.settings.max_image_bytes + 1)
        if not payload or len(payload) > self.settings.max_image_bytes:
            return None
        with Image.open(BytesIO(payload)) as opened:
            return label_rgb(opened)

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
        lookalikes = self._lookalike_cards(matches)
        alternatives = recommend_alternatives(self.catalog, best.wine)
        if best.score >= self.MATCH_THRESHOLD and self._text_evidence(best):
            return {
                "status": "matched",
                "slug": best.wine.slug,
                "wine": best.wine.to_card(),
                "confidence": round(best.score, 3),
                "recognition": metrics,
                "lookalikes": lookalikes,
                "alternatives": alternatives,
            }
        if best.score >= self.UNCERTAIN_THRESHOLD:
            return {
                "status": "uncertain",
                "slug": best.wine.slug,
                "confidence": round(best.score, 3),
                "recognition": {**metrics, "reason": "low_confidence"},
                "lookalikes": lookalikes,
                "alternatives": alternatives,
            }
        return {
            "status": "unknown",
            "confidence": round(best.score, 3),
            "recognition": {**metrics, "reason": "no_confident_match"},
            "lookalikes": lookalikes,
            "alternatives": alternatives,
        }
