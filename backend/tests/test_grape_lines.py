import os
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from PIL import Image

from app.catalog import WineCatalog
from app.grape_lines import (
    crop_quad,
    in_confusable_series,
    line_for_grapes,
    prefer_by_grape_score,
    select_query_line,
)


def wine(slug, name, grapes, winery="Denisov"):
    catalog = WineCatalog.from_rows(
        [{
            "Slug": slug,
            "Название вина": name,
            "Винодельня": winery,
            "Сорт винограда": grapes,
        }],
        "https://example.com/",
    )
    return catalog.get(slug)


class LookalikeTests(unittest.TestCase):
    def test_rubin_lookalikes_become_cyrillic(self):
        from app.pp_ocr import fold_lookalikes
        from app.catalog import tokens

        self.assertIn("рубин", tokens(fold_lookalikes("Py6iH")))
        self.assertEqual(fold_lookalikes("DENISOV"), "DENISOV")


class GrapeLineTests(unittest.TestCase):
    def test_series_are_the_confusable_labels_only(self):
        rubin = wine("denisov_rubin_klaret_krasnaya_strelka", "Рубин кларет . Красная стрелка", "Рубин Голодриги")
        velvet = wine("fanagoriya-velvet-season-muskat", "Velvet Season Мускат", "Мускат Оттонель", "Фанагория")
        other = wine("usadba-mezyb-shishka-saperavi", "Шишка. Саперави", "Саперави", "Усадьба Мезыбь")
        mezyb = wine("usadba-mezyb-mezyb-kaberne", "Мезыбь Каберне Совиньон", "Каберне Совиньон", "Усадьба Мезыбь")
        self.assertTrue(in_confusable_series(rubin))
        self.assertTrue(in_confusable_series(velvet))
        self.assertTrue(in_confusable_series(mezyb))
        self.assertFalse(in_confusable_series(other))

    def test_grape_line_is_the_one_that_names_the_variety(self):
        lines = [
            {"text": "КРАСНАЯ СТРЕЛКА", "box": [[0, 10], [80, 10], [80, 30], [0, 30]]},
            {"text": "Рубин", "box": [[0, 40], [40, 40], [40, 55], [0, 55]]},
        ]
        chosen = line_for_grapes(lines, {"рубин", "голодриги"})
        self.assertEqual(chosen["text"], "Рубин")

    def test_query_uses_the_line_under_the_shared_title_when_the_word_is_missed(self):
        rubin = wine("denisov_rubin_klaret_krasnaya_strelka", "Рубин кларет . Красная стрелка", "Рубин Голодриги")
        pinot = wine("denisov_pino_noir_klaret", "Пино Нуар кларет. Красная стрелка", "Пино Нуар")
        lines = [
            {"text": "КРАСНАЯ СТРЕЛКА", "box": [[0, 10], [90, 10], [90, 40], [0, 40]]},
            {"text": "????", "box": [[0, 48], [40, 48], [40, 62], [0, 62]]},
        ]
        box = select_query_line(lines, [pinot, rubin])
        self.assertEqual(box, lines[1]["box"])

    def test_a_read_grape_word_selects_that_line(self):
        rubin = wine("denisov_rubin_klaret_krasnaya_strelka", "Рубин кларет . Красная стрелка", "Рубин Голодриги")
        pinot = wine("denisov_pino_noir_klaret", "Пино Нуар кларет. Красная стрелка", "Пино Нуар")
        lines = [
            {"text": "КРАСНАЯ СТРЕЛКА", "box": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"text": "Рубин", "box": [[0, 20], [10, 20], [10, 30], [0, 30]]},
        ]
        self.assertEqual(select_query_line(lines, [pinot, rubin]), lines[1]["box"])

    def test_closer_grape_line_passes_the_visual_leader(self):
        ranked = prefer_by_grape_score(
            [
                {"slug": "pinot", "score": 0.84, "siglip": 0.84},
                {"slug": "rubin", "score": 0.81, "siglip": 0.81},
            ],
            {"pinot": 0.42, "rubin": 0.61},
        )
        self.assertEqual(ranked[0]["slug"], "rubin")

    def test_a_weak_grape_line_keeps_the_visual_leader(self):
        ranked = prefer_by_grape_score(
            [
                {"slug": "pinot", "score": 0.84, "siglip": 0.84},
                {"slug": "rubin", "score": 0.81, "siglip": 0.81},
            ],
            {"pinot": 0.20, "rubin": 0.28},
        )
        self.assertEqual(ranked[0]["slug"], "pinot")

    def test_crop_keeps_the_word_and_a_little_paper(self):
        image = Image.new("RGB", (100, 80), "white")
        crop = crop_quad(image, [[10, 20], [50, 20], [50, 40], [10, 40]])
        self.assertGreater(crop.width, 40)
        self.assertGreater(crop.height, 20)


class SiblingReadTests(unittest.TestCase):
    def test_a_read_rubin_passes_the_closer_pinot(self):
        from app.label_detection import LabelDetection
        from app.recognition import Recognizer
        from app.settings import settings

        catalog = WineCatalog.from_rows(
            [
                {
                    "Slug": "pinot",
                    "Название вина": "Пино Нуар кларет. Красная стрелка",
                    "Винодельня": "Denisov Winery",
                    "Сорт винограда": "Пино Нуар",
                },
                {
                    "Slug": "rubin",
                    "Название вина": "Рубин кларет . Красная стрелка",
                    "Винодельня": "Denisov Winery",
                    "Сорт винограда": "Рубин Голодриги",
                },
            ],
            "https://example.com/",
        )
        with patch.dict(os.environ, {"CV_ENABLED": "false"}):
            recognizer = Recognizer(catalog, settings)
        recognizer.visual = Mock()
        recognizer.visual.score_grape_views = Mock(return_value={})
        recognizer.visual.search.return_value = [
            {"slug": "pinot", "score": 0.8457},
            {"slug": "rubin", "score": 0.8016},
        ]
        recognizer._read_sibling = Mock(return_value=(
            "Красная стрелка Рубин",
            "ppocr",
            [{"text": "Рубин", "box": [[0, 20], [30, 20], [30, 36], [0, 36]]}],
        ))
        output = BytesIO()
        Image.new("RGB", (80, 120), "white").save(output, format="PNG")
        with patch("app.recognition.detect_label", return_value=LabelDetection((0.2, 0.2, 0.8, 0.8), 0.8)):
            result = recognizer.recognize(output.getvalue(), ocr_enabled=True)
        self.assertEqual(result["slug"], "rubin")
        recognizer._read_sibling.assert_called_once()
