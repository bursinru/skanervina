import unittest

from PIL import Image

from app.catalog import WineCatalog
from app.label_signals import color_delta, crop_color_features, wine_tone


class LabelColorTests(unittest.TestCase):
    def test_ocr_uses_reference_label_and_tolerates_one_missing_letter(self):
        from app.label_signals import ocr_delta
        wine = WineCatalog.from_rows([{'Slug': 'test', 'Название вина': 'Рислинг',
            'Винодельня': 'Другое имя'}], 'https://example.com/').get('test')
        self.assertEqual(ocr_delta('КАЛИСТЫЙ БЕРЕГ 2024 Рислинг', wine), 0)
        self.assertGreater(ocr_delta('КАЛИСТЫЙ БЕРЕГ 2024 Рислинг', wine,
                                    'СКАЛИСТЫЙ БЕРЕГ Рислинг'), 0)

    def test_grape_and_year_alone_do_not_identify_wine(self):
        from app.label_signals import ocr_delta
        wine = WineCatalog.from_rows([{'Slug': 'test', 'Название вина': 'Рислинг 2024',
            'Винодельня': 'Скалистый берег'}], 'https://example.com/').get('test')
        self.assertEqual(ocr_delta('Рислинг 2024', wine), 0)

    def setUp(self):
        self.catalog = WineCatalog.from_rows(
            [
                {
                    "Slug": "orange",
                    "Название вина": "Оранж",
                    "Винодельня": "Loco Cimbali",
                    "svoe_vino_color": "Оранжевое сухое",
                },
                {
                    "Slug": "red",
                    "Название вина": "Пино Менье",
                    "Винодельня": "Loco Cimbali",
                    "svoe_vino_color": "Красное сухое",
                },
            ],
            "https://example.com/",
        )

    def test_cream_label_with_orange_print_is_not_red_wine(self):
        image = Image.new("RGB", (160, 220), (236, 226, 208))
        for x in range(40, 120):
            for y in range(70, 170):
                image.putpixel((x, y), (210, 92, 42))
        features = crop_color_features(image)
        self.assertNotEqual(features["bottle_tone"], "red")
        orange = color_delta(features, self.catalog.get("orange"))
        red = color_delta(features, self.catalog.get("red"))
        self.assertGreaterEqual(orange, 0.0)
        self.assertGreaterEqual(orange, red)

    def test_orange_print_prefers_orange_wine_over_white_sibling(self):
        from app.label_signals import blend_candidates, crop_color_features

        catalog = WineCatalog.from_rows(
            [
                {"Slug": "orange", "Название вина": "Оранж Мускат", "Винодельня": "Loco Cimbali", "svoe_vino_color": "Оранжевое сухое"},
                {"Slug": "white", "Название вина": "Оранж Мускат белый", "Винодельня": "Loco Cimbali", "svoe_vino_color": "Белое сухое"},
            ],
            "https://example.com/",
        )
        image = Image.new("RGB", (160, 180), (236, 226, 208))
        for x in range(70, 150):
            for y in range(40, 160):
                image.putpixel((x, y), (196, 92, 42))
        ranked = blend_candidates(
            [
                {"slug": "white", "score": 0.916, "crop_score": 0.916},
                {"slug": "orange", "score": 0.834, "crop_score": 0.834},
            ],
            catalog,
            crop_color_features(image),
            "",
            False,
        )
        self.assertEqual(ranked[0]["slug"], "orange")

    def test_orange_name_is_orange_tone(self):
        self.assertEqual(wine_tone(self.catalog.get("orange")), "orange")

    def test_color_does_not_bury_label_leader(self):
        from app.label_signals import blend_candidates

        features = {"bottle_tone": "red", "paper": "mixed"}
        ranked = blend_candidates(
            [
                {"slug": "red", "score": 0.70, "crop_score": 0.70, "full_score": 0.64},
                {"slug": "orange", "score": 0.73, "crop_score": 0.73, "full_score": 0.76},
            ],
            self.catalog,
            features,
            "",
            False,
        )
        self.assertEqual(ranked[0]["slug"], "orange")
        self.assertGreaterEqual(ranked[0]["color_delta"], 0.0)
        self.assertGreaterEqual(ranked[1]["color_delta"], 0.0)

    def test_ocr_can_keep_a_near_sibling_over_visual_lock(self):
        from app.label_signals import blend_candidates

        catalog = WineCatalog.from_rows(
            [
                {"Slug": "premium", "Название вина": "Каберне Фран Премиум", "Винодельня": "Шато де Талю"},
                {"Slug": "reserve", "Название вина": "Каберне Фран Резерв", "Винодельня": "Шато де Талю"},
            ],
            "https://example.com/",
        )
        ranked = blend_candidates(
            [
                {"slug": "reserve", "score": 0.857, "crop_score": 0.857},
                {"slug": "premium", "score": 0.824, "crop_score": 0.824},
            ],
            catalog,
            {},
            "Каберне Фран Премиум",
            True,
        )
        self.assertEqual(ranked[0]["slug"], "premium")

    def test_bottle_match_beats_a_stronger_wrong_label_crop(self):
        from app.label_signals import blend_candidates

        catalog = WineCatalog.from_rows(
            [
                {"Slug": "relicta", "Название вина": "Реликта", "Винодельня": "Реликта"},
                {"Slug": "flamingo", "Название вина": "Фламинго", "Винодельня": "Николаев и сыновья"},
            ],
            "https://example.com/",
        )
        ranked = blend_candidates(
            [
                {"slug": "relicta", "score": 0.829, "crop_score": 0.829, "full_score": 0.701},
                {"slug": "flamingo", "score": 0.914, "crop_score": 0.597, "full_score": 0.914},
            ],
            catalog,
            {},
            "",
            False,
        )
        self.assertEqual(ranked[0]["slug"], "flamingo")
