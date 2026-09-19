import unittest

from PIL import Image

from app.catalog import WineCatalog
from app.label_signals import color_delta, crop_color_features, wine_tone


class LabelColorTests(unittest.TestCase):
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
