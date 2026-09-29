import unittest

from app.catalog import WineCatalog
from app.fusion import FEATURES, rank
from app.label_text import TextIndex, Word, catalog_documents, lines_on_label, raw_words, reference_documents, style_of

ROWS = [
    {"Slug": "loco-rkatsiteli", "Название вина": "Ркацители", "Винодельня": "Loco Cimbali",
     "Категория": "Оранжевое сухое", "Сорт винограда": "Ркацители"},
    {"Slug": "loco-shardone", "Название вина": "Шардоне", "Винодельня": "Loco Cimbali",
     "Категория": "Белое сухое", "Сорт винограда": "Шардоне"},
    {"Slug": "sober-krasnostop", "Название вина": "Красностоп", "Винодельня": "Собер Баш",
     "Категория": "Красное сухое", "Сорт винограда": "Красностоп Золотовский"},
    {"Slug": "sober-fran", "Название вина": "Каберне Фран", "Винодельня": "Собер Баш",
     "Категория": "Красное сухое", "Сорт винограда": "Каберне Фран"},
    {"Slug": "perovskih-polusladkoe", "Название вина": "Полусладкое Красное", "Винодельня": "Усадьба Перовских",
     "Категория": "Красное полусладкое", "Сорт винограда": "Мерло"},
    {"Slug": "perovskih-polusuhoe", "Название вина": "Полусухое Красное", "Винодельня": "Усадьба Перовских",
     "Категория": "Красное полусухое", "Сорт винограда": "Мерло"},
]


def words(*texts):
    return [Word(text, 1.0) for text in texts]


class LabelTextTests(unittest.TestCase):
    def setUp(self):
        wines = list(WineCatalog.from_rows(ROWS, "https://example.com/"))
        self.index = TextIndex(catalog_documents(wines), wines)

    def test_ocr_digits_and_lookalikes_are_repaired(self):
        self.assertEqual(raw_words("УРОЖАЙ 2о24"), ["урожай", "2024"])
        self.assertEqual(style_of("CYXOE KPACHOE"), {"sweetness": "dry", "colour": "red"})

    def test_latin_grape_on_label_finds_cyrillic_catalog_wine(self):
        text = self.index.score(words("LOCO", "CIMBALI", "RKATSITELI"))
        self.assertEqual(text.top(1), ["loco-rkatsiteli"])

    def test_latin_lookalike_capitals_are_read_as_cyrillic(self):
        text = self.index.score(words("СОБЕР", "БАШ", "KPACHOCTOP"))
        self.assertEqual(text.top(1), ["sober-krasnostop"])

    def test_one_typo_still_matches(self):
        text = self.index.score(words("Собер Баш", "Красностоб"))
        self.assertEqual(text.top(1), ["sober-krasnostop"])

    def test_sweetness_is_style_evidence_not_name_score(self):
        text = self.index.score(words("Усадьба Перовских", "ПОЛУСЛАДКОЕ"))
        sweet, dry = text.evidence("perovskih-polusladkoe"), text.evidence("perovskih-polusuhoe")
        self.assertEqual(sweet.score, dry.score)
        self.assertEqual((sweet.sweetness, dry.sweetness), (1, -1))

    def test_rival_of_same_winery_explaining_the_grape_is_a_contradiction(self):
        text = self.index.score(words("Собер Баш", "Красностоп"))
        rivals = ["sober-krasnostop", "sober-fran"]
        self.assertGreater(text.evidence("sober-fran", rivals).contradiction, 0)
        self.assertEqual(text.evidence("sober-krasnostop", rivals).contradiction, 0)

    def test_catalog_photo_text_is_its_own_index(self):
        photo = TextIndex(reference_documents([
            {"slug": "loco-shardone", "size": [100, 100],
             "lines": [{"text": "CUVÉE PREMIUM", "score": 0.9, "box": [[30, 40], [70, 40], [70, 50], [30, 50]]}]},
        ]))
        self.assertEqual(photo.score(words("premium")).top(1), ["loco-shardone"])


class FusionTests(unittest.TestCase):
    def setUp(self):
        wines = list(WineCatalog.from_rows(ROWS, "https://example.com/"))
        self.index = TextIndex(catalog_documents(wines), wines)
        weights = [0.0] * len(FEATURES)
        weights[FEATURES.index("visual_gap")] = 30.0
        weights[FEATURES.index("coverage")] = 9.0
        weights[FEATURES.index("text_share")] = 1.5
        self.weights = {"features": list(FEATURES), "weights": weights, "visual_k": 20, "text_n": 10}

    def test_label_grape_lifts_close_sibling(self):
        visual = {"loco-shardone": 0.84, "loco-rkatsiteli": 0.80, "sober-fran": 0.60}
        text = self.index.score(words("LOCO CIMBALI", "RKATSITELI"))
        self.assertEqual(rank(visual, text, self.weights)[0]["slug"], "loco-rkatsiteli")

    def test_sibling_named_on_label_overrides_a_confident_ranker(self):
        self.weights["weights"][FEATURES.index("text_share")] = 0.0
        self.weights["weights"][FEATURES.index("coverage")] = 0.0
        visual = {"sober-fran": 0.90, "sober-krasnostop": 0.87}
        ranked = rank(visual, self.index.score(words("Собер Баш", "Красностоп")), self.weights)
        self.assertEqual(ranked[0]["slug"], "sober-krasnostop")
        self.assertTrue(ranked[0]["sibling_override"])

    def test_without_text_visual_order_stays(self):
        visual = {"loco-shardone": 0.84, "loco-rkatsiteli": 0.80}
        ranked = rank(visual, self.index.score([]), self.weights)
        self.assertEqual([item["slug"] for item in ranked], ["loco-shardone", "loco-rkatsiteli"])
        self.assertAlmostEqual(sum(item["probability"] for item in ranked), 1.0)


class ShelfLineTests(unittest.TestCase):
    def test_neighbor_name_stays_off_the_center_label(self):
        def line(text, x, y):
            return {"text": text, "score": 0.9, "box": [[x, y], [x + 8, y], [x + 8, y + 4], [x, y + 4]]}

        # Box is the center pink panel. «ТО РУЖ» is the bottle on the left.
        kept = lines_on_label(
            [
                line("ТО РУЖ", 14, 71),
                line("BYCCO", 47, 72),
                line("INKERMAN", 47, 46),
                line("INKERMAN", 48, 11),
                line("ERMAN", 10, 46),
            ],
            (100, 100),
            (0.318, 0.575, 0.637, 0.839),
        )
        self.assertEqual([item["text"] for item in kept], ["BYCCO", "INKERMAN"])


class ShareLeadTests(unittest.TestCase):
    def test_half_with_a_ten_point_gap_opens_the_card(self):
        from app.recognition import share_leads

        self.assertTrue(share_leads(0.575, 0.194))
        self.assertTrue(share_leads(0.50, 0.40))

    def test_a_narrow_lead_stays_a_list(self):
        from app.recognition import share_leads

        self.assertFalse(share_leads(0.52, 0.45))
        self.assertFalse(share_leads(0.49, 0.20))


if __name__ == "__main__":
    unittest.main()
