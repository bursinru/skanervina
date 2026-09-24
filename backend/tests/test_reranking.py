import unittest

import numpy as np
from PIL import Image, ImageDraw

from app.catalog import WineCatalog
from app.label_signals import family_tiebreak
from app.local_features import geometric_confirmed, rerank


def _catalog():
    return WineCatalog.from_rows(
        [
            {"Slug": "velvet-dry", "Название вина": "Вельвет Сизон Рислинг сухое", "Винодельня": "Фанагория",
             "Сорт винограда": "Рислинг"},
            {"Slug": "velvet-sweet", "Название вина": "Вельвет Сизон Рислинг полусладкое", "Винодельня": "Фанагория",
             "Сорт винограда": "Рислинг"},
            {"Slug": "other", "Название вина": "Мускатель", "Винодельня": "Массандра", "Сорт винограда": "Мускат"},
        ],
        "https://example.com/",
    )


class FamilyTiebreakTests(unittest.TestCase):
    def test_sweetness_word_picks_sibling(self):
        rows = [{"slug": "velvet-dry", "score": 0.81}, {"slug": "velvet-sweet", "score": 0.79}]
        ranked = family_tiebreak(rows, _catalog(), "ФАНАГОРИЯ Вельвет Сизон ПОЛУСЛАДКОЕ")
        self.assertEqual(ranked[0]["slug"], "velvet-sweet")

    def test_no_distinguishing_word_keeps_order(self):
        rows = [{"slug": "velvet-dry", "score": 0.81}, {"slug": "velvet-sweet", "score": 0.79}]
        ranked = family_tiebreak(rows, _catalog(), "ФАНАГОРИЯ Вельвет Сизон Рислинг")
        self.assertEqual(ranked[0]["slug"], "velvet-dry")

    def test_other_producer_is_never_promoted(self):
        rows = [{"slug": "velvet-dry", "score": 0.81}, {"slug": "other", "score": 0.80}]
        ranked = family_tiebreak(rows, _catalog(), "Мускатель Массандра")
        self.assertEqual(ranked[0]["slug"], "velvet-dry")

    def test_distant_sibling_is_not_promoted(self):
        rows = [{"slug": "velvet-dry", "score": 0.90}, {"slug": "velvet-sweet", "score": 0.70}]
        ranked = family_tiebreak(rows, _catalog(), "полусладкое")
        self.assertEqual(ranked[0]["slug"], "velvet-dry")


class GeometricRerankTests(unittest.TestCase):
    def test_dominant_inliers_move_candidate_to_top(self):
        rows = [{"slug": "a", "score": 0.80}, {"slug": "b", "score": 0.78}, {"slug": "c", "score": 0.70}]
        ranked = rerank(rows, {"a": 9, "b": 55, "c": 4})
        self.assertEqual(ranked[0]["slug"], "b")
        self.assertTrue(geometric_confirmed(ranked))

    def test_weak_evidence_changes_nothing(self):
        rows = [{"slug": "a", "score": 0.80}, {"slug": "b", "score": 0.78}]
        ranked = rerank(rows, {"a": 5, "b": 9})
        self.assertEqual([row["slug"] for row in ranked], ["a", "b"])
        self.assertFalse(geometric_confirmed(ranked))

    def test_matcher_prefers_same_artwork(self):
        from app.local_features import LocalMatcher

        rng = np.random.default_rng(1)

        def label(seed):
            local = np.random.default_rng(seed)
            image = Image.new("RGB", (400, 300), "white")
            draw = ImageDraw.Draw(image)
            for _ in range(60):
                x, y = local.integers(0, 380, 1)[0], local.integers(0, 280, 1)[0]
                draw.rectangle((x, y, x + local.integers(5, 30), y + local.integers(5, 30)),
                               fill=tuple(int(c) for c in local.integers(0, 255, 3)))
            return image

        target, other = label(7), label(8)
        query = target.rotate(6, fillcolor="white").resize((360, 270))
        query = Image.fromarray(np.clip(np.asarray(query) + rng.normal(0, 4, (270, 360, 3)), 0, 255).astype(np.uint8))
        matcher = LocalMatcher()
        counts = matcher.verify([query], ["target", "other"], lambda slug: [target if slug == "target" else other])
        self.assertGreater(counts["target"], 20)
        self.assertGreater(counts["target"], 3 * max(counts["other"], 1))


if __name__ == "__main__":
    unittest.main()
