import unittest

from app.bottle_detection import rank_bottle_candidates, selection_is_ambiguous


class BottleDetectionTests(unittest.TestCase):
    def test_ranks_front_bottle_over_small_background_objects(self):
        candidates = rank_bottle_candidates(
            [
                {"label": "bottle", "score": 0.91, "box": [340, 70, 650, 970]},
                {"label": "bottle", "score": 0.88, "box": [760, 240, 850, 690]},
                {"label": "book", "score": 0.99, "box": [100, 100, 350, 900]},
            ],
            (1000, 1000),
        )
        self.assertEqual(len(candidates), 2)
        self.assertGreater(candidates[0]["bbox"][0], 0.3)
        self.assertGreater(candidates[0]["priority"], candidates[1]["priority"])
        self.assertFalse(selection_is_ambiguous(candidates))

    def test_requests_web_choice_for_two_similarly_prominent_bottles(self):
        candidates = rank_bottle_candidates(
            [
                {"label": "bottle", "score": 0.90, "box": [180, 100, 430, 920]},
                {"label": "bottle", "score": 0.89, "box": [540, 100, 790, 920]},
            ],
            (1000, 1000),
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(selection_is_ambiguous(candidates))

    def test_ignores_low_confidence_and_non_bottle_detections(self):
        candidates = rank_bottle_candidates(
            [
                {"label": "bottle", "score": 0.10, "box": [200, 100, 450, 900]},
                {"label": "book", "score": 0.99, "box": [500, 100, 750, 900]},
            ],
            (1000, 1000),
        )
        self.assertEqual(candidates, [])
        self.assertFalse(selection_is_ambiguous(candidates))


if __name__ == "__main__":
    unittest.main()
