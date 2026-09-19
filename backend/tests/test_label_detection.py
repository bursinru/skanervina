import unittest
from pathlib import Path

from PIL import Image

from app.label_detection import crop_label, detect_label, enhance_label


class LabelDetectionTests(unittest.TestCase):
    QUERIES = Path(__file__).parents[2] / "Датасет" / "eval" / "queries"

    def test_detector_returns_a_valid_conservative_box(self):
        for filename in ("019c68d0.jpg", "02eef911.webp", "096ca74e.jpg"):
            with self.subTest(filename=filename):
                image = Image.open(self.QUERIES / filename)
                detection = detect_label(image)
                left, top, right, bottom = detection.bbox
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, 1)
                self.assertLessEqual(bottom, 1)
                self.assertGreater(right - left, 0.16)
                self.assertGreater(bottom - top, 0.22)
                self.assertGreater(detection.confidence, 0)
                self.assertLess((right - left) * (bottom - top), 0.85)

    def test_crop_preserves_image_and_enhancement_geometry(self):
        image = Image.open(self.QUERIES / "096ca74e.jpg")
        detection = detect_label(image)
        cropped = crop_label(image, detection)
        enhanced = enhance_label(cropped)
        self.assertGreater(cropped.width, 32)
        self.assertGreater(cropped.height, 32)
        self.assertEqual(cropped.size, enhanced.size)
        self.assertEqual(cropped.mode, "RGB")

    def test_detector_prefers_portrait_paper_over_full_frame(self):
        image = Image.new("RGB", (400, 640), (18, 16, 14))
        for x in range(145, 255):
            for y in range(150, 470):
                image.putpixel((x, y), (236, 224, 196))
        detection = detect_label(image)
        left, top, right, bottom = detection.bbox
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        self.assertGreater(cx, 0.32)
        self.assertLess(cx, 0.68)
        self.assertGreater(cy, 0.28)
        self.assertLess(cy, 0.72)
        self.assertLess(right - left, 0.62)
        self.assertGreater(bottom - top, right - left)


if __name__ == "__main__":
    unittest.main()
