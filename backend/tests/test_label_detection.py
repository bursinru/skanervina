import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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

    def test_keeps_full_paper_instead_of_the_drawing(self):
        image = Image.new("RGB", (400, 640), (10, 10, 12))
        for x in range(132, 268):
            for y in range(148, 508):
                image.putpixel((x, y), (236, 224, 196))
        for x in range(168, 244):
            for y in range(318, 468):
                image.putpixel((x, y), (36, 32, 28))
        left, top, right, bottom = detect_label(image).bbox
        self.assertLess(left, 0.34)
        self.assertGreater(right, 0.66)
        self.assertLess(top, 0.28)
        self.assertGreater(bottom, 0.50)

    def test_left_bottle_is_not_pulled_into_black_space(self):
        image = Image.new("RGB", (640, 480), (6, 6, 8))
        for x in range(70, 210):
            for y in range(70, 410):
                image.putpixel((x, y), (238, 228, 208))
        for x in range(110, 175):
            for y in range(210, 360):
                image.putpixel((x, y), (32, 30, 26))
        left, top, right, bottom = detect_label(image).bbox
        self.assertLess(left, 0.16)
        self.assertLess(right, 0.48)
        self.assertGreater(right, 0.28)
        self.assertGreater(bottom - top, right - left)

    def test_trims_black_glass_on_the_right(self):
        image = Image.new("RGB", (640, 400), (5, 5, 7))
        for x in range(90, 250):
            for y in range(40, 360):
                image.putpixel((x, y), (240, 228, 206))
        left, top, right, bottom = detect_label(image).bbox
        self.assertLess(left, 90 / 640 + 0.05)
        self.assertGreater(right, 250 / 640 - 0.05)
        self.assertLess(right, 0.52)

    def test_warps_trapezoid_paper_to_a_front_rectangle(self):
        pixels = np.zeros((420, 520, 3), dtype=np.uint8)
        pixels[:] = (6, 6, 8)
        for y in range(50, 390):
            t = (y - 50) / 340
            left = int(190 + t * (-55))
            right = int(300 + t * 70)
            pixels[y, left:right] = (236, 224, 196)
        image = Image.fromarray(pixels, 'RGB')
        detection = detect_label(image)
        self.assertIsNotNone(detection.quad)
        self.assertEqual(len(detection.quad), 4)
        cropped = crop_label(image, detection)
        sample = np.asarray(cropped.resize((32, 48)))
        self.assertGreater(sample.mean(), 140)
        self.assertGreater(cropped.height, cropped.width * 0.9)

    def test_arched_label_preserves_top_without_false_perspective(self):
        image = Image.new("RGB", (400, 500), (12, 15, 14))
        draw = ImageDraw.Draw(image)
        draw.ellipse((105, 80, 295, 200), fill=(236, 224, 196))
        draw.rectangle((105, 140, 295, 430), fill=(236, 224, 196))
        draw.rectangle((155, 210, 250, 290), fill=(24, 24, 24))
        detection = detect_label(image)
        self.assertIsNotNone(detection.contour)
        self.assertIsNone(detection.quad)
        self.assertLess(min(y for x, y in detection.contour), .19)
        cropped = crop_label(image, detection)
        self.assertEqual(cropped.getpixel((0, 0)), (255, 255, 255))
        # A dark illustration inside the paper must not become a background hole.
        self.assertLess(np.asarray(cropped)[cropped.height // 2, cropped.width // 2].mean(), 60)

    def test_shop_photo_does_not_join_shelf_labels(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/yandex-real-eval/01-loco-cimbali-oranzh-muskat-oranzhevoe-suhoe-127.jpg")
        detection = detect_label(image)
        self.assertIsNotNone(detection.contour)
        xs, ys = zip(*detection.contour)
        self.assertGreater(min(ys), .62)
        self.assertLess(max(ys), .94)
        self.assertGreater(min(xs), .27)
        self.assertLess(max(xs), .72)
        self.assertGreater(max(xs) - min(xs), .33)
        self.assertGreater(max(ys) - min(ys), .23)

    def test_catalog_label_can_touch_both_sides(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/loco_cimbali_oranzh_muskat_oranzhevoe_suhoe_127_dc98a21033.webp")
        detection = detect_label(image, catalog=True)
        self.assertGreater(detection.bbox[1], .60)
        self.assertLess(detection.bbox[3] - detection.bbox[1], .38)
        self.assertIsNotNone(detection.contour)

    def test_tall_product_shot_crops_the_label_band(self):
        pixels = np.full((900, 280, 3), 248, dtype=np.uint8)
        pixels[50:850, 80:200] = (36, 72, 40)
        pixels[500:790, 84:196] = (236, 224, 196)
        image = Image.fromarray(pixels)
        left, top, right, bottom = detect_label(image).bbox
        self.assertGreater(top, 0.42)
        self.assertLess(bottom, 0.95)
        self.assertLess(bottom - top, 0.48)
        self.assertGreater(bottom - top, 0.18)


if __name__ == "__main__":
    unittest.main()
