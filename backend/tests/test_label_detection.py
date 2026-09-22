import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.label_detection import LabelDetection, crop_front_design, crop_label, crop_quality, detect_label, enhance_label


class LabelDetectionTests(unittest.TestCase):
    def test_small_upper_glare_region_is_rejected(self):
        image = Image.new('RGB', (3024, 4032))
        detection = LabelDetection((.4358, .2087, .635, .4413), .9)
        quality = crop_quality(image, detection)
        self.assertFalse(quality['usable'])
        self.assertIn('small_upper_fragment', quality['reasons'])

    def test_printed_glass_fallback_crops_below_glare_without_changing_pixels(self):
        image = Image.open(Path(__file__).parents[2] / 'Датасет/Реальные фото/1.73_06-09-2026_14-54-06.webp').convert('RGB')
        detection = detect_label(image)
        self.assertIn('small_upper_fragment', crop_quality(image, detection)['reasons'])
        cropped = crop_front_design(image, detection)
        self.assertLess(cropped.width, image.width)
        self.assertLess(cropped.height, image.height)
        self.assertEqual(cropped.getpixel((cropped.width // 2, cropped.height // 2)),
                         image.getpixel((round(((detection.bbox[0] + detection.bbox[2]) / 2 - .32) * image.width) + cropped.width // 2,
                                         round((detection.bbox[3] + .02) * image.height) + cropped.height // 2)))

    def test_query_crop_keeps_artwork_outside_paper_color_mask(self):
        image = Image.new('RGB', (100, 100), 'blue')
        detection = LabelDetection((0, 0, 1, 1), .8,
                                   contour=((0, 0), (1, 0), (.5, .5)))
        self.assertEqual(crop_label(image, detection, preserve_pixels=True).getpixel((50, 90)), (0, 0, 255))
        self.assertEqual(crop_label(image, detection).getpixel((50, 90)), (255, 255, 255))

    def test_narrow_real_label_is_not_rejected_by_aspect_alone(self):
        image = Image.new('RGB', (400, 600))
        detection = LabelDetection((.3, .1, .6, .9), .8)
        self.assertTrue(crop_quality(image, detection)['usable'])

    def test_transparent_packshot_excludes_glass_and_preserves_illustration(self):
        image = Image.open(Path(__file__).parent / 'fixtures/riesling-catalog.png')
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertGreater(top, .56)
        self.assertLess(top, .65)
        self.assertGreater(bottom, .90)
        self.assertLess(bottom, .97)
        self.assertGreater(right - left, .65)

    def test_label_touching_light_table_keeps_brand_and_bottom(self):
        image = Image.open(Path(__file__).parent / 'fixtures/riesling-table.png')
        detection = detect_label(image)
        left, top, right, bottom = detection.bbox
        self.assertGreater(top, .50)
        self.assertLess(top, .62)
        self.assertGreater(bottom, .85)
        self.assertLess(bottom, .93)
        self.assertLess(left, .32)
        self.assertGreater(right, .70)
        self.assertLess(right - left, .60)

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

    def test_gallery_shot_keeps_the_white_label_not_the_shelf(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/yandex-real-eval/41-risling.jpg")
        detection = detect_label(image)
        cropped = crop_label(image, detection)
        self.assertGreater(np.asarray(cropped).mean(), 140)
        self.assertGreater(detection.bbox[1], 0.45)
        self.assertLess(detection.bbox[0], 0.45)
        self.assertGreater(detection.bbox[2], 0.55)

    def test_closeup_does_not_crop_the_left_glass_strip(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/yandex-real-eval/48-usadba-markoth-kyuve-blan-shardone-beloe-suhoe-12.jpg")
        detection = detect_label(image)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.55)
        self.assertLess(detection.bbox[0], 0.15)
        cropped = crop_label(image, detection)
        self.assertGreater(np.asarray(cropped).mean(), 150)

    def test_printed_on_glass_rose_keeps_the_front_bottle(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/yandex-real-eval/10-roze-1.jpg")
        detection = detect_label(image)
        cx = (detection.bbox[0] + detection.bbox[2]) / 2
        self.assertLess(abs(cx - 0.5), 0.22)
        self.assertGreater(detection.bbox[3] - detection.bbox[1], 0.16)
        self.assertIn(detection.method, {"amber_panel", "full_frame", "label_panel", "paper_band", "connected_paper", "paper_quad"})
        if detection.method != "full_frame":
            self.assertGreater(detection.bbox[1], 0.35)

    def test_rkatsiteli_closeup_keeps_the_left_of_the_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/yandex-real-eval/23-loco-cimbali-rkatsiteli-oranzhevoe-suhoe-13.jpg")
        detection = detect_label(image)
        self.assertLess(detection.bbox[0], 0.12)
        self.assertGreater(detection.bbox[3] - detection.bbox[1], 0.22)

    def test_catalog_label_can_touch_both_sides(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/loco_cimbali_oranzh_muskat_oranzhevoe_suhoe_127_dc98a21033.webp")
        detection = detect_label(image, catalog=True)
        self.assertGreater(detection.bbox[1], .60)
        self.assertLess(detection.bbox[3] - detection.bbox[1], .38)
        self.assertLess(detection.bbox[0], .20)
        self.assertGreater(detection.bbox[2], .62)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], .50)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.width, 200)
        self.assertGreater(cropped.width / max(1, cropped.height), 0.55)
        self.assertIsNotNone(detection.contour)

    def test_catalog_crop_drops_the_dark_shoulder_above_a_flat_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_abrau_dyurso_pino_nuar_krasnoe_suhoe_13_2c1e6330fe.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "trimmed_panel")
        self.assertGreater(detection.bbox[1], 0.68)
        self.assertLess(detection.bbox[1], 0.74)
        self.assertLess(detection.bbox[3], 0.95)
        cropped = crop_label(image, detection)
        self.assertLess(cropped.height, 420)
        top_row = np.asarray(cropped)[2]
        self.assertGreater(float(top_row.mean()), 140)

    def test_catalog_gold_print_covers_the_dark_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_abrau_estates_krasnoe_kaberne_sovinon_suhoe_13_1770f5150f.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "dark_print")
        self.assertGreater(detection.bbox[1], 0.36)
        self.assertLess(detection.bbox[1], 0.50)
        self.assertGreater(detection.bbox[3], 0.84)
        self.assertLess(detection.bbox[3], 0.96)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.45)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.height, 350)
        self.assertLess(cropped.height, image.height * 0.65)

    def test_catalog_red_panel_replaces_a_glass_sliver(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_abrau_estates_dostoynyy_krasnoe_suhoe_135_a2b9e120e4.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "saturated_panel")
        self.assertGreater(detection.bbox[1], 0.40)
        self.assertLess(detection.bbox[3], 0.94)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.7)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.height, 300)
        self.assertLess(cropped.height, image.height * 0.6)
        sample = np.asarray(cropped)
        field = sample[int(sample.shape[0] * 0.35), int(sample.shape[1] * 0.72)]
        self.assertGreater(int(field[0]), 140)
        self.assertLess(int(field[1]), 90)

    def test_catalog_dark_print_crops_the_label_not_the_bottle(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_abrau_estates_amurskiy_potapenko_krasnoe_suhoe_105_a70c9cabe2.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "dark_print")
        self.assertGreater(detection.bbox[1], 0.37)
        self.assertLess(detection.bbox[1], 0.46)
        self.assertGreater(detection.bbox[3], 0.80)
        self.assertLess(detection.bbox[3], 0.90)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.25)
        cropped = crop_label(image, detection)
        self.assertLess(cropped.height, image.height * 0.55)
        self.assertGreater(cropped.width, 400)
        sample = np.asarray(cropped)
        self.assertLess(float(sample.mean()), 90)
        self.assertGreater(int(sample[:, :, 0].max()), 150)

    def test_catalog_saturated_panel_crops_the_colour_not_the_bottle(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        cases = (
            ("a_gordienko_m_nikolaev_pino_nuar_krasnoe_suhoe_135_660d8a5012.webp", (180, 40, 70)),
            ("a_gordienko_m_nikolaev_sira_nuvo_krasnoe_suhoe_115_3eff431cec.webp", (40, 160, 210)),
        )
        for filename, colour in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                self.assertEqual(detection.method, "saturated_panel")
                self.assertGreater(detection.bbox[1], 0.55)
                self.assertLess(detection.bbox[3] - detection.bbox[1], 0.42)
                self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.55)
                cropped = crop_label(image, detection)
                self.assertGreater(cropped.width, 120)
                self.assertLess(cropped.height, image.height * 0.45)
                sample = np.asarray(cropped)
                field = sample[int(sample.shape[0] * 0.72), sample.shape[1] // 2].astype(int)
                self.assertGreater(int(field.max()), 80)
                self.assertLess(abs(int(field[0]) - colour[0]) + abs(int(field[1]) - colour[1]) + abs(int(field[2]) - colour[2]), 180)

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

    def test_catalog_white_gap_crops_the_pale_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_abrau_estates_roze_kaberne_sovinon_rozovoe_suhoe_12_416c8f123b.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "white_gap")
        self.assertGreater(detection.bbox[1], 0.38)
        self.assertLess(detection.bbox[1], 0.50)
        self.assertGreater(detection.bbox[3], 0.84)
        self.assertLess(detection.bbox[3], 0.94)
        cropped = crop_label(image, detection)
        self.assertGreater(float(np.asarray(cropped).mean()), 160)
        self.assertLess(cropped.height, image.height * 0.55)

    def test_catalog_lower_print_crops_the_diamond(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        names = (
            "abrau_dyurso_brut_dor_blanc_de_noirs_pino_nuar_beloe_bryut_125_2d16522a48.webp",
            "abrau_dyurso_brut_dor_rose_pino_nuar_rozovoe_bryut_12_8b5bd155e6.webp",
            "abrau_dyurso_brut_dor_blanc_de_noirs_pino_nuar_igristoe_bryut_beloe_115_d3fbb6965d.webp",
        )
        for name in names:
            with self.subTest(name=name):
                image = Image.open(uploads / name)
                detection = detect_label(image, catalog=True)
                self.assertEqual(detection.method, "lower_print")
                self.assertGreater(detection.bbox[1], 0.55)
                self.assertLess(detection.bbox[3], 0.96)
                self.assertGreater(detection.bbox[3] - detection.bbox[1], 0.18)
                self.assertLess(detection.bbox[3] - detection.bbox[1], 0.40)
                cropped = crop_label(image, detection)
                self.assertGreater(cropped.height, 200)

    def test_catalog_emblem_panel_crops_the_dark_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_az_abrau_madrasa_krasnoe_suhoe_14_a2a0b9e8a3.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "emblem_panel")
        self.assertGreater(detection.bbox[1], 0.40)
        self.assertLess(detection.bbox[1], 0.52)
        self.assertGreater(detection.bbox[3], 0.84)
        self.assertLess(detection.bbox[3], 0.94)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.45)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.height, 800)
        self.assertLess(cropped.height, image.height * 0.55)

    def test_catalog_crest_band_keeps_the_monogram_and_name(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_aleksandr_ii_magnum_shardone_beloe_bryut_125_256840c1d1.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "crest_band")
        self.assertGreater(detection.bbox[1], 0.48)
        self.assertLess(detection.bbox[1], 0.62)
        self.assertGreater(detection.bbox[3], 0.88)
        self.assertLess(detection.bbox[3], 0.97)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.height, 250)
        self.assertLess(cropped.height, image.height * 0.5)

    def test_catalog_artwork_crop_does_not_punch_white_holes(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_femme_dabrau_amurskiy_potapenko_krasnoe_suhoe_107_d599cb8a22.webp")
        detection = detect_label(image, catalog=True)
        self.assertIsNone(detection.contour)
        cropped = crop_label(image, detection)
        white = (np.asarray(cropped) > 250).all(axis=2).mean()
        self.assertLess(float(white), 0.12)
        self.assertGreater(cropped.height, 1000)

    def test_catalog_paper_below_keeps_the_cream_oval(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_imperial_brut_vintage_shardone_beloe_bryut_12_8f6dbe7a71.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "paper_below")
        self.assertGreater(detection.bbox[1], 0.65)
        self.assertLess(detection.bbox[3], 0.95)
        self.assertGreater(detection.bbox[3] - detection.bbox[1], 0.12)
        self.assertLess(detection.bbox[3] - detection.bbox[1], 0.26)
        cropped = crop_label(image, detection)
        self.assertGreater(float(np.asarray(cropped).mean()), 160)

    def test_catalog_shield_keeps_a_narrow_diamond(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_russkoe_igristoe_koshernoe_polusladkoe_shardone_beloe_12_cce09fd580.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "shield")
        self.assertGreater(detection.bbox[1], 0.52)
        self.assertLess(detection.bbox[3], 0.92)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.24)
        self.assertLess(detection.bbox[2] - detection.bbox[0], 0.40)

    def test_catalog_dark_print_keeps_the_illustration_and_footer(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/Agora_Blek_Stoun_e98339ae1b.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "dark_print")
        self.assertLess(detection.bbox[1], 0.48)
        self.assertGreater(detection.bbox[3], 0.80)
        cropped = crop_label(image, detection)
        sample = np.asarray(cropped)
        self.assertGreater(float(sample[: sample.shape[0] // 3].mean()), 80)
        self.assertLess(float(sample[-sample.shape[0] // 5 :].mean()), 70)

    def test_catalog_footer_keeps_the_name_under_the_painting(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/Agora_Bastardo_59a04897ff.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "label_footer")
        self.assertIsNone(detection.contour)
        self.assertGreater(detection.bbox[3], 0.84)
        self.assertLess(detection.bbox[3], 0.95)
        cropped = crop_label(image, detection)
        self.assertLess(float(np.asarray(cropped)[-8:].mean()), 40)

    def test_catalog_dark_print_keeps_the_riesling_name_band(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/Agora_Risling_1b886b07f8.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "dark_print")
        self.assertLess(detection.bbox[1], 0.48)
        self.assertGreater(detection.bbox[3], 0.88)
        self.assertIsNone(detection.contour)

    def test_catalog_matte_band_keeps_the_white_still_label(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/B_Yr_Ab_T0x_R_Ed_Rg8j_1775199389_86635deb7f.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "matte_band")
        self.assertGreater(detection.bbox[1], 0.55)
        self.assertLess(detection.bbox[3], 0.95)
        self.assertGreater(detection.bbox[3] - detection.bbox[1], 0.18)
        self.assertLess(detection.bbox[3] - detection.bbox[1], 0.36)

    def test_catalog_gold_label_keeps_the_vintage_ornament(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/Blan_de_Nuar_f3d5c90643.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "gold_label")
        self.assertGreater(detection.bbox[1], 0.62)
        self.assertLess(detection.bbox[3], 0.98)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.60)

    def test_catalog_colour_panel_keeps_the_saperavi_eagle(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/agrolayn_mountain_eagle_saperavi_saperavi_krasnoe_suhoe_135_6d3f5cfe1e.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "colour_panel")
        self.assertLess(detection.bbox[1], 0.50)
        self.assertGreater(detection.bbox[3], 0.78)

    def test_catalog_ink_shield_keeps_the_rose_cuvee(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/agora_winery_cuvee_pino_nuar_shardone_bryut_rozovoe_igristoe_bryut_rozovoe_125_c2b30c8f55.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "ink_shield")
        self.assertGreater(detection.bbox[1], 0.64)
        self.assertLess(detection.bbox[3], 0.95)

    def test_catalog_shield_keeps_the_reserve_under_the_ram(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/alma_valley_kaberne_fran_rezerv_krasnoe_suhoe_15_6ef14d79ec.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "shield")
        self.assertLess(detection.bbox[1], 0.55)
        self.assertGreater(detection.bbox[3], 0.78)

    def test_catalog_glass_band_keeps_the_dark_gravity_square(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/alma_valley_alma_graviti_pino_nuar_merlo_kaberne_sovinon_krasnoe_suhoe_14_aed5a84143.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "glass_band")
        self.assertGreater(detection.bbox[1], 0.55)
        self.assertLess(detection.bbox[3], 0.95)
        self.assertLess(detection.bbox[3] - detection.bbox[1], 0.40)

    def test_catalog_flat_paper_stays_a_rectangle(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/fp_ZE_Jh_YZ_Pst_BYLV_1775199670_92154c67f4.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "connected_paper")
        self.assertIsNone(detection.contour)
        self.assertGreater(detection.bbox[1], 0.55)
        self.assertLess(detection.bbox[3], 0.95)

    def test_catalog_curved_label_keeps_the_last_word(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/Agora_Rezerv_Yahting_Pino_Gridzhio_5daf688202.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "trimmed_panel")
        self.assertGreater(detection.bbox[2], 0.88)
        self.assertLess(detection.bbox[1], 0.50)
        self.assertGreater(detection.bbox[3], 0.84)

    def test_catalog_label_bite_stays_rectangular(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/DSC_09173_4a9ff95cc2.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "connected_paper")
        self.assertIsNone(detection.contour)
        self.assertGreater(detection.bbox[3], 0.90)

    def test_catalog_short_wide_diamond_stays_rectangular(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/abrau_dyurso_victor_dravigny_extra_brut_shardone_beloe_bryut_125_86dc223df3.webp")
        detection = detect_label(image, catalog=True)
        self.assertIsNone(detection.contour)
        self.assertGreater(detection.bbox[1], 0.62)
        self.assertLess(detection.bbox[3], 0.96)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.70)


if __name__ == "__main__":
    unittest.main()
