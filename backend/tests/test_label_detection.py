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

    def test_transparent_packshot_dark_label_band_is_not_glass(self):
        image = Image.open(
            Path(__file__).parents[2]
            / 'Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/'
            / 'PU_Sjv7dn_M_Nil_H25_TO_9_Bcb_Bjo_R8u_Zy_Sbk_V5_WTE_Jjg3_C_Ky_5_U_v_Ue_Qv_P6o_l_XGG_5q_NED_Mtd_Ztu_Uj_Etjwa_Qrt_TX_Ag_c31d9f8afc.webp'
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, 'dark_band')
        self.assertGreater(top, .60)
        self.assertLess(top, .68)
        self.assertGreater(bottom, .82)
        self.assertLess(bottom, .90)
        self.assertGreater(right - left, .75)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.width, 350)
        self.assertGreater(float(np.asarray(cropped).mean()), 50)

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

    def test_crowded_shop_photo_prefers_front_bottle_over_side_shelf_cards(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/Реальные фото/90.5_02-09-2026_10-32-11.webp"
        )
        detection = detect_label(image)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "amber_panel")
        self.assertGreater(left, 0.28)
        self.assertLess(left, 0.36)
        self.assertGreater(right, 0.60)
        self.assertLess(right, 0.68)
        self.assertGreater(top, 0.52)
        self.assertLess(top, 0.63)
        self.assertGreater(bottom, 0.78)
        self.assertLess(bottom, 0.88)
        crop = crop_label(image, detection)
        self.assertGreater(np.asarray(crop).mean(), 80)

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

    def test_catalog_lower_gold_print_ignores_neck_text_and_bottle_body(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        images = (
            uploads
            / "fanagoriya_fanagoriya_sakra_sennoy_krasnostop_krasnoe_suhoe_135_f22347e4bd.webp",
            next(uploads.glob("*412368f0b0.webp")),
        )
        for path in images:
            with self.subTest(path=path.name):
                image = Image.open(path)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "dark_print")
                self.assertGreater(top, 0.53)
                self.assertLess(top, 0.63)
                self.assertGreater(bottom, 0.88)
                self.assertLess(bottom, 0.96)
                self.assertGreater(right - left, 0.85)
                self.assertLess(bottom - top, 0.45)
                self.assertGreater(crop_label(image, detection).width, 300)

    def test_catalog_light_label_wins_over_dark_glass_above_it(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"
            / "gavras_chardonnay_shardone_beloe_suhoe_13_c69d79ea2c.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "connected_paper")
        self.assertLess(top, 0.68)
        self.assertGreater(top, 0.60)
        self.assertGreater(bottom, 0.90)
        self.assertGreater(right - left, 0.85)
        crop = crop_label(image, detection)
        self.assertGreater(crop.width, 240)
        self.assertGreater(np.asarray(crop).mean(), 100)

    def test_catalog_wide_dark_sleeve_wins_over_central_diamond(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"
            / "Screenshot_2_66036870aa.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "emblem_panel")
        self.assertLess(left, 0.25)
        self.assertGreater(right, 0.75)
        self.assertGreater(top, 0.55)
        self.assertLess(top, 0.70)
        self.assertGreater(bottom, 0.85)
        self.assertLess(bottom, 0.95)
        crop = crop_label(image, detection)
        self.assertGreater(crop.width, 150)
        self.assertGreater(crop.height, 150)

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

    def test_catalog_golubitskoe_petnat_expands_a_white_label_fragment(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/golubitskoe_estate_petnat_white_risling_reynskiy_petnat_suhoy_belyy_115_e6d0ac50f9.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "white_gap")
        self.assertLess(detection.bbox[0], 0.08)
        self.assertGreater(detection.bbox[2], 0.92)
        self.assertGreater(detection.bbox[1], 0.58)
        self.assertGreater(detection.bbox[3], 0.88)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.width, image.width * 0.85)
        self.assertGreater(cropped.height, image.height * 0.20)

    def test_catalog_golubitskoe_tete_de_cheval_labels_keep_the_full_front_panel(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        filenames = (
            "golubitskoe_estate_tte_de_cheval_bryut_risling_igristoe_bryut_beloe_12_4852fa5a2f.webp",
            "golubitskoe_estate_tte_de_cheval_zero_dosage_ekstra_bryut_sovinon_blan_igristoe_ekstra_bryut_beloe_12_9438066f79.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                self.assertEqual(detection.method, "glass_band")
                self.assertLess(detection.bbox[0], 0.08)
                self.assertGreater(detection.bbox[2], 0.92)
                self.assertGreater(detection.bbox[1], 0.55)
                self.assertGreater(detection.bbox[3], 0.90)
                cropped = crop_label(image, detection)
                self.assertGreater(cropped.width, image.width * 0.85)
                self.assertGreater(cropped.height, image.height * 0.28)

    def test_catalog_gunko_monochrome_labels_keep_the_full_panel(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        filenames = (
            "gunko_winery_risling_rezerv_beloe_suhoe_135_e2e2bb75a2.webp",
            "gunko_winery_saperavi_gunko_winery_krasnoe_suhoe_135_530300631c.webp",
            "gunko_winery_sovinon_blan_gunko_winery_beloe_suhoe_135_018fec8f4b.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                self.assertEqual(detection.method, "achromatic_band")
                self.assertGreater(detection.bbox[1], 0.36)
                self.assertLess(detection.bbox[1], 0.47)
                self.assertGreater(detection.bbox[3], 0.84)
                self.assertLess(detection.bbox[3], 0.94)
                cropped = crop_label(image, detection)
                self.assertGreater(cropped.width, image.width * 0.85)
                self.assertGreater(cropped.height, image.height * 0.40)
                self.assertLess(cropped.height, image.height * 0.58)

    def test_catalog_di_caspico_uses_the_white_cyan_front_banner(self):
        image = Image.open(Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/bel_plsl_16457d7b78.webp")
        detection = detect_label(image, catalog=True)
        self.assertEqual(detection.method, "bright_banner")
        self.assertLess(detection.bbox[1], 0.80)
        self.assertGreater(detection.bbox[1], 0.70)
        self.assertGreater(detection.bbox[3], 0.90)
        self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.90)
        cropped = crop_label(image, detection)
        self.assertGreater(cropped.width, cropped.height)
        self.assertGreater(cropped.height, image.height * 0.15)
        self.assertLess(cropped.height, image.height * 0.25)

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

    def test_sikory_packshots_crop_the_printed_label_instead_of_the_bottle(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        filenames = (
            "imenie_sikory_gerts_kaberne_sovinon_krasnoe_suhoe_vyderzhannoe_14_c875bac299.webp",
            "imenie_sikory_gerts_sikory_kaberne_sovinon_krasnoe_suhoe_14_2dc7c653fd.webp",
            "imenie_sikory_kaberne_fran_roze_rozovoe_suhoe_13_4f0044a40b.webp",
            "imenie_sikory_kaberne_fran_sikory_rozovoe_suhoe_13_734a89a002.webp",
            "imenie_sikory_kaberne_sovinon_semeynyy_rezerv_krasnoe_suhoe_vyderzhannoe_14_dc4aa1ddb5.webp",
            "imenie_sikory_krasnostop_zolotovskiy_na_terrasah_krasnoe_suhoe_14_93b75eb2a0.webp",
            "imenie_sikory_krasnostop_zolotovskiy_na_terrasah_rozovoe_suhoe_13_fd7deda788.webp",
            "imenie_sikory_krasnostop_zolotovskiy_pozdniy_sbor_rozovoe_sladkoe_13_305f6270ac.webp",
            "imenie_sikory_merlo_semeynyy_rezerv_krasnoe_suhoe_vyderzhannoe_14_ddcc4b5b79.webp",
            "imenie_sikory_pino_nuar_semeynyy_rezerv_krasnoe_suhoe_14_e1d961aa68.webp",
            "imenie_sikory_pino_nuar_sikory_krasnoe_suhoe_14_4a36d13ce1.webp",
            "imenie_sikory_risling_pozdniy_sbor_beloe_sladkoe_13_e60898ab3b.webp",
            "imenie_sikory_risling_semeynyy_rezerv_beloe_suhoe_13_3ef1716553.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertGreater(top, 0.30)
                self.assertLess(bottom, 0.98)
                self.assertLess(bottom - top, 0.60)
                self.assertGreater(right - left, 0.20)
                self.assertIsNone(detection.contour)
                crop = crop_label(image, detection)
                self.assertLess(crop.height, image.height * 0.65)

    def test_argonne_striped_packshots_use_the_centered_name_panel(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        filenames = (
            "ivan_ksenia_kruz_argonne_chardonay_shardone_beloe_suhoe_127_d4a21f7bc6.webp",
            "ivan_ksenia_kruz_argonne_pinot_noir_pino_nuar_rozovoe_suhoe_122_956b877f5c.webp",
            "ivan_ksenia_kruz_argonne_riesling_risling_beloe_suhoe_119_7c2a4d6ca1.webp",
            "ivan_ksenia_kruz_argonne_sauvignon_blanc_sovinon_blan_beloe_suhoe_115_e25c7aa235.webp",
            "ivan_ksenia_kruz_argonne_syrah_sira_krasnoe_suhoe_124_9d92c7e4ae.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "crest_band")
                self.assertGreater(left, 0.20)
                self.assertLess(left, 0.30)
                self.assertGreater(right, 0.70)
                self.assertLess(right, 0.82)
                self.assertGreater(top, 0.80)
                self.assertLess(top, 0.92)
                self.assertGreater(bottom, 0.90)
                self.assertLess(bottom, 0.98)
                crop = crop_label(image, detection)
                self.assertGreater(crop.width, image.width * 0.40)
                self.assertLess(crop.height, image.height * 0.20)

    def test_jd_transparent_packshots_keep_the_whole_front_label_band(self):
        uploads = Path(__file__).parents[2] / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        filenames = (
            "20260617_071733_Photoroom_31245772bb.webp",
            "20260617_072415_Photoroom_6dda2f8425.webp",
            "20260617_070147_Photoroom_2bf16b8d06.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "alpha_label_band")
                self.assertGreater(left, 0.08)
                self.assertLess(right, 0.93)
                self.assertGreater(right - left, 0.32)
                self.assertGreater(top, 0.50)
                self.assertLess(top, 0.70)
                self.assertGreater(bottom, 0.82)
                self.assertLess(bottom, 0.98)
                self.assertGreater(crop_label(image, detection).height, image.height * 0.15)

    def test_katharon_brut_crop_uses_the_front_label_instead_of_the_neck(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kataron_bryut_3a1af9a036.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "trimmed_panel")
        self.assertGreater(left, 0.32)
        self.assertLess(left, 0.42)
        self.assertGreater(right, 0.58)
        self.assertLess(right, 0.68)
        self.assertGreater(top, 0.58)
        self.assertLess(top, 0.70)
        self.assertGreater(bottom, 0.78)
        self.assertLess(bottom, 0.88)
        crop = crop_label(image, detection)
        self.assertGreater(crop.width, image.width * 0.20)
        self.assertGreater(crop.height, image.height * 0.16)

    def test_aristov_pinot_nero_uses_the_complete_coloured_front_label(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_aristov_anima_pino_nero_sandzhoveze_pino_nuar_krasnoe_suhoe_125_6fc8b03351.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "saturated_panel")
        self.assertLess(left, 0.03)
        self.assertGreater(right, 0.97)
        self.assertGreater(top, 0.60)
        self.assertLess(top, 0.69)
        self.assertGreater(bottom, 0.86)
        self.assertLess(bottom, 0.95)
        crop = crop_label(image, detection)
        self.assertGreater(crop.width, image.width * 0.80)
        self.assertGreater(crop.height, image.height * 0.20)

    def test_aristov_byanko_uses_its_printed_label_not_the_bottle_body(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_aristov_byanko_polusladkoe_shardone_beloe_13_4167038d66.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "printed_band")
        self.assertLess(left, 0.04)
        self.assertGreater(right, 0.96)
        self.assertGreater(top, 0.60)
        self.assertLess(top, 0.72)
        self.assertGreater(bottom, 0.87)
        self.assertLess(bottom, 0.96)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.20)
        self.assertLess(crop.height, image.height * 0.40)

    def test_hrustaleva_muscat_brut_excludes_the_bottle_base(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/hrustaleva_bryut_muskat_314381c682.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "matte_band")
        self.assertGreater(left, 0.05)
        self.assertLess(right, 0.96)
        self.assertGreater(top, 0.55)
        self.assertLess(top, 0.64)
        self.assertGreater(bottom, 0.76)
        self.assertLess(bottom, 0.88)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.18)
        self.assertLess(crop.height, image.height * 0.30)

    def test_aristov_sangiovese_keeps_both_halves_of_the_wrap_label(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_aristov_sandzhoveze_krasnoe_suhoe_13_346f0e6195.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "printed_band")
        self.assertLess(left, 0.04)
        self.assertGreater(right, 0.96)
        self.assertGreater(top, 0.60)
        self.assertLess(top, 0.68)
        self.assertGreater(bottom, 0.90)
        self.assertLess(bottom, 0.96)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.25)
        self.assertLess(crop.height, image.height * 0.36)

    def test_aristov_chardonnay_uses_the_full_illustrated_label(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_aristov_shardone_318_beloe_suhoe_115_da1c20930f.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "colour_panel")
        self.assertLess(left, 0.06)
        self.assertGreater(right, 0.94)
        self.assertGreater(top, 0.46)
        self.assertLess(top, 0.55)
        self.assertGreater(bottom, 0.84)
        self.assertLess(bottom, 0.92)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.32)
        self.assertLess(crop.height, image.height * 0.46)

    def test_aristov_zweigelt_trims_the_qr_panel_to_the_full_label_band(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_aristov_tsvaygelt_320_rozovoe_suhoe_125_42f977144e.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "printed_band")
        self.assertLess(left, 0.03)
        self.assertGreater(right, 0.97)
        self.assertGreater(top, 0.60)
        self.assertLess(top, 0.67)
        self.assertGreater(bottom, 0.90)
        self.assertLess(bottom, 0.96)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.25)
        self.assertLess(crop.height, image.height * 0.35)

    def test_shato_taman_reserve_label_crop_excludes_the_lower_bottle(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/kuban_vino_shato_tamane_rezerv_kaberne_limited_edishn_2015_kaberne_sovinon_krasnoe_suhoe_13_ba74d4f88e.webp"
        )
        detection = detect_label(image, catalog=True)
        _left, top, _right, bottom = detection.bbox
        self.assertEqual(detection.method, "trimmed_panel")
        self.assertGreater(top, 0.40)
        self.assertLess(top, 0.48)
        self.assertGreater(bottom, 0.73)
        self.assertLess(bottom, 0.82)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.30)
        self.assertLess(crop.height, image.height * 0.40)

    def test_le_k2_dark_rose_uses_the_complete_front_label_band(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/le_k2_temnaya_roza_sira_rozovoe_suhoe_13_80e154d18a.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "printed_band")
        self.assertLess(left, 0.03)
        self.assertGreater(right, 0.97)
        self.assertGreater(top, 0.64)
        self.assertLess(top, 0.71)
        self.assertGreater(bottom, 0.86)
        self.assertLess(bottom, 0.93)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.20)
        self.assertLess(crop.height, image.height * 0.26)

    def test_le_k2_vassal_riesling_selects_the_front_badge(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/le_k2_vassal_risling_beloe_suhoe_14_19652f8fb4.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "alpha_label_band")
        self.assertGreater(left, 0.18)
        self.assertLess(left, 0.27)
        self.assertGreater(right, 0.70)
        self.assertLess(right, 0.78)
        self.assertGreater(top, 0.64)
        self.assertLess(top, 0.73)
        self.assertGreater(bottom, 0.89)
        self.assertLess(bottom, 0.96)
        crop = crop_label(image, detection)
        self.assertGreater(crop.height, image.height * 0.20)
        self.assertLess(crop.height, image.height * 0.30)

    def test_marquetry_and_molodoe_use_the_complete_front_wrap_band(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        cases = (
            (
                "AYA_marquetry_syrah_merlot_cabernet_franc_2023_bottle_1174396457.webp",
                "glass_band",
                (0.02, 0.60, 0.88, 0.98),
            ),
            (
                "AYA_marquetry_syrah_2023_bottle_90a876b523.webp",
                "printed_band",
                (0.02, 0.66, 0.95, 0.95),
            ),
            (
                "Molodoe_Atlas_2025_Normal_Krasnoe_748e72dcb7.webp",
                "printed_band",
                (0.0, 0.68, 1.0, 0.97),
            ),
            (
                "Molodoe_Atlas_2025_Normal_Beloe_da0ba8a14a.webp",
                "printed_band",
                (0.0, 0.69, 1.0, 0.96),
            ),
        )
        for filename, expected_method, bounds in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, expected_method)
                self.assertGreaterEqual(left, bounds[0])
                self.assertGreater(top, bounds[1])
                self.assertGreaterEqual(right, bounds[2])
                self.assertLess(bottom, bounds[3])
                crop = crop_label(image, detection, preserve_pixels=True)
                self.assertGreater(crop.width, image.width * 0.80)
                self.assertGreater(crop.height, image.height * 0.20)
                self.assertLess(crop.height, image.height * 0.38)

    def test_vaynkraft_illustrated_wraps_keep_the_photo_and_footer(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        filenames = (
            "vaynkraft_kaberne_sovinon_krasnoe_suhoe_14_ae9402b920.webp",
            "vaynkraft_rkatsiteli_oranzh_oranzhevoe_suhoe_12_65fe9c2119.webp",
            "vaynkraft_traminer_oranzh_2_traminer_rozovyy_oranzhevoe_suhoe_125_3c507d40ce.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "printed_band")
                self.assertLess(left, 0.02)
                self.assertGreater(right, 0.98)
                self.assertGreater(top, 0.58)
                self.assertLess(top, 0.66)
                self.assertGreater(bottom, 0.92)
                self.assertLess(bottom, 0.99)
                crop = crop_label(image, detection)
                self.assertGreater(crop.height, image.height * 0.32)

    def test_vaynkraft_risling_uses_the_full_illustrated_label(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"
            / "vaynkraft_risling_beloe_suhoe_123_a80dc48cd5.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertLess(left, 0.02)
        self.assertGreater(right, 0.98)
        self.assertGreater(top, 0.56)
        self.assertLess(top, 0.64)
        self.assertGreater(bottom, 0.90)
        self.assertLess(bottom, 0.95)

    def test_square_transparent_packshot_uses_the_warm_front_wrap(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"
            / "villa_di_alma_ekstra_bryut_kokur_igristoe_ekstra_bryut_beloe_125_531725a54a.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "warm_panel")
        self.assertGreater(left, 0.32)
        self.assertLess(left, 0.40)
        self.assertGreater(right, 0.60)
        self.assertLess(right, 0.68)
        self.assertGreater(top, 0.66)
        self.assertLess(top, 0.73)
        self.assertGreater(bottom, 0.87)
        self.assertLess(bottom, 0.93)
        crop = crop_label(image, detection)
        self.assertLess(crop.width, image.width * 0.32)
        self.assertLess(crop.height, image.height * 0.28)

    def test_vinodelnya_dark_wraps_keep_brand_emblem_and_variety(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        cases = (
            (
                "vinodelnya_batrak_kaberne_sovinon_krasnoe_suhoe_14_50425a88e9.webp",
                "crest_band",
                (0.60, 0.66, 0.93, 0.98),
            ),
            (
                "vinodelnya_78_saperavi_krasnoe_suhoe_135_1ea4d62431.webp",
                "printed_band",
                (0.60, 0.66, 0.88, 0.94),
            ),
            (
                "vinodelnya_batrak_perfekt_klassik_merlo_krasnoe_suhoe_135_1949b2cb69.webp",
                "printed_band",
                (0.46, 0.53, 0.90, 0.97),
            ),
            (
                "vinodelnya_batrak_perfekt_klassik_saperavi_krasnoe_suhoe_135_b23c1c2ed9.webp",
                "printed_band",
                (0.44, 0.52, 0.89, 0.96),
            ),
        )
        for filename, method, (min_top, max_top, min_bottom, max_bottom) in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, method)
                self.assertLess(left, 0.02)
                self.assertGreater(right, 0.98)
                self.assertGreater(top, min_top)
                self.assertLess(top, max_top)
                self.assertGreater(bottom, min_bottom)
                self.assertLess(bottom, max_bottom)
                crop = crop_label(image, detection)
                self.assertGreater(crop.width, image.width * 0.95)
                self.assertGreater(crop.height, image.height * 0.25)

    def test_zolotoe_pole_white_labels_are_not_confused_with_the_studio_backdrop(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        filenames = (
            "zolotoe_pole_legend_of_crimea_muscat_ottonel_muskat_ottonel_beloe_suhoe_14_b38aa9e08c.webp",
            "zolotoe_pole_legend_of_crimea_white_blend_malvaziya_beloe_suhoe_135_d791def8a0.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "matte_band")
                self.assertGreater(left, 0.17)
                self.assertLess(left, 0.23)
                self.assertGreater(right, 0.77)
                self.assertLess(right, 0.83)
                self.assertGreater(top, 0.56)
                self.assertLess(top, 0.63)
                self.assertGreater(bottom, 0.84)
                self.assertLess(bottom, 0.89)
                crop = crop_label(image, detection, preserve_pixels=True)
                self.assertGreater(crop.width, image.width * 0.55)
                self.assertGreater(crop.height, image.height * 0.20)
                self.assertLess(crop.height, image.height * 0.32)

    def test_fanagoriya_green_wine_prefers_main_label_over_variety_strip(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
            / "fanagoriya_zelyonoe_vino_risling_tsitronnyy_magaracha_beloe_polusuhoe_11_e4de13ab6b.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "paper_above_color")
        self.assertLess(top, 0.43)
        self.assertGreater(bottom, 0.74)
        self.assertLess(bottom, 0.78)
        self.assertGreater(right - left, 0.90)
        self.assertGreater(crop_label(image, detection).height, image.height * 0.30)

    def test_catalog_detector_reproduces_manually_corrected_label_regions(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        # Pixel boxes recovered from the hand-corrected label crops. A modest
        # IoU allowance keeps these useful as visual regressions without
        # requiring the detector to duplicate a person's exact padding.
        cases = (
            ("aya_organic_wine_vineyards_purity_in_syrah_sira_krasnoe_suhoe_146_1d4164b8db.webp", (7, 613, 273, 919)),
            ("aya_organic_wine_vineyards_purity_in_trinity_pino_nuar_rozovoe_suhoe_13_a16516170d.webp", (7, 613, 273, 919)),
            ("vinodelnya_myshako_flute_blaufrankish_igristoe_bryut_rozovoe_126_131ed09b46.webp", (16, 711, 290, 910)),
            ("vinodelnya_myshako_flute_bryut_beloe_shardone_igristoe_bryut_beloe_11_ddbe12b9ec.webp", (15, 711, 291, 911)),
            ("vinodelnya_myshako_flute_gevyurtstraminer_sovinon_blan_igristoe_bryut_beloe_122_9d73a2d644.webp", (15, 711, 291, 911)),
            ("vinodelnya_myshako_flute_vione_aligote_igristoe_bryut_beloe_12_84a5f5532f.webp", (15, 711, 291, 911)),
            ("vinodelnya_myshako_quintessence_brut_muskat_muskat_belyy_igristoe_bryut_beloe_12_b16de08f25.webp", (14, 711, 291, 911)),
            ("vinodelnya_myshako_quintessence_brut_vione_igristoe_bryut_beloe_115_31a5f96a86.webp", (14, 711, 291, 911)),
            ("vinodelnya_myshako_gevyurtstraminer_kyuve_bryut_igristoe_bryut_beloe_123_8da56b2ab5.webp", (38, 693, 266, 918)),
            ("vinodelnya_myshako_kaberne_fran_avtorskaya_tehnologiya_krasnoe_suhoe_142_adbc6382f4.webp", (11, 437, 261, 743)),
            ("vinodelnya_myshako_marselan_avtorskaya_tehnologiya_krasnoe_suhoe_144_d0c6e20dc5.webp", (11, 437, 261, 743)),
            ("vinodelnya_myshako_risling_beloe_suhoe_123_6680d29abd.webp", (29, 468, 243, 741)),
            ("vinodelnya_myshako_quintessence_new_generation_kaberne_fran_krasnoe_suhoe_145_bc712f73ef.webp", (10, 431, 262, 736)),
            ("vinodelnya_myshako_quintessence_new_generation_sovinon_blan_beloe_suhoe_122_c45916b311.webp", (10, 431, 262, 736)),
            ("vinodelnya_myshako_quintessence_shtorm_merlo_krasnoe_sladkoe_147_6c9e12b5a1.webp", (12, 432, 261, 748)),
            ("vinodelnya_marko_merlo_krasnoe_suhoe_14_384d0e4a2a.webp", (11, 406, 230, 820)),
            ("vinodelnya_myshako_merlo_black_out_krasnoe_sladkoe_14_83578ec427.webp", (25, 713, 303, 920)),
            ("vinodelnya_myshako_chernoe_iz_chernogo_appassimento_kaberne_sovinon_krasnoe_suhoe_15_19d090caca.webp", (31, 433, 239, 835)),
            ("vinodelnya_myshako_chernyy_udar_marselan_krasnoe_suhoe_162_0fb37b486f.webp", (31, 433, 239, 835)),
            ("vinodelnya_myshako_pino_nuar_karmener_reserve_krasnoe_suhoe_138_142d47602d.webp", (32, 432, 238, 838)),
            ("vinodelnya_myshako_primitivo_blaufrankish_reserve_krasnoe_suhoe_142_c96e36183b.webp", (32, 432, 238, 838)),
        )
        for filename, expected in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                width, height = image.size
                detection = detect_label(image, catalog=True)
                actual = tuple(
                    round(value * size)
                    for value, size in zip(detection.bbox, (width, height, width, height))
                )
                intersection_width = max(0, min(expected[2], actual[2]) - max(expected[0], actual[0]))
                intersection_height = max(0, min(expected[3], actual[3]) - max(expected[1], actual[1]))
                intersection = intersection_width * intersection_height
                expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
                actual_area = (actual[2] - actual[0]) * (actual[3] - actual[1])
                iou = intersection / max(1, expected_area + actual_area - intersection)
                self.assertGreaterEqual(iou, 0.60, f"{filename}: {actual=} {expected=}, {iou=:.3f}")


    def test_catalog_printed_band_replaces_bottle_body_or_detached_fragment(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        cases = (
            ("Czimlyanskoe_Bianka_beloe_04a59b0f1e.webp", 0.405, 0.693),
            ("Czimlyanskoe_Risling_cea1a80937.webp", 0.400, 0.695),
            ("PETNAT_Rubin_prozrachnaya_3964ed9fec.webp", 0.685, 0.915),
            ("denisov_winery_petnat_rkatsiteli_petnat_ekstra_bryut_belyy_107_5da935c18a.webp", 0.638, 0.915),
            ("denisov_winery_petnat_tsitron_risling_tsitronnyy_magaracha_petnat_ekstra_bryut_belyy_102_8f425093ef.webp", 0.638, 0.915),
            ("denisov_winery_rkatsiteli_beloe_suhoe_112_ed8e9acfcd.webp", 0.546, 0.875),
        )
        for filename, expected_top, expected_bottom in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                self.assertEqual(detection.method, "printed_band")
                self.assertAlmostEqual(detection.bbox[1], expected_top, delta=0.025)
                self.assertAlmostEqual(detection.bbox[3], expected_bottom, delta=0.025)
                self.assertGreater(detection.bbox[2] - detection.bbox[0], 0.34)

    def test_catalog_printed_band_replaces_a_side_edge_fragment(self):
        image = Image.open(
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"
            / "vinodelnya_raevskoe_genezis_beloe_risling_beloe_suhoe_123_e1e994dbf3.webp"
        )
        detection = detect_label(image, catalog=True)
        left, top, right, bottom = detection.bbox
        self.assertEqual(detection.method, "printed_band")
        self.assertLess(left, 0.02)
        self.assertGreater(right, 0.98)
        self.assertGreater(top, 0.43)
        self.assertLess(top, 0.48)
        self.assertGreater(bottom, 0.73)
        self.assertLess(bottom, 0.80)

    def test_catalog_recovers_red_dark_and_wrapped_front_panels(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        cases = (
            (
                "aya_organic_wine_vineyards_purity_in_syrah_sira_rozovoe_suhoe_132_56159ed421.webp",
                "red_panel", 0.58, 0.66, 0.92, 0.99,
            ),
            (
                "belmas_winery_pinot_gris_belmas_pino_gri_beloe_suhoe_122_1fe1641b7c.webp",
                "dark_text_panel", 0.65, 0.75, 0.92, 0.99,
            ),
            (
                "derbent_vino_kavkazian_merlo_krasnoe_polusladkoe_105_125_d2973b5f66.webp",
                "red_panel", 0.70, 0.80, 0.87, 0.95,
            ),
            (
                "nesterov_winery_krasnostop_zolotovskiy_krasnoe_suhoe_12_6036ce85a5.webp",
                "printed_band", 0.44, 0.48, 0.84, 0.91,
            ),
        )
        for filename, method, min_top, max_top, min_bottom, max_bottom in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, method)
                self.assertLess(left, 0.03)
                self.assertGreater(right, 0.95)
                self.assertGreater(top, min_top)
                self.assertLess(top, max_top)
                self.assertGreater(bottom, min_bottom)
                self.assertLess(bottom, max_bottom)
                crop = crop_label(image, detection)
                self.assertGreater(crop.width, image.width * 0.90)
                self.assertGreater(crop.height, image.height * 0.14)

    def test_catalog_prefers_full_desono_front_panel_over_side_glass(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        filenames = (
            "Desono_Kaberne_Fran_d35065b98b.webp",
            "Desono_Kaberne_Sovinon_92912df82f.webp",
            "Desono_Merlo_8f58649b26.webp",
            "Desono_Risling_b70f2d30ac.webp",
            "Desono_Saperavi_606c582da8.webp",
            "Desono_Saperavi_roze_a6f976b72e.webp",
            "Desono_Shardone_46ac5a58e0.webp",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, "printed_band")
                self.assertAlmostEqual(left, 0.124, delta=0.025)
                self.assertAlmostEqual(right, 0.873, delta=0.025)
                self.assertAlmostEqual(top, 0.47, delta=0.04)
                self.assertAlmostEqual(bottom, 0.85, delta=0.04)

    def test_catalog_recovers_name_panels_from_neck_and_bottom_fragments(self):
        uploads = (
            Path(__file__).parents[2]
            / "Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads"
        )
        cases = (
            ("product_3_5598a285a2.webp", "printed_band", 0.32, 0.76, 0.80),
            ("product_2_03f0763593.webp", "printed_band", 0.32, 0.76, 0.80),
            ("sober_bash_rkatsiteli_beloe_suhoe_14_5c76e13a55.webp", "dark_print", 0.48, 0.82, 0.65),
            ("soyuz_vino_mosavali_saperavi_suhoe_krasnoe_11_4664250a21.webp", "saturated_panel", 0.43, 0.96, 0.80),
        )
        for filename, method, min_top, max_bottom, min_width in cases:
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                detection = detect_label(image, catalog=True)
                left, top, right, bottom = detection.bbox
                self.assertEqual(detection.method, method)
                self.assertGreaterEqual(top, min_top)
                self.assertLessEqual(bottom, max_bottom)
                self.assertGreater(right - left, min_width)

        # These WebP packshots have transparent backgrounds. Keeping their
        # alpha channel lets the detector isolate the paper panel from the
        # white studio composite instead of seeing black-filled transparency.
        for filename, min_width in (
            ("stn_winery_trio_aligote_beloe_suhoe_13_6adf0dafd2.webp", 0.70),
            ("skalistyy_bereg_skb_blan_de_nuar_pino_nuar_beloe_ekstra_bryut_118_7f63c74351.webp", 0.55),
        ):
            with self.subTest(filename=filename):
                image = Image.open(uploads / filename)
                self.assertIn("A", image.getbands())
                detection = detect_label(image, catalog=True)
                self.assertGreater(detection.bbox[2] - detection.bbox[0], min_width)


if __name__ == "__main__":
    unittest.main()
