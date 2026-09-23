import os
import unittest
from unittest.mock import patch, Mock
from io import BytesIO
from PIL import Image
from app.catalog import WineCatalog
from app.recognition import Recognizer
from app.settings import settings


class VisualDecisionTests(unittest.TestCase):
    def test_glare_fragment_uses_front_design_without_auto_match(self):
        from app.label_detection import LabelDetection
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}]
        fragment = LabelDetection((.46, .22, .53, .44), .9)
        with patch('app.recognition.detect_label', return_value=fragment):
            result = self.recognizer.recognize(self.photo, ocr_enabled=True)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['recognition']['query_view'], 'front_design')
        self.assertIn('tiny_region', result['recognition']['crop_quality']['reasons'])
        self.assertEqual(self.recognizer.visual.search.call_count, 2)
        self.assertLess(self.recognizer.visual.search.call_args_list[0].args[0].width, 50)
        self.assertEqual(self.recognizer.visual.search.call_args_list[1].args[0].size, (50, 50))
        self.recognizer._ocr.assert_called_once()

    def setUp(self):
        from app.label_detection import LabelDetection
        # Ranking tests isolate recognition from the detector on a blank image.
        detector = patch('app.recognition.detect_label', return_value=LabelDetection((.2, .3, .8, .9), .8))
        detector.start()
        self.addCleanup(detector.stop)
        self.catalog = WineCatalog.from_rows([
            {'Slug': 'a', 'Название вина': 'Alpha'},
            {'Slug': 'b', 'Название вина': 'Beta'},
        ], 'https://example.com/')
        with patch.dict(os.environ, {'CV_ENABLED': 'false'}):
            self.recognizer = Recognizer(self.catalog, settings)
        self.recognizer.visual = Mock()
        self.recognizer._ocr = Mock(return_value=('', 'ok'))
        output = BytesIO()
        Image.new('RGB', (50, 50), 'white').save(output, format='PNG')
        self.photo = output.getvalue()

    def test_match_from_seventy_five_percent(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .76}, {'slug': 'b', 'score': .70}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'a')
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .74}, {'slug': 'b', 'score': .71}]
        self.assertEqual(self.recognizer.recognize(self.photo)['status'], 'uncertain')
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        self.assertEqual(self.recognizer.recognize(self.photo)['slug'], 'a')

    def test_script_automatically_searches_inside_primary_bottle(self):
        self.recognizer._bottle_detector = Mock()
        self.recognizer._bottle_detector.detect.return_value = {
            'available': True,
            'count': 2,
            'primary_box': [.2, .1, .8, .9],
            'candidates': [],
            'needs_selection': True,
        }
        self.recognizer.visual.search.return_value = []
        with patch.dict(os.environ, {'CV_ENABLED': 'true'}):
            result = self.recognizer.recognize(self.photo, mode='image_full')
        self.assertEqual(self.recognizer.visual.search.call_args.args[0].size, (34, 44))
        self.assertEqual(result['recognition']['bottle_detection']['count'], 2)
        self.assertEqual(result['recognition']['bottle_detection']['selection'], 'automatic')
        self.assertAlmostEqual(result['recognition']['label_detection']['bbox'][0], 0.296, places=3)

    def test_web_selected_bottle_skips_automatic_instance_selection(self):
        self.recognizer._bottle_detector = Mock()
        self.recognizer.visual.search.return_value = []
        with patch.dict(os.environ, {'CV_ENABLED': 'true'}):
            result = self.recognizer.recognize(
                self.photo, mode='image_full', bottle_box=[.2, .1, .8, .9]
            )
        self.recognizer._bottle_detector.detect.assert_not_called()
        self.assertEqual(result['recognition']['bottle_detection']['selection'], 'user')

    def test_full_bottle_leader_is_not_replaced_by_a_wrong_crop(self):
        self.recognizer.catalog = WineCatalog.from_rows([
            {'Slug': 'relicta', 'Название вина': 'Реликта', 'Винодельня': 'Реликта'},
            {'Slug': 'flamingo', 'Название вина': 'Фламинго', 'Винодельня': 'Николаев и сыновья'},
        ], 'https://example.com/')
        self.recognizer.visual.search.return_value = [
            {'slug': 'relicta', 'score': .829, 'full_score': .701, 'crop_score': .829, 'best_view': 'crop'},
            {'slug': 'flamingo', 'score': .914, 'full_score': .914, 'crop_score': .597, 'best_view': 'full'},
        ]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'flamingo')

    def test_strong_top_match_does_not_need_margin(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .871}, {'slug': 'b', 'score': .864}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'a')
        self.assertEqual(result['recognition']['compared_with'][0]['slug'], 'a')

    def test_same_winery_line_does_not_block_match(self):
        self.recognizer.catalog = WineCatalog.from_rows([
            {'Slug': 'sikory-sb-reserve', 'Название вина': 'Совиньон Блан. Семейный резерв', 'Винодельня': 'Имение Сикоры', 'Сорт винограда': 'Совиньон Блан'},
            {'Slug': 'sikory-sb', 'Название вина': 'Совиньон Блан Сикоры', 'Винодельня': 'Имение Сикоры', 'Сорт винограда': 'Совиньон Блан'},
            {'Slug': 'litav-sb', 'Название вина': 'Совиньон Блан', 'Винодельня': 'Литавщук', 'Сорт винограда': 'Совиньон Блан'},
        ], 'https://example.com/')
        self.recognizer.visual.search.return_value = [
            {'slug': 'sikory-sb-reserve', 'score': .838},
            {'slug': 'sikory-sb', 'score': .817},
            {'slug': 'litav-sb', 'score': .799},
        ]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'sikory-sb-reserve')
        self.assertTrue(result['recognition']['family_tie'])

    def test_clear_gap_matches_below_threshold(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .72}, {'slug': 'b', 'score': .60}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'a')

    def test_compare_views_can_show_catalog_label_crop(self):
        panel = Image.new('RGB', (80, 140), (12, 12, 14))
        for x in range(18, 62):
            for y in range(24, 122):
                panel.putpixel((x, y), (236, 224, 196))
        self.recognizer._load_catalog_image = Mock(return_value=panel)
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .9}]
        result = self.recognizer.recognize(self.photo, include_candidates=True)
        view = result['recognition']['compared_with'][0]
        self.assertEqual(view['indexed_views'], ['full', 'crop'])
        self.assertTrue(view['label_jpeg_base64'])
        hidden = self.recognizer.recognize(self.photo)
        self.assertIsNone(hidden['recognition']['compared_with'][0]['label_jpeg_base64'])

    def test_compare_slug_scores_without_changing_winner(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .9, 'full_score': .9, 'crop_score': .81, 'best_view': 'full'}]
        self.recognizer.visual.score_slug.return_value = {
            'slug': 'b', 'score': .71, 'full_score': .64, 'crop_score': .71, 'best_view': 'crop', 'image_hash': 'x',
        }
        result = self.recognizer.recognize(self.photo, include_candidates=True, compare_slug='b')
        self.assertEqual(result['slug'], 'a')
        probe = result['recognition']['probe']
        self.assertEqual(probe['slug'], 'b')
        self.assertAlmostEqual(probe['crop_score'], .71)
        skipped = self.recognizer.recognize(self.photo, include_candidates=True)
        self.assertNotIn('probe', skipped['recognition'])
        unknown = self.recognizer.recognize(self.photo, include_candidates=True, compare_slug='missing-wine')
        self.assertTrue(unknown['recognition']['probe']['missing'])

    def test_combined_mode_skips_ocr_by_default(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['slug'], 'a')
        self.recognizer._ocr.assert_not_called()
        self.assertEqual(result['recognition']['ocr'], 'skipped')
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector')

    def test_combined_mode_runs_ocr_when_enabled(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .80}, {'slug': 'b', 'score': .79}]
        with patch.dict(os.environ, {'CV_OCR_ENABLED': 'true'}):
            result = self.recognizer.recognize(self.photo)
        self.recognizer._ocr.assert_called_once()
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector+ocr')

    def test_ocr_can_be_forced_off(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .80}, {'slug': 'b', 'score': .79}]
        with patch.dict(os.environ, {'CV_OCR_ENABLED': 'true'}):
            result = self.recognizer.recognize(self.photo, ocr_enabled=False)
        self.recognizer._ocr.assert_not_called()
        self.assertEqual(result['recognition']['ocr'], 'skipped')
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector')

    def test_image_only_mode_does_not_run_ocr(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        result = self.recognizer.recognize(self.photo, mode='image_auto')
        self.assertEqual(result['slug'], 'a')
        self.recognizer._ocr.assert_not_called()
        self.assertEqual(self.recognizer.visual.search.call_count, 1)
        self.assertIn('label_detection', result['recognition'])

    def test_image_auto_runs_ocr_when_forced(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .80}, {'slug': 'b', 'score': .79}]
        result = self.recognizer.recognize(self.photo, mode='image_auto', ocr_enabled=True)
        self.recognizer._ocr.assert_called_once()
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector+ocr')

    def test_ocr_only_mode_does_not_run_visual_search(self):
        self.recognizer._ocr.return_value = ('Alpha Winery', 'ok')
        result = self.recognizer.recognize(self.photo, mode='ocr_auto')
        self.assertEqual(result['recognition']['method'], 'ocr+catalog')
        self.recognizer.visual.search.assert_not_called()
        self.assertGreaterEqual(result['recognition']['timings_ms']['ocr'], 0)

    def test_lookalikes_are_uncertain(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .80}, {'slug': 'b', 'score': .79}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['slug'], 'a')
        self.assertNotIn('wine', result)
        self.assertEqual([card['slug'] for card in result['lookalikes']], ['a', 'b'])
        self.assertEqual(result['lookalikes'][0]['label_score'], 0.80)
        self.assertEqual(result['lookalikes'][1]['label_score'], 0.79)
        self.assertEqual(result['ranking']['top5'][0]['slug'], 'a')
        self.assertGreaterEqual(result['recognition']['timings_ms']['total'], 0)

    def test_unknown_still_returns_visual_neighbors(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .4}, {'slug': 'b', 'score': .38}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('wine', result)
        self.assertEqual([card['slug'] for card in result['lookalikes']], ['a', 'b'])

    def test_unrelated_image_has_no_card(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .4}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('wine', result)
        self.assertEqual(result.get('slug'), 'a')
        self.assertGreater(result['ranking']['f1_top1'], -0.01)

    def test_empty_index_does_not_fall_back_to_ocr(self):
        self.recognizer.visual.search.return_value = []
        self.assertEqual(self.recognizer.recognize(self.photo)['recognition']['reason'], 'index_empty')
        self.recognizer._ocr.assert_not_called()

    def test_ocr_noise_cannot_become_a_confident_match(self):
        self.recognizer.visual = None
        self.recognizer._ocr.return_value = ('— ii 2023 №', 'ok')
        with patch.dict(os.environ, {'CV_ENABLED': 'false'}):
            result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('wine', result)


class ExtraGalleryTests(unittest.TestCase):
    def test_best_by_slug_keeps_highest_score(self):
        from app.vision import best_by_slug
        rows = [
            {'slug': 'a', 'score': 0.81},
            {'slug': 'a', 'score': 0.94},
            {'slug': 'b', 'score': 0.90},
            {'slug': 'c', 'score': 0.70},
        ]
        ranked = best_by_slug(rows, limit=2)
        self.assertEqual([item['slug'] for item in ranked], ['a', 'b'])
        self.assertEqual(ranked[0]['score'], 0.94)

    def test_split_full_crop_scores_bottle_and_label(self):
        import hashlib
        from app.vision import split_full_crop
        bottle = 'abc123'
        label = hashlib.sha256(b'crop:abc123').hexdigest()
        full, crop, winner = split_full_crop([
            {'image_hash': bottle, 'score': 0.71},
            {'image_hash': label, 'score': 0.88},
        ])
        self.assertEqual(full, 0.71)
        self.assertEqual(crop, 0.88)
        self.assertEqual(winner, 'crop')


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.catalog = WineCatalog.from_rows([
            {
                'Slug': 'white',
                'Название вина': 'Шардоне',
                'Винодельня': 'Скалистый берег',
                'svoe_vino_color': 'Белое сухое',
            },
            {
                'Slug': 'red',
                'Название вина': 'Каберне Фран',
                'Винодельня': 'Скалистый берег',
                'svoe_vino_color': 'Красное сухое',
            },
        ], 'https://example.com/')
        with patch.dict(os.environ, {'CV_ENABLED': 'false', 'CV_OCR_ENABLED': 'false'}):
            self.recognizer = Recognizer(self.catalog, settings)
        self.recognizer.visual = Mock()
        self.recognizer._ocr = Mock(return_value=('', 'ok'))
        output = BytesIO()
        Image.new('RGB', (80, 120), (240, 228, 200)).save(output, format='PNG')
        self.photo = output.getvalue()

    def test_wine_color_does_not_change_ranking(self):
        self.recognizer.visual.search.return_value = [
            {'slug': 'red', 'score': 0.783},
            {'slug': 'white', 'score': 0.782},
        ]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['slug'], 'red')
        self.assertEqual(result['recognition']['color_delta'], 0)
        self.assertFalse(result['recognition']['color_enabled'])
        self.assertEqual(result['ranking']['top5'][1]['slug'], 'white')
        self.assertEqual(result['ranking']['top5'][1]['color_delta'], 0)

    def test_ocr_adds_percentage_points_when_enabled(self):
        self.recognizer.visual.search.return_value = [
            {'slug': 'red', 'score': 0.80},
            {'slug': 'white', 'score': 0.79},
        ]
        self.recognizer._ocr.return_value = ('Скалистый берег Шардоне', 'ok')
        with patch.dict(os.environ, {'CV_OCR_ENABLED': 'true'}):
            result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['slug'], 'white')
        self.assertGreater(result['recognition']['ocr_delta'], 0)
        self.assertGreater(result['recognition']['ocr_delta'], result['ranking']['top5'][1]['ocr_delta'])
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector+ocr')
