import os
import unittest
from unittest.mock import patch, Mock
from io import BytesIO
from PIL import Image
from app.catalog import WineCatalog
from app.recognition import Recognizer
from app.settings import settings


class VisualDecisionTests(unittest.TestCase):
    def setUp(self):
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

    def test_match_from_eighty_percent(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .81}, {'slug': 'b', 'score': .70}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['wine']['slug'], 'a')
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        self.assertEqual(self.recognizer.recognize(self.photo)['slug'], 'a')

    def test_combined_mode_skips_ocr_by_default(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['slug'], 'a')
        self.recognizer._ocr.assert_not_called()
        self.assertEqual(result['recognition']['ocr'], 'skipped')
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector')

    def test_combined_mode_runs_ocr_when_enabled(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .94}]
        with patch.dict(os.environ, {'CV_OCR_ENABLED': 'true'}):
            result = self.recognizer.recognize(self.photo)
        self.recognizer._ocr.assert_called_once()
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['recognition']['method'], 'siglip2+pgvector+ocr')

    def test_image_only_mode_does_not_run_ocr(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        result = self.recognizer.recognize(self.photo, mode='image_auto')
        self.assertEqual(result['slug'], 'a')
        self.recognizer._ocr.assert_not_called()
        self.assertIn('label_detection', result['recognition'])

    def test_ocr_only_mode_does_not_run_visual_search(self):
        self.recognizer._ocr.return_value = ('Alpha Winery', 'ok')
        result = self.recognizer.recognize(self.photo, mode='ocr_auto')
        self.assertEqual(result['recognition']['method'], 'ocr+catalog')
        self.recognizer.visual.search.assert_not_called()
        self.assertGreaterEqual(result['recognition']['timings_ms']['ocr'], 0)

    def test_lookalikes_are_uncertain(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .94}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['slug'], 'a')
        self.assertNotIn('wine', result)
        self.assertEqual([card['slug'] for card in result['lookalikes']], ['a', 'b'])
        self.assertEqual(result['lookalikes'][0]['label_score'], 0.95)
        self.assertEqual(result['lookalikes'][1]['label_score'], 0.94)
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
