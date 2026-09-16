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

    def test_separated_visual_match(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .80}]
        self.assertEqual(self.recognizer.recognize(self.photo)['slug'], 'a')

    def test_lookalikes_are_uncertain(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .95}, {'slug': 'b', 'score': .94}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['slug'], 'a')
        self.assertNotIn('candidates', result['recognition'])
        self.assertGreaterEqual(result['recognition']['timings_ms']['total'], 0)

    def test_unrelated_image_has_no_card(self):
        self.recognizer.visual.search.return_value = [{'slug': 'a', 'score': .4}]
        result = self.recognizer.recognize(self.photo)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('wine', result)

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
