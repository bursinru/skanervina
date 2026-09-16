import json
import tempfile
import unittest
from pathlib import Path
from app.import_catalog import load_catalog


class ImportTests(unittest.TestCase):
    def test_strapi_wrapped_card(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.json'
            path.write_text(json.dumps({'data': [{'id': 1, 'attributes': {
                'slug': 'test-wine', 'name': 'Test Wine', 'winery': 'Example',
                'grapes': ['Шардоне'], 'public_rating': 4.2,
                'image_name': 'bottle.webp', 'dishes': ['Рыба'],
            }}]}))
            catalog = load_catalog(path)
            self.assertEqual(catalog.size, 1)
            self.assertEqual(catalog.get('test-wine').to_card()['public_rating'], 4.2)
            self.assertEqual(catalog.get('test-wine').to_card()['dishes'], ['Рыба'])

    def test_unsupported_schema_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.json'
            path.write_text('{"data": [{"other": "value"}]}')
            with self.assertRaises(ValueError):
                load_catalog(path)
