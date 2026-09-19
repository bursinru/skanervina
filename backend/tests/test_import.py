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

    def test_extra_files_only_images_in_slug_folder(self):
        from app.import_catalog import extra_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'massandra-muskatel'
            folder.mkdir()
            (folder / 'front.jpg').write_bytes(b'x')
            (folder / 'notes.txt').write_text('no')
            (root / 'other.jpg').write_bytes(b'x')
            names = [path.name for path in extra_files(root, 'massandra-muskatel')]
            self.assertEqual(names, ['front.jpg'])
            self.assertEqual(extra_files(root, '../etc'), [])

    def test_crop_view_digest_is_stable_and_distinct(self):
        from app.import_catalog import crop_view_digest
        self.assertEqual(crop_view_digest('abc'), crop_view_digest('abc'))
        self.assertNotEqual(crop_view_digest('abc'), crop_view_digest('def'))
        self.assertNotEqual(crop_view_digest('abc'), 'abc')

    def test_unsupported_schema_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.json'
            path.write_text('{"data": [{"other": "value"}]}')
            with self.assertRaises(ValueError):
                load_catalog(path)
