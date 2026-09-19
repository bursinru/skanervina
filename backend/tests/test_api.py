import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app import storage
from app.catalog import WineCatalog
from app.recognition import Recognizer
from app.settings import settings


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(storage, 'DB_PATH', Path(self.temp.name) / 'test.sqlite3')
        self.patch.start()
        self.env = patch.dict(os.environ, {'DATABASE_URL': '', 'CV_ENABLED': 'false'})
        self.env.start()
        fixture = WineCatalog.from_rows([{'Slug': 'fanagoria-test', 'Название вина': 'Blanc de Blancs', 'Винодельня': 'Фанагория'}], 'https://example.com/')
        self.catalog_patch = patch('app.main.catalog', fixture)
        self.service_patch = patch('app.main.recognizer', Recognizer(fixture, settings))
        self.catalog_patch.start()
        self.service_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.service_patch.stop()
        self.catalog_patch.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_persistence_and_isolation(self):
        self.assertEqual(self.client.get('/v1/profile').json()['saved'], [])
        state = {'saved': [{'slug': 'test', 'name': 'Wine', 'winery': 'Winery'}], 'ratings': {'test': 4}}
        result = self.client.put('/v1/profile', json=state, headers={'X-Scanner-Client': 'web'})
        self.assertEqual(result.status_code, 200)
        with TestClient(app) as restarted:
            restarted.cookies.update(self.client.cookies)
            self.assertEqual(restarted.get('/v1/profile').json()['ratings'], {'test': 4})
        with TestClient(app) as stranger:
            self.assertEqual(stranger.get('/v1/profile').json()['ratings'], {})
        self.assertIn('HttpOnly', self.client.get('/v1/profile').headers['set-cookie'])

    def test_validation_and_cross_origin_write(self):
        self.assertEqual(self.client.put('/v1/profile', json={}).status_code, 403)
        self.assertEqual(self.client.put('/v1/profile', content=b'x' * 512001).status_code, 413)
        self.assertEqual(self.client.put('/v1/profile', json={'ratings': {'x': True}}, headers={'X-Scanner-Client': 'web'}).status_code, 422)
        self.assertEqual(self.client.put('/v1/profile', json={'ratings': {'x': 6}}, headers={'X-Scanner-Client': 'web'}).status_code, 422)
        self.assertEqual(self.client.post('/v1/recognize', files={'image': ('x.txt', b'test', 'text/plain')}).status_code, 415)
        result = self.client.post('/v1/recognize', files={'image': ('x.jpg', b'bad', 'image/jpeg')})
        self.assertEqual(result.json()['recognition']['reason'], 'invalid_image')

    def test_site_and_catalog(self):
        self.assertEqual(self.client.get('/scanner').status_code, 200)
        self.assertEqual(self.client.get('/scanner/fanagoria-test').status_code, 200)
        self.assertIn('profileEndpoint', self.client.get('/config.js').text)
        self.assertGreater(self.client.get('/healthz').json()['catalog_size'], 0)
        items = self.client.get('/v1/search', params={'q': 'Фанагория Blanc de Blancs'}).json()['items']
        self.assertTrue(items)
        self.assertEqual(self.client.get('/v1/catalog/' + items[0]['slug']).status_code, 200)

    def test_metrics_are_protected_and_debug_is_open(self):
        result = {'status': 'uncertain', 'slug': 'fanagoria-test', 'wine': {'slug': 'fanagoria-test', 'name': 'Wine', 'winery': 'Test'}, 'confidence': .8, 'recognition': {'similarity': .8, 'timings_ms': {'total': 123}}}
        with patch.dict(os.environ, {'SCANNER_ADMIN_TOKEN': 'test-secret'}), patch('app.main.recognize_upload', return_value=result):
            photo = {'image': ('test.jpg', b'photo', 'image/jpeg')}
            public = self.client.post('/v1/recognize', files=photo).json()
            self.assertIsNone(public['confidence'])
            self.assertNotIn('timings_ms', public['recognition'])
            self.assertEqual(public['slug'], 'fanagoria-test')
            self.assertEqual(self.client.post('/v1/recognize', files=photo, headers={'X-Scanner-Admin': 'wrong'}).status_code, 403)
            admin = self.client.post('/v1/recognize', files=photo, headers={'X-Scanner-Admin': 'test-secret'})
            self.assertEqual(admin.json()['recognition']['similarity'], .8)
            self.assertEqual(admin.headers['cache-control'], 'no-store')
            debug = self.client.post('/v1/recognize', files=photo, headers={'X-Scanner-Debug': '1'})
            self.assertEqual(debug.json()['recognition']['similarity'], .8)
            self.assertEqual(self.client.post('/v1/eval/predict', files=photo).json()['slug'], 'fanagoria-test')
