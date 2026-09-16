"""Black-box checks against the running Nuxt gateway; uses a new anonymous profile."""
import argparse
import json
import mimetypes
import time
from pathlib import Path
import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:3000')
parser.add_argument('--queries', type=Path, default=Path('Датасет/eval/queries'))
parser.add_argument('--output', type=Path, default=Path('backend/data/verification.json'))
args = parser.parse_args()
report = {}
with httpx.Client(base_url=args.url, timeout=60) as client:
    response = client.get('/healthz')
    response.raise_for_status()
    report['health'] = response.json()
    cards = client.get('/v1/search', params={'q': 'Фанагория Blanc de Blancs'}).json()['items']
    assert cards, 'No catalog search results'
    client.get('/v1/profile').raise_for_status()
    state = {'saved': [cards[0]], 'ratings': {cards[0]['slug']: 4}}
    client.put('/v1/profile', json=state, headers={'X-Scanner-Client': 'web'}).raise_for_status()
    assert client.get('/v1/profile').json()['ratings'] == state['ratings']
    with httpx.Client(base_url=args.url) as separate:
        assert separate.get('/v1/profile').json()['saved'] == []
    report['profile_persistence_and_isolation'] = 'passed'
    client.put('/v1/profile', json={'saved': [], 'ratings': {}}, headers={'X-Scanner-Client': 'web'}).raise_for_status()
    assert client.put('/v1/profile', json={}).status_code == 403
    assert client.post('/v1/recognize', files={'image': ('x.txt', b'invalid', 'text/plain')}).status_code == 415
    report['validation'] = 'passed'
    report['queries'] = []
    for path in sorted(args.queries.glob('*')):
        if not path.is_file():
            continue
        start = time.monotonic()
        response = client.post('/v1/recognize', files={'image': (path.name, path.read_bytes(), mimetypes.guess_type(path.name)[0])})
        response.raise_for_status()
        report['queries'].append({'file': path.name, 'seconds': round(time.monotonic() - start, 2), **response.json()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({'health': report['health'], 'checks': 'passed', 'queries': [{k: row.get(k) for k in ('file', 'seconds', 'status', 'slug')} for row in report['queries']], 'report': str(args.output)}, ensure_ascii=False, indent=2))
