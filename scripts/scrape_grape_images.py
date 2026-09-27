"""Collect grape-variety photos from vino-svoe.ru wine pages.

The site keeps one photo per grape variety. We open one public wine page per
variety found in the catalog and read the variety entity from the Nuxt payload.

Usage:
    PYTHONPATH=backend python scripts/scrape_grape_images.py [catalog.csv]
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from app.catalog import WineCatalog
from app.settings import settings

OUTPUT = Path(__file__).resolve().parents[1] / 'backend' / 'app' / 'grape_images.json'
PAYLOAD = re.compile(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.S)
DELAY_SECONDS = 0.8


def grape_entities(page: str) -> dict:
    match = PAYLOAD.search(page)
    if not match:
        return {}
    data = json.loads(match.group(1))

    def url_of(ref):
        value = data[ref] if isinstance(ref, int) else ref
        if isinstance(value, dict) and 'url' in value:
            value = data[value['url']]
        return value if isinstance(value, str) and value.startswith('/uploads/') else None

    found = {}
    for item in data:
        if isinstance(item, dict) and {'backgroundImage', 'image', 'name'} <= set(item):
            name = data[item['name']]
            image = url_of(item['image'])
            if isinstance(name, str) and image:
                found[name.strip().lower()] = image
    return found


def main():
    catalog_path = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.catalog_csv
    catalog = WineCatalog.from_csv(catalog_path, settings.image_base_url)
    known = json.loads(OUTPUT.read_text(encoding='utf-8')) if OUTPUT.exists() else {}
    pages = {}
    for wine in catalog:
        for grape in wine.grapes:
            key = grape.strip().lower()
            if key and key not in known and wine.source_url:
                pages.setdefault(key, wine.source_url)
    print(f'{len(known)} known, {len(pages)} to look up')
    for index, (grape, url) in enumerate(sorted(pages.items()), 1):
        if grape in known:
            continue
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'svoe-vino-scanner/0.1 (grape image index)'})
            with urllib.request.urlopen(request, timeout=20) as response:
                entities = grape_entities(response.read().decode('utf-8', 'replace'))
        except Exception as error:
            print(f'[{index}/{len(pages)}] {grape}: {error}')
            continue
        known.update({name: image for name, image in entities.items() if name not in known})
        print(f'[{index}/{len(pages)}] {grape}: {entities.get(grape, "—")}')
        time.sleep(DELAY_SECONDS)
    OUTPUT.write_text(json.dumps(dict(sorted(known.items())), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Saved {len(known)} varieties → {OUTPUT}')


if __name__ == '__main__':
    main()
