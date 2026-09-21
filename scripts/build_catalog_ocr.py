"""Resumable local catalog OCR. Run with PYTHONPATH=backend; no uploads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps
from app.import_catalog import load_catalog
from app.label_detection import detect_label, crop_label, label_rgb
from app.label_ocr import read_label, pipeline_version
from app.settings import settings


def process(item):
    slug, path, digest, languages, crop_dir = item
    try:
        with Image.open(path) as source:
            image = label_rgb(source)
        detection = detect_label(image, catalog=True)
        crop = crop_label(image, detection)
        crop_path = Path(crop_dir) / (Path(slug).name + '.png')
        crop.save(crop_path)
        text = read_label(crop, languages)
        return slug, dict(text=text, status="ok", image_sha256=digest,
                          bbox=detection.bbox, method=detection.method, crop_path=str(crop_path))
    except Exception as exc:
        return slug, dict(text="", status="failed", error=str(exc), image_sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads'))
    parser.add_argument('--output', type=Path, default=Path('backend/data/catalog-ocr.json'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--crops', type=Path, default=Path('backend/data/catalog-label-crops-v2'))
    args = parser.parse_args()
    version = pipeline_version() + ':' + settings.ocr_languages
    payload = dict(pipeline_version=pipeline_version(), language_version=version, entries={}, missing=[])
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old.get('language_version') == version:
            payload['entries'] = old['entries']
    pending = []
    for wine in load_catalog(settings.catalog_csv):
        filename = unquote(Path(urlparse(wine.direct_image_url or '').path).name) or wine.image_name
        path = args.images / Path(filename).name
        if not path.is_file():
            payload['missing'].append(wine.slug)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = payload['entries'].get(wine.slug, {})
        if previous.get('image_sha256') == digest and previous.get('status') == 'ok' and Path(previous.get('crop_path', '')).is_file():
            continue
        pending.append((wine.slug, str(path), digest, settings.ocr_languages, str(args.crops)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.crops.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        temp.replace(args.output)

    print(json.dumps(dict(pending=len(pending), missing=len(payload['missing']))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (slug, result) in enumerate(pool.map(process, pending), 1):
            payload['entries'][slug] = result
            if i % 50 == 0:
                save()
                print(json.dumps(dict(completed=i, total=len(pending))), flush=True)
    save()
    print(json.dumps(dict(entries=len(payload['entries']), failed=sum(e['status'] != 'ok' for e in payload['entries'].values()))), flush=True)


if __name__ == '__main__':
    main()
