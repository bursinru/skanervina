"""CPU/API benchmark: full photo vs manually marked label, visual vs OCR.

No official query ground truth is provided. Reference self-retrieval is a smoke
check only. Crop coordinates are fixed by looking at labels, not search scores.
Run with a CPU API already listening; no weights or catalogue are changed.
"""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time
from urllib.parse import unquote, urlparse

import httpx
from dotenv import dotenv_values
from PIL import Image, ImageOps
import pytesseract
from app.catalog import WineCatalog
from app.settings import settings
from app.recognition import Recognizer

CROPS = {
    '019c68d0.jpg': [.25, .35, .79, .83],
    '02eef911.webp': [.26, .19, .71, .79],
    '096ca74e.jpg': [.21, .24, .83, .73],
}
VISIBLE_LABELS = {
    '019c68d0.jpg': 'Табия, Пино Нуар, полусухое, 2025',
    '02eef911.webp': 'Массандра, Мускатель Массандра белый, 2023',
    '096ca74e.jpg': 'Aristov, Donum XXIV, брют, 2023',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:3000')
    parser.add_argument('--output', type=Path, default=Path('reports/recognition/cpu-label-ocr-optimized.json'))
    args = parser.parse_args()
    token = os.getenv('SCANNER_ADMIN_TOKEN') or dotenv_values('backend/.env.local').get('SCANNER_ADMIN_TOKEN')
    if not token:
        raise SystemExit('Set SCANNER_ADMIN_TOKEN')
    catalog = WineCatalog.from_csv(settings.catalog_csv, settings.image_base_url)
    os.environ['CV_ENABLED'] = 'false'
    ocr_service = Recognizer(catalog, settings)
    report = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'platform': platform.platform(), 'device': 'CPU (API must be started with CV_DEVICE=cpu)', 'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() if platform.system() == 'Darwin' else platform.processor(), 'notes': ['Manual label rectangles fixed before running search.', 'Query slugs are withheld; no real-world accuracy is calculated.', 'Reference self-retrieval is not an independent accuracy test.', 'Sequential requests, no concurrent browser scans; model is already loaded.'], 'queries': [], 'references': []}
    with httpx.Client(base_url=args.url, timeout=90, headers={'X-Scanner-Admin': token}) as client:
        health = client.get('/healthz'); health.raise_for_status(); report['health'] = health.json()
        if report['health'].get('cv_device') != 'cpu':
            raise SystemExit('This benchmark requires the API to report cv_device=cpu')
        report['ocr_max_side'] = 1000
        def request(data, name, mime):
            started = time.perf_counter()
            response = client.post('/v1/recognize', files={'image': (name, data, mime)})
            response.raise_for_status()
            result = response.json()
            return {'wall_ms': round((time.perf_counter() - started) * 1000, 1), 'status': result['status'], 'slug': result.get('slug'), 'name': (result.get('wine') or {}).get('name'), **result['recognition']}
        warmup_path = Path('Датасет/eval/queries/019c68d0.jpg')
        report['warmup'] = request(warmup_path.read_bytes(), warmup_path.name, 'image/jpeg')
        report['repeats_per_variant'] = 3
        report['sources_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path('backend/app/recognition.py'), Path('backend/app/vision.py'), Path(__file__))}
        for name, rect in CROPS.items():
            path = Path('Датасет/eval/queries') / name
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                crop = image.crop(tuple(round(value * (image.width if i % 2 == 0 else image.height)) for i, value in enumerate(rect)))
            crop_buffer = BytesIO(); crop.save(crop_buffer, format='JPEG', quality=95)
            crop_dir = Path('backend/data/label-crops'); crop_dir.mkdir(parents=True, exist_ok=True)
            (crop_dir / (path.stem + '.jpg')).write_bytes(crop_buffer.getvalue())
            row = {'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'visible_text_manual': VISIBLE_LABELS[name], 'crop_normalized': rect, 'variants': {}}
            for variant, picture in [('full', image), ('label', crop)]:
                data = path.read_bytes() if variant == 'full' else crop_buffer.getvalue()
                mime = 'image/webp' if variant == 'full' and path.suffix == '.webp' else 'image/jpeg'
                samples = [request(data, name, mime) for _ in range(3)]
                result = {**samples[-1], 'wall_ms': round(median(sample['wall_ms'] for sample in samples), 1), 'timings_ms': {key: round(median(sample['timings_ms'][key] for sample in samples), 1) for key in ('total', 'visual', 'ocr')}, 'samples': samples}
                picture.thumbnail((1000, 1000))
                result['ocr_comparison'] = {}
                for psm in (6, 11):
                    started = time.perf_counter()
                    text = pytesseract.image_to_string(picture, lang='rus+eng', config=f'--psm {psm}', timeout=20).strip()
                    elapsed = (time.perf_counter() - started) * 1000
                    matches = ocr_service._text_matches(text)
                    result['ocr_comparison'][str(psm)] = {'text': text, 'ocr_ms': round(elapsed, 1), 'top1_slug': matches[0].wine.slug if matches else None, 'top1_score': round(matches[0].score, 4) if matches else 0}
                row['variants'][variant] = result
                print(json.dumps({'file': name, 'variant': variant, 'ms': result['wall_ms'], 'name': result['name'], 'similarity': result.get('similarity')}, ensure_ascii=False), flush=True)
            report['queries'].append(row)
        images = Path('Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads')
        for old in json.loads(Path('backend/data/catalog-self-check.json').read_text()):
            wine = catalog.get(old['expected'])
            image_path = images / (unquote(Path(urlparse(wine.direct_image_url or '').path).name) or wine.image_name)
            import mimetypes
            result = request(image_path.read_bytes(), image_path.name, mimetypes.guess_type(image_path.name)[0])
            report['references'].append({'expected': wine.slug, 'correct': result['slug'] == wine.slug, **result})
    for variant in ('full', 'label'):
        report[variant + '_summary'] = {key: round(median(row['variants'][variant][key] for row in report['queries']), 1) for key in ('wall_ms',)}
        report[variant + '_summary']['server_median_ms'] = round(median(row['variants'][variant]['timings_ms']['total'] for row in report['queries']), 1)
        report[variant + '_summary']['visual_median_ms'] = round(median(row['variants'][variant]['timings_ms']['visual'] for row in report['queries']), 1)
        report[variant + '_summary']['ocr_median_ms'] = round(median(row['variants'][variant]['timings_ms']['ocr'] for row in report['queries']), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'report': str(args.output), 'full': report['full_summary'], 'label': report['label_summary'], 'reference_correct': sum(row['correct'] for row in report['references'])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
