"""Paired query-crop check against the unchanged September 20 gallery."""
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from app.label_detection import detect_label, crop_label
from app.label_ocr import read_label, load_references, pipeline_version
from app.label_signals import blend_candidates
from app.import_catalog import load_catalog
from app.settings import settings
from app.vision import ImageEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', type=Path, default=Path('backend/data/ablation-2026-09-20'))
    parser.add_argument('--name', default='query-check')
    args = parser.parse_args()
    out = Path('reports/recognition/label-revision-2026-09-21')
    out.mkdir(parents=True, exist_ok=True)
    cache = args.gallery
    gallery = np.load(cache / 'gallery.npy')
    meta = json.loads((cache / 'gallery.json').read_text())
    slugs = list(dict.fromkeys(r['slug'] for r in meta['rows']))
    ids = {slug: i for i, slug in enumerate(slugs)}
    row_ids = np.array([ids[r['slug']] for r in meta['rows']])
    kinds = np.array([r['kind'] for r in meta['rows']])
    manifest = json.loads(Path('reports/recognition/ablation-2026-09-20/manifest.json').read_text())
    old = {r['id']: r for r in json.loads(Path('reports/recognition/ablation-2026-09-20/results.json').read_text())}
    manifest.append(dict(id='user', path='backend/tests/fixtures/riesling-table.png',
        expected_slug='skalistyy-bereg-skalistyy-bereg-risling-beloe-suhoe-113', subset='user', scene='scene'))
    encoder = None
    catalog = load_catalog(settings.catalog_csv)
    references = load_references('backend/data/catalog-ocr.json')
    query_cache = Path('backend/data/label-revision-query-cache')
    query_cache.mkdir(exist_ok=True)
    results = []
    for case in manifest:
        image = Image.open(case['path'])
        detection = detect_label(image)
        crop = crop_label(image, detection)
        key = hashlib.sha256(Path(case['path']).read_bytes() + pipeline_version().encode()).hexdigest()
        vector_path = query_cache / (key + '.npy')
        if vector_path.exists():
            vector = np.load(vector_path)
        else:
            if encoder is None:
                encoder = ImageEncoder()
            vector = encoder.encode([crop])[0]
            np.save(vector_path, vector)
        similarity = gallery @ np.asarray(vector, dtype=np.float32)
        views = {}
        for kind in ['full', 'crop']:
            scores = np.full(len(slugs), -1., dtype=np.float32)
            mask = kinds == kind
            np.maximum.at(scores, row_ids[mask], similarity[mask])
            views[kind] = scores
        scores = np.maximum(views['full'], views['crop'])
        candidates = [dict(slug=slugs[i], score=float(scores[i]), full_score=float(views['full'][i]),
                           crop_score=float(views['crop'][i])) for i in np.argsort(-scores, kind='stable')[:8]]
        text = read_label(crop)
        ranked = blend_candidates(candidates, catalog, {}, text, True)
        with_references = blend_candidates(candidates, catalog, {}, text, True, references)
        result = dict(id=case['id'], subset=case['subset'], scene=case['scene'],
            expected=case['expected_slug'], bbox=detection.bbox, method=detection.method, ocr=text,
            visual_top1=candidates[0]['slug'], ocr_top1=ranked[0]['slug'],
            reference_top1=with_references[0]['slug'], reference_count=len(references),
            visual_top5=[r['slug'] for r in candidates[:5]], candidates=with_references)
        if case['id'] in old:
            result['old_top1'] = old[case['id']]['modes']['label_both']['slug']
        if case['id'] == 'user':
            preview = crop.copy()
            preview.thumbnail((480, 640))
            preview.save(out / 'query-label.png')
            for candidate in candidates[:5]:
                row = next(r for r in meta['rows'] if r['slug'] == candidate['slug'])
                ref = Image.open(row['path'])
                preview = crop_label(ref, detect_label(ref, catalog=True))
                preview.thumbnail((480, 640))
                preview.save(out / (candidate['slug'] + '.png'))
        results.append(result)
        print(f"completed {len(results)}/{len(manifest)}", flush=True)
    (out / (args.name + '.json')).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for name, rows in [('reviewed', [r for r in results if r['subset'] == 'reviewed']),
                       ('scene', [r for r in results if r['subset'] == 'reviewed' and r['scene'] == 'scene'])]:
        print(name, len(rows), {field: sum(r[field] == r['expected'] for r in rows)
                               for field in ['old_top1', 'visual_top1', 'ocr_top1', 'reference_top1']})
    print(json.dumps(results[-1], ensure_ascii=False))


if __name__ == '__main__':
    main()
