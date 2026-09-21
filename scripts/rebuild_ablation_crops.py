"""Re-encode catalog crops for an isolated comparison; do not touch PostgreSQL."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from app.label_ocr import pipeline_version
from app.vision import ImageEncoder


def main():
    source = Path('backend/data/ablation-2026-09-20')
    destination = Path('backend/data/label-revision-gallery')
    destination.mkdir(exist_ok=True)
    metadata = json.loads((source / 'gallery.json').read_text())
    previous = np.load(source / 'gallery.npy')
    reference_path = Path('backend/data/catalog-ocr.json')
    references = json.loads(reference_path.read_text())
    assert references['pipeline_version'] == pipeline_version(), 'Rebuild OCR/crops first'
    full = [(r, previous[i]) for i, r in enumerate(metadata['rows']) if r['kind'] == 'full']
    rows = [r for r, _ in full]
    vectors = [v for _, v in full]
    encoder = ImageEncoder()
    for offset in range(0, len(full), 16):
        deadline = time.monotonic() + 1800
        while True:
            references = json.loads(reference_path.read_text())
            pending = [r['slug'] for r, _ in full[offset:offset + 16]
                       if not references['entries'].get(r['slug'], {}).get('crop_path')]
            if not pending:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'Catalog crop generation incomplete: {pending}')
            time.sleep(2)
        batch_rows, images = [], []
        for row, _ in full[offset:offset + 16]:
            entry = references['entries'].get(row['slug'])
            assert entry and entry.get('crop_path'), row['slug']
            assert entry['image_sha256'] == row['sha256'], 'Catalog image changed'
            with Image.open(entry['crop_path']) as image:
                images.append(image.convert('RGB'))
            batch_rows.append(dict(row, kind='crop'))
        vectors.extend(encoder.encode(images))
        rows.extend(batch_rows)
        print(f"encoded {min(offset + 16, len(full))}/{len(full)}", flush=True)
    np.save(destination / 'gallery.npy', np.asarray(vectors, dtype=np.float32))
    metadata.update(rows=rows, version=pipeline_version(), note='Unchanged full vectors; refreshed crop vectors')
    (destination / 'gallery.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
