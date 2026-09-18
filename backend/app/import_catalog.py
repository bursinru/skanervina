"""Idempotent CSV/JSON import and resumable image indexing. No user uploads involved."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from PIL import Image, ImageOps
from psycopg.types.json import Jsonb
from .catalog import WineCatalog
from .database import connect, migrate
from .settings import settings


def load_catalog(path):
    if path.suffix.lower() != '.json':
        return WineCatalog.from_csv(path, settings.image_base_url)
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = raw.get('data', raw.get('wines', [])) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError('JSON must be an array or {"data": [...]} / {"wines": [...]}')
    # Supports canonical card JSON and Strapi attributes wrapping those fields.
    mapped = []
    for entry in rows:
        row = entry.get('attributes', entry)
        if 'Slug' in row:
            mapped.append(row)
            continue
        mapped.append({
            'Slug': row.get('slug'), 'Название вина': row.get('name'),
            'Винодельня': row.get('winery'), 'Категория': row.get('category'),
            'Регион': row.get('region'), 'Сорт винограда': ', '.join(row.get('grapes') or []),
            'Описание': row.get('description'), 'Название фото': row.get('image_name'),
            **{'svoe_vino_' + key: row.get(key) for key in ('image_url', 'public_rating', 'quality_rating', 'color', 'temperature', 'alcohol', 'region_image_url', 'grape_image_url', 'source_url')},
            'svoe_vino_dishes_json': json.dumps(row.get('dishes', [])),
            'svoe_vino_dish_image_urls_json': json.dumps(row.get('dish_image_urls', [])),
        })
    result = WineCatalog.from_rows(mapped, settings.image_base_url)
    if not result.size:
        raise ValueError('No usable wines. JSON requires slug and name fields; see README.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=settings.catalog_csv)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error('--batch-size must be between 1 and 64')
    migrate()
    catalog = load_catalog(args.catalog)
    wines = list(catalog._wines.values())
    with connect() as db:
        for wine in wines:
            db.execute('DELETE FROM wine_embeddings WHERE slug = %s AND EXISTS (SELECT 1 FROM wines WHERE slug = %s AND image_name <> %s)', (wine.slug, wine.slug, wine.image_name))
            db.execute('INSERT INTO wines (slug, card, image_name) VALUES (%s,%s,%s) ON CONFLICT (slug) DO UPDATE SET card=excluded.card, image_name=excluded.image_name',
                       (wine.slug, Jsonb(wine.to_card()), wine.image_name))
    print(json.dumps({'imported': len(wines)}), flush=True)
    if not args.index:
        return
    if not args.images or not args.images.is_dir():
        raise ValueError('--images must point to the local catalog uploads directory')
    from .vision import ImageEncoder, MODEL_ID
    encoder = ImageEncoder()
    pending, skipped, missing = [], 0, []
    root = args.images.resolve()
    with connect() as db:
        existing = {r['slug']: r for r in db.execute('SELECT slug, image_hash, model FROM wine_embeddings')}
    for wine in wines:
        # Strapi stores hashed upload filenames; CSV often contains the original display name.
        # Optional extra reference: uploads/<slug>.* is preferred over the CDN filename.
        filename = unquote(Path(urlparse(wine.direct_image_url or '').path).name) or wine.image_name
        files = []
        primary = (root / filename).resolve()
        if primary.is_relative_to(root) and primary.is_file():
            files.append(primary)
        for extra in sorted(root.glob(f'{wine.slug}.*')):
            extra = extra.resolve()
            if extra.is_relative_to(root) and extra.is_file() and extra not in files:
                files.append(extra)
        if not files:
            missing.append(wine.slug)
            continue
        path = files[-1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = existing.get(wine.slug, {})
        if previous.get('image_hash') == digest and previous.get('model') == MODEL_ID:
            skipped += 1
            continue
        pending.append((wine.slug, path, digest))
    processed, failed = 0, []
    for offset in range(0, len(pending), args.batch_size):
        batch, images = [], []
        for item in pending[offset:offset + args.batch_size]:
            try:
                with Image.open(item[1]) as im:
                    images.append(ImageOps.exif_transpose(im).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
                batch.append(item)
            except Exception:
                failed.append(item[0])
        if not images:
            continue
        vectors = encoder.encode(images)
        with connect() as db:
            for (slug, _, digest), vector in zip(batch, vectors):
                db.execute('''INSERT INTO wine_embeddings (slug, model, image_hash, embedding) VALUES (%s,%s,%s,%s::vector)
                    ON CONFLICT (slug) DO UPDATE SET model=excluded.model, image_hash=excluded.image_hash, embedding=excluded.embedding, updated_at=now()''',
                           (slug, MODEL_ID, digest, str(vector)))
        processed += len(batch)
        print(json.dumps({'indexed': processed, 'remaining': len(pending) - offset - len(batch)}), flush=True)
    report = {'indexed': processed, 'unchanged': skipped, 'missing': missing, 'failed': failed}
    report_path = Path(__file__).resolve().parents[1] / 'data' / 'index-report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'indexed': processed, 'unchanged': skipped, 'missing_count': len(missing), 'failed_count': len(failed), 'report': str(report_path)}), flush=True)


if __name__ == '__main__':
    main()
