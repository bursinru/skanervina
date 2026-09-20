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
from .label_detection import crop_label, detect_label

EXTRA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
CROP_AREA_MAX = 0.92

EXTRA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def extra_files(root: Path, slug: str) -> list:
    folder = (root / slug).resolve()
    if not root.exists() or not folder.is_dir() or not folder.is_relative_to(root.resolve()):
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in EXTRA_SUFFIXES
    )


def crop_view_digest(file_digest: str) -> str:
    return hashlib.sha256(f'crop:{file_digest}'.encode()).hexdigest()


def label_crop_if_useful(image: Image.Image):
    detection = detect_label(image, catalog=True)
    left, top, right, bottom = detection.bbox
    area = (right - left) * (bottom - top)
    if area >= CROP_AREA_MAX:
        return None
    cropped = crop_label(image, detection)
    if cropped.width < 32 or cropped.height < 32:
        return None
    return cropped


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
    parser.add_argument('--extra-images', type=Path)
    parser.add_argument('--only-slug', type=str)
    parser.add_argument('--slug-prefix', type=str)
    parser.add_argument('--crops', action='store_true', default=True)
    parser.add_argument('--no-crops', action='store_false', dest='crops')
    parser.add_argument('--force-crops', action='store_true')
    parser.add_argument('--save-crops', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error('--batch-size must be between 1 and 64')
    migrate()
    catalog = load_catalog(args.catalog)
    wines = list(catalog._wines.values())
    if args.only_slug:
        wine = catalog.get(args.only_slug)
        if not wine:
            raise ValueError(f'Unknown slug: {args.only_slug}')
        wines = [wine]
    if args.slug_prefix:
        wines = [wine for wine in wines if wine.slug.startswith(args.slug_prefix)]
        if not wines:
            raise ValueError(f'No wines with slug prefix: {args.slug_prefix}')
    if not args.only_slug and not args.slug_prefix:
        with connect() as db:
            for wine in wines:
                db.execute('INSERT INTO wines (slug, card, image_name) VALUES (%s,%s,%s) ON CONFLICT (slug) DO UPDATE SET card=excluded.card, image_name=excluded.image_name',
                           (wine.slug, Jsonb(wine.to_card()), wine.image_name))
        print(json.dumps({'imported': len(wines)}), flush=True)
    if not args.index:
        return
    extra_root = args.extra_images.resolve() if args.extra_images else None
    if extra_root and not extra_root.is_dir():
        raise ValueError('--extra-images must be a directory')
    if not extra_root and (not args.images or not args.images.is_dir()):
        raise ValueError('--images must point to the local catalog uploads directory')
    from .vision import ImageEncoder, MODEL_ID
    print(json.dumps({'loading_encoder': True}), flush=True)
    encoder = ImageEncoder()
    print(json.dumps({'encoder_ready': True, 'model': MODEL_ID}), flush=True)
    pending, skipped, missing = [], 0, []
    root = args.images.resolve() if args.images and args.images.is_dir() else extra_root
    with connect() as db:
        existing = {(r['slug'], r['image_hash']): r for r in db.execute('SELECT slug, image_hash, model FROM wine_embeddings')}
    scanned = 0
    for wine in wines:
        scanned += 1
        if scanned == 1 or scanned % 200 == 0:
            print(json.dumps({'scanning': scanned, 'total': len(wines), 'pending': len(pending)}), flush=True)
        filename = unquote(Path(urlparse(wine.direct_image_url or '').path).name) or wine.image_name
        files = []
        if args.images and args.images.is_dir():
            primary = (root / filename).resolve()
            if primary.is_relative_to(root) and primary.is_file():
                files.append(primary)
            for extra in sorted(root.glob(f'{wine.slug}.*')):
                extra = extra.resolve()
                if extra.is_relative_to(root) and extra.is_file() and extra not in files:
                    files.append(extra)
        if extra_root:
            files.extend(path for path in extra_files(extra_root, wine.slug) if path not in files)
        if not files:
            missing.append(wine.slug)
            continue
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            views = [('full', digest)]
            if args.crops:
                views.append(('crop', crop_view_digest(digest)))
            for kind, view_digest in views:
                previous = existing.get((wine.slug, view_digest), {})
                if previous.get('model') == MODEL_ID and not (args.force_crops and kind == 'crop'):
                    skipped += 1
                    continue
                pending.append((wine.slug, path, view_digest, kind))
    print(json.dumps({'pending': len(pending), 'skipped': skipped, 'missing': len(missing), 'force_crops': args.force_crops}), flush=True)
    processed, failed = 0, []
    for offset in range(0, len(pending), args.batch_size):
        batch, images = [], []
        for item in pending[offset:offset + args.batch_size]:
            try:
                with Image.open(item[1]) as im:
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                if item[3] == 'crop':
                    cropped = label_crop_if_useful(rgb)
                    if cropped is None:
                        skipped += 1
                        continue
                    rgb = cropped
                    if args.save_crops:
                        try:
                            args.save_crops.mkdir(parents=True, exist_ok=True)
                            rgb.convert('RGB').save(args.save_crops / f'{item[0]}.jpg', quality=92)
                        except Exception as exc:
                            print(json.dumps({'save_failed': item[0], 'error': str(exc)[:200]}), flush=True)
                images.append(rgb)
                batch.append(item)
            except Exception as exc:
                failed.append(item[0])
                print(json.dumps({'failed': item[0], 'kind': item[3], 'error': str(exc)[:200]}), flush=True)
        if not images:
            continue
        vectors = encoder.encode(images)
        with connect() as db:
            for (slug, _, digest, _), vector in zip(batch, vectors):
                db.execute('''INSERT INTO wine_embeddings (slug, model, image_hash, embedding) VALUES (%s,%s,%s,%s::vector)
                    ON CONFLICT (slug, image_hash) DO UPDATE SET model=excluded.model, embedding=excluded.embedding, updated_at=now()''',
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
