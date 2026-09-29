"""SigLIP 2 image embeddings and PostgreSQL cosine retrieval."""
import hashlib
import logging
import os
from collections import defaultdict
from threading import Lock
from PIL import Image, ImageOps

MODEL_ID = 'google/siglip2-base-patch16-224'
MODEL_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
# The whole frame is also scored at 384 px, where small label print survives.
# Averaging it with the 224 label/frame score raised exact Top-1 on 205 labelled
# photos (65 shop/hand shots, 140 web photos) from 60.5% to 67.3%.
SECONDARY_MODEL_ID = 'google/siglip2-base-patch16-384'
SECONDARY_MODEL_REVISION = 'f775b65a79762255128c981547af89addcfe0f88'
SECONDARY_WEIGHT = 0.5


def secondary_enabled():
    return os.getenv('CV_SECONDARY_MODEL', 'true').lower() == 'true'


def indexed_models():
    return [MODEL_ID, SECONDARY_MODEL_ID] if secondary_enabled() else [MODEL_ID]


def best_by_slug(rows, limit=5):
    best = {}
    for row in rows:
        slug = row['slug']
        if slug not in best or row['score'] > best[slug]['score']:
            best[slug] = row
    return sorted(best.values(), key=lambda item: item['score'], reverse=True)[:limit]


def merge_query_views(crop_rows, full_rows, limit=8):
    """Keep the stronger of a label-crop query and a full-bottle query per slug."""

    return best_by_slug(list(crop_rows) + list(full_rows), limit)


def split_full_crop(rows):
    """Bottle photos vs catalog label crops. Crops are hashed as sha256('crop:' + file digest)."""

    hashes = {row['image_hash']: float(row['score']) for row in rows if row.get('image_hash') is not None}
    crop_ids = {
        hashlib.sha256(f'crop:{image_hash}'.encode()).hexdigest()
        for image_hash in hashes
    } & set(hashes)
    full = [score for image_hash, score in hashes.items() if image_hash not in crop_ids]
    crop = [score for image_hash, score in hashes.items() if image_hash in crop_ids]
    full_score = max(full) if full else None
    crop_score = max(crop) if crop else None
    if crop_score is None and full_score is None:
        best_view = None
    elif crop_score is None:
        best_view = 'full'
    elif full_score is None:
        best_view = 'crop'
    else:
        best_view = 'crop' if crop_score >= full_score else 'full'
    return full_score, crop_score, best_view


def fuse_scores(primary_rows, secondary_rows, weight=SECONDARY_WEIGHT, limit=8):
    """Average the best 224 and 384 gallery score of each candidate wine.

    primary_rows already hold the best score over the label and frame queries.
    A wine missing from one index keeps its other score for both terms.
    """

    primary = defaultdict(list)
    for row in primary_rows:
        primary[row['slug']].append(row)
    secondary = defaultdict(float)
    for row in secondary_rows:
        secondary[row['slug']] = max(secondary.get(row['slug'], -1.0), float(row['score']))
    fused = []
    for slug in set(primary) | set(secondary):
        rows = primary.get(slug, [])
        best = max(rows, key=lambda row: float(row['score'])) if rows else None
        primary_score = float(best['score']) if best else secondary[slug]
        secondary_score = secondary.get(slug, primary_score)
        full_score, crop_score, best_view = split_full_crop(rows)
        fused.append({
            'slug': slug,
            'image_hash': best['image_hash'] if best else None,
            'score': (1 - weight) * primary_score + weight * secondary_score,
            'primary_score': round(primary_score, 4),
            'secondary_score': round(secondary_score, 4),
            'full_score': None if full_score is None else round(full_score, 4),
            'crop_score': None if crop_score is None else round(crop_score, 4),
            'best_view': best_view,
        })
    fused.sort(key=lambda item: item['score'], reverse=True)
    return fused[:limit]


class ImageEncoder:
    def __init__(self, model_id=MODEL_ID, revision=MODEL_REVISION):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.torch = torch
        self.lock = Lock()
        self.model_id = model_id
        self.device = os.getenv('CV_DEVICE', 'cpu')
        torch.set_num_threads(int(os.getenv('CV_THREADS', '4')))
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = SiglipVisionModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.size = int(self.processor.size.get('height', 224))

    def encode(self, images):
        # Match the pinned processor's bilinear resize before NumPy conversion.
        # This avoids costly full-resolution arrays for phone photos on CPU.
        size = (self.size, self.size)
        prepared = [ImageOps.exif_transpose(image).convert('RGB').resize(size, Image.Resampling.BILINEAR) for image in images]
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=prepared, return_tensors='pt').to(self.device)
            features = self.model(**inputs).pooler_output
            features = self.torch.nn.functional.normalize(features, p=2, dim=-1)
            return features.cpu().tolist()


class VisualSearch:
    def __init__(self):
        self.encoder = ImageEncoder()
        self.secondary = None
        if secondary_enabled():
            try:
                self.secondary = ImageEncoder(SECONDARY_MODEL_ID, SECONDARY_MODEL_REVISION)
            except Exception:
                logging.exception('Secondary encoder unavailable; using %s only', MODEL_ID)

    @property
    def secondary_ready(self):
        return self.secondary is not None

    @staticmethod
    def _nearest(db, vector, model, limit):
        # The model is a code constant, inlined so the per-model partial HNSW index applies.
        return db.execute(f'''
            SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
            FROM wine_embeddings WHERE model = '{model}'
            ORDER BY embedding <=> %s::vector LIMIT %s
        ''', (str(vector), str(vector), max(limit * 8, 16))).fetchall()

    @staticmethod
    def _slug_rows(db, vectors, model, slugs):
        """Every gallery row of the given wines, scored by the best of the query vectors."""
        scores = ', '.join('1 - (embedding <=> %s::vector)' for _ in vectors)
        return db.execute(f'''
            SELECT slug, image_hash, GREATEST({scores}) AS score
            FROM wine_embeddings WHERE model = %s AND slug = ANY(%s)
        ''', (*[str(vector) for vector in vectors], model, list(slugs))).fetchall()

    def search_combined(self, label_image, full_image, limit=8):
        """224 label + frame queries fused with a 384 frame query, scored exactly per wine."""
        from .database import connect
        images = [label_image] if label_image is full_image else [label_image, full_image]
        primary_vectors = self.encoder.encode(images)
        secondary_vector = self.secondary.encode([full_image])[0]
        with connect() as db:
            slugs = []
            for vector in primary_vectors:
                slugs += [row['slug'] for row in best_by_slug(self._nearest(db, vector, MODEL_ID, limit), limit)]
            slugs += [row['slug'] for row in best_by_slug(self._nearest(db, secondary_vector, SECONDARY_MODEL_ID, limit), limit)]
            slugs = list(dict.fromkeys(slugs))
            primary = self._slug_rows(db, primary_vectors, MODEL_ID, slugs)
            secondary = self._slug_rows(db, [secondary_vector], SECONDARY_MODEL_ID, slugs)
        return fuse_scores(primary, secondary, limit=limit)

    def search(self, image, limit=5):
        from .database import connect
        vector = self.encoder.encode([image])[0]
        with connect() as db:
            ranked = best_by_slug(self._nearest(db, vector, MODEL_ID, limit), limit)
            slugs = [item['slug'] for item in ranked]
            detailed = self._slug_rows(db, [vector], MODEL_ID, slugs) if slugs else []
        grouped = defaultdict(list)
        for row in detailed:
            grouped[row['slug']].append(row)
        scored = []
        for item in ranked:
            full_score, crop_score, best_view = split_full_crop(grouped.get(item['slug'], []))
            scored.append({
                **item,
                'full_score': None if full_score is None else round(full_score, 4),
                'crop_score': None if crop_score is None else round(crop_score, 4),
                'best_view': best_view,
            })
        return scored

    def score_slug(self, image, slug):
        """Score one catalog wine against the query crop, even if ANN missed it."""
        from .database import connect
        if not slug:
            return None
        vector = self.encoder.encode([image])[0]
        with connect() as db:
            detailed = db.execute(
                '''
                SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
                FROM wine_embeddings WHERE model = %s AND slug = %s
                ''',
                (str(vector), MODEL_ID, slug),
            ).fetchall()
        if not detailed:
            return None
        full_score, crop_score, best_view = split_full_crop(detailed)
        best = max(detailed, key=lambda row: float(row['score']))
        return {
            **best,
            'full_score': None if full_score is None else round(full_score, 4),
            'crop_score': None if crop_score is None else round(crop_score, 4),
            'best_view': best_view,
        }

    def score_grape_views(self, image, hashes_by_slug):
        """Cosine of a grape-line crop against the stored line of each sibling."""

        from .database import connect
        from .grape_lines import grape_model

        if not hashes_by_slug:
            return {}
        vectors = [(MODEL_ID, self.encoder.encode([image])[0])]
        if self.secondary is not None:
            vectors.append((SECONDARY_MODEL_ID, self.secondary.encode([image])[0]))
        found = {slug: [] for slug in hashes_by_slug}
        with connect() as db:
            for model, vector in vectors:
                rows = db.execute(
                    '''
                    SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
                    FROM wine_embeddings
                    WHERE model = %s AND slug = ANY(%s)
                    ''',
                    (str(vector), grape_model(model), list(hashes_by_slug)),
                ).fetchall()
                for row in rows:
                    if row['image_hash'] == hashes_by_slug.get(row['slug']):
                        found[row['slug']].append(float(row['score']))
        return {
            slug: sum(values) / len(values)
            for slug, values in found.items()
            if values
        }
