"""SigLIP 2 image embeddings and PostgreSQL cosine retrieval."""
import hashlib
import os
from collections import defaultdict
from threading import Lock
from PIL import Image, ImageOps

DEFAULT_MODEL_ID = 'google/siglip2-base-patch16-224'
DEFAULT_MODEL_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
MODEL_ID = os.getenv('CV_MODEL_ID', DEFAULT_MODEL_ID)
MODEL_REVISION = DEFAULT_MODEL_REVISION if MODEL_ID == DEFAULT_MODEL_ID else (os.getenv('CV_MODEL_REVISION') or None)
# squash: legacy stretch to a square; pad: keep aspect ratio on a neutral square.
RESIZE_MODE = os.getenv('CV_RESIZE', 'squash')
PAD_COLOR = (255, 255, 255)


def encoder_cache_key():
    return hashlib.sha256(f'{MODEL_ID}@{MODEL_REVISION}:{RESIZE_MODE}'.encode()).hexdigest()[:16]


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


class ImageEncoder:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.torch = torch
        self.lock = Lock()
        self.model_id = MODEL_ID
        self.device = os.getenv('CV_DEVICE', 'cpu')
        torch.set_num_threads(int(os.getenv('CV_THREADS', '4')))
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=False)
        self.model = SiglipVisionModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(self.device).eval()
        self.size = int(self.model.config.image_size)

    def _prepare(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB')
        if RESIZE_MODE == 'pad':
            return ImageOps.pad(image, (self.size, self.size), Image.Resampling.BICUBIC, color=PAD_COLOR)
        return image.resize((self.size, self.size), Image.Resampling.BILINEAR)

    def encode(self, images):
        # Match the pinned processor's 224px bilinear resize before NumPy conversion.
        # This avoids costly full-resolution arrays for phone photos on CPU.
        prepared = [self._prepare(image) for image in images]
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=prepared, return_tensors='pt').to(self.device)
            features = self.model(**inputs).pooler_output
            features = self.torch.nn.functional.normalize(features, p=2, dim=-1)
            return features.cpu().tolist()


class VisualSearch:
    def __init__(self):
        self.encoder = ImageEncoder()

    def search(self, image, limit=5):
        from .database import connect
        vector = self.encoder.encode([image])[0]
        with connect() as db:
            rows = db.execute('''
                SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
                FROM wine_embeddings WHERE model = %s
                ORDER BY embedding <=> %s::vector LIMIT %s
            ''', (str(vector), MODEL_ID, str(vector), max(limit * 8, 16))).fetchall()
            ranked = best_by_slug(rows, limit)
            slugs = [item['slug'] for item in ranked]
            detailed = []
            if slugs:
                detailed = db.execute('''
                    SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
                    FROM wine_embeddings WHERE model = %s AND slug = ANY(%s)
                ''', (str(vector), MODEL_ID, slugs)).fetchall()
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
