"""SigLIP 2 image embeddings and PostgreSQL cosine retrieval."""
import os
from threading import Lock
from PIL import Image, ImageOps

MODEL_ID = 'google/siglip2-base-patch16-224'
MODEL_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'


def best_by_slug(rows, limit=5):
    best = {}
    for row in rows:
        slug = row['slug']
        if slug not in best or row['score'] > best[slug]['score']:
            best[slug] = row
    return sorted(best.values(), key=lambda item: item['score'], reverse=True)[:limit]


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

    def encode(self, images):
        # Match the pinned processor's 224px bilinear resize before NumPy conversion.
        # This avoids costly full-resolution arrays for phone photos on CPU.
        prepared = [ImageOps.exif_transpose(image).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR) for image in images]
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
                SELECT slug, 1 - (embedding <=> %s::vector) AS score
                FROM wine_embeddings WHERE model = %s
                ORDER BY embedding <=> %s::vector LIMIT %s
            ''', (str(vector), MODEL_ID, str(vector), max(limit * 8, 16))).fetchall()
        return best_by_slug(rows, limit)
