"""Index the grape line of series that share one label. Does not rewrite other crops."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from .catalog import WineCatalog
from .database import connect
from .grape_lines import crop_quad, grape_model, grape_tokens, grape_view_digest, in_confusable_series, index_path, line_for_grapes
from .import_catalog import label_crop_if_useful
from .label_detection import label_rgb
from .pp_ocr import read_lines
from .settings import settings
from .vision import MODEL_ID, MODEL_REVISION, SECONDARY_MODEL_ID, SECONDARY_MODEL_REVISION, ImageEncoder, indexed_models


def studio_file(root: Path, wine) -> Path | None:
    filename = unquote(Path(urlparse(wine.direct_image_url or "").path).name) or wine.image_name
    if not filename:
        return None
    path = (root / filename).resolve()
    if path.is_file() and path.is_relative_to(root.resolve()):
        return path
    return None


def collect(catalog: WineCatalog, images: Path) -> list:
    pending = []
    skipped = []
    for wine in catalog:
        if not in_confusable_series(wine):
            continue
        path = studio_file(images, wine)
        if path is None:
            skipped.append({"slug": wine.slug, "reason": "no_file"})
            continue
        with Image.open(path) as image:
            rgb = label_rgb(image)
        view = label_crop_if_useful(rgb) or rgb
        words = grape_tokens(wine)
        line = line_for_grapes(read_lines(view), words)
        if line is None:
            skipped.append({"slug": wine.slug, "reason": "no_grape_line", "grapes": sorted(words)})
            continue
        crop = crop_quad(view, line.get("box"))
        if crop is None:
            skipped.append({"slug": wine.slug, "reason": "tiny_line"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pending.append((wine.slug, grape_view_digest(digest), crop, line.get("text")))
    return pending, skipped


def main():
    images = Path(os.environ["CATALOG_IMAGES"])
    catalog = WineCatalog.from_csv(Path(os.environ.get("CATALOG_CSV", settings.catalog_csv)), settings.image_base_url)
    print(json.dumps({"reading_lines": True}), flush=True)
    pending, skipped = collect(catalog, images)
    print(json.dumps({"grape_lines": len(pending), "skipped": len(skipped)}, ensure_ascii=False), flush=True)
    for item in skipped:
        print(json.dumps({"skip": item}, ensure_ascii=False), flush=True)
    revisions = {MODEL_ID: MODEL_REVISION, SECONDARY_MODEL_ID: SECONDARY_MODEL_REVISION}
    encoders = {}
    for model in indexed_models():
        print(json.dumps({"loading_encoder": grape_model(model)}), flush=True)
        encoders[model] = ImageEncoder(model, revisions[model])
    written = {}
    batch = 8
    for offset in range(0, len(pending), batch):
        chunk = pending[offset:offset + batch]
        crops = [item[2] for item in chunk]
        for model, encoder in encoders.items():
            vectors = encoder.encode(crops)
            with connect() as db:
                for (slug, digest, _crop, _text), vector in zip(chunk, vectors):
                    db.execute(
                        '''INSERT INTO wine_embeddings (slug, model, image_hash, embedding)
                           VALUES (%s, %s, %s, %s::vector)
                           ON CONFLICT (slug, model, image_hash) DO UPDATE
                           SET embedding = excluded.embedding, updated_at = now()''',
                        (slug, grape_model(model), digest, str(vector)),
                    )
                    written[slug] = digest
        print(json.dumps({"indexed": min(offset + batch, len(pending)), "total": len(pending)}), flush=True)
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, ensure_ascii=False, indent=2))
    print(json.dumps({"index": str(path), "wines": len(written)}), flush=True)


if __name__ == "__main__":
    main()
