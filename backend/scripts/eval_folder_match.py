"""Batch visual match for a folder of images against the live catalog index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

from app.database import connect
from app.image_io import register_decoders
from app.import_catalog import load_catalog
from app.label_detection import crop_label, detect_label
from app.label_signals import blend_candidates, crop_color_features
from app.settings import settings
from app.vision import MODEL_ID, ImageEncoder, best_by_slug, split_full_crop

SUFFIXES = {".webp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jfif", ".heic"}


def decide(score: float, margin: float, threshold: float, min_margin: float, clear_match: float) -> str:
    if score >= threshold and (margin >= min_margin or score >= clear_match):
        return "matched"
    if score >= 0.65:
        return "uncertain"
    return "unknown"


def search_vector(db, vector, limit=5):
    from collections import defaultdict

    rows = db.execute(
        """
        SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
        FROM wine_embeddings WHERE model = %s
        ORDER BY embedding <=> %s::vector LIMIT %s
        """,
        (str(vector), MODEL_ID, str(vector), max(limit * 8, 16)),
    ).fetchall()
    ranked = best_by_slug(rows, limit)
    slugs = [item["slug"] for item in ranked]
    detailed = []
    if slugs:
        detailed = db.execute(
            """
            SELECT slug, image_hash, 1 - (embedding <=> %s::vector) AS score
            FROM wine_embeddings WHERE model = %s AND slug = ANY(%s)
            """,
            (str(vector), MODEL_ID, slugs),
        ).fetchall()
    grouped = defaultdict(list)
    for row in detailed:
        grouped[row["slug"]].append(row)
    scored = []
    for item in ranked:
        full_score, crop_score, best_view = split_full_crop(grouped.get(item["slug"], []))
        scored.append(
            {
                **item,
                "full_score": None if full_score is None else round(full_score, 4),
                "crop_score": None if crop_score is None else round(crop_score, 4),
                "best_view": best_view,
            }
        )
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    register_decoders()
    catalog = load_catalog(Path(os.getenv("CATALOG_CSV", str(settings.catalog_csv))))
    encoder = ImageEncoder()
    threshold = float(os.getenv("CV_MATCH_THRESHOLD", "0.75"))
    min_margin = float(os.getenv("CV_MATCH_MARGIN", "0.04"))
    clear_match = float(os.getenv("CV_CLEAR_MATCH", "0.85"))
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                continue
    files = sorted(
        path
        for path in args.folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES and path.name not in done
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"todo": len(files), "already": len(done)}), flush=True)
    with connect() as db, args.out.open("a", encoding="utf-8") as out:
        for offset in range(0, len(files), args.batch_size):
            batch_paths = files[offset : offset + args.batch_size]
            images = []
            kept = []
            errors = []
            for path in batch_paths:
                try:
                    with Image.open(path) as im:
                        rgb = ImageOps.exif_transpose(im).convert("RGB")
                    if rgb.width * rgb.height > 24_000_000:
                        errors.append({"file": path.name, "status": "error", "reason": "too_large"})
                        continue
                    rgb.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
                    label = crop_label(rgb, detect_label(rgb))
                    images.append(label)
                    kept.append(path)
                except Exception as exc:
                    errors.append({"file": path.name, "status": "error", "reason": type(exc).__name__})
            for item in errors:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            if not images:
                out.flush()
                continue
            vectors = encoder.encode(images)
            for path, image, vector in zip(kept, images, vectors):
                candidates = search_vector(db, vector)
                if not candidates:
                    row = {"file": path.name, "status": "unknown", "reason": "index_empty"}
                else:
                    color_features = crop_color_features(image)
                    ranked = blend_candidates(candidates, catalog, color_features, "", False)
                    best = ranked[0]
                    second = ranked[1]["score"] if len(ranked) > 1 else 0.0
                    margin = best["score"] - second
                    wine = catalog.get(best["slug"])
                    status = decide(best["score"], margin, threshold, min_margin, clear_match)
                    row = {
                        "file": path.name,
                        "status": status,
                        "score": round(best["score"], 4),
                        "margin": round(margin, 4),
                        "slug": best["slug"],
                        "name": wine.name if wine else best["slug"],
                        "winery": wine.winery if wine else "",
                        "full_score": best.get("full_score"),
                        "crop_score": best.get("crop_score"),
                        "best_view": best.get("best_view"),
                    }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps({"done": min(offset + args.batch_size, len(files)), "todo": len(files)}),
                flush=True,
            )


if __name__ == "__main__":
    main()
