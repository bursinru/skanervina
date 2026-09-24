"""Offline accuracy benchmark: the real Recognizer on an in-memory gallery.

No PostgreSQL is needed. The gallery is built exactly like app.import_catalog
(one full view + one detector label crop per photo), embeddings are cached on
disk per encoder configuration, and queries run through Recognizer.recognize,
so ranking, OCR blending and thresholds are the production code paths.

Query sets (all have ground-truth slugs):
  real       every photo in Датасет/extra-labels; that photo is removed from the
             gallery while it is queried (leave-one-out).
  shelf      seeded synthetic shelf shots: the target packshot between two
             other catalog bottles, perspective, light, glare, blur, JPEG.
  closeup    seeded synthetic close-ups of the catalog label crop with
             cylinder warp, rotation, glare and camera noise.

Synthetic queries are rendered from the same packshots the gallery indexes, so
they are optimistic in absolute terms. Use them to compare variants against
each other, and the real set for an honest (small) absolute check.

    PYTHONPATH=backend python backend/scripts/eval_offline.py \
      --catalog Датасет/strapi_output0709_enriched.csv \
      --images Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads \
      --sets real,shelf,closeup --per-set 300 --out reports/eval/baseline.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.image_io import register_decoders  # noqa: E402
from app.import_catalog import crop_view_digest, extra_files, label_crop_if_useful, load_catalog  # noqa: E402
from app.label_detection import label_rgb  # noqa: E402
from app.ranking import best_slug, same_label_family  # noqa: E402
from app.settings import settings  # noqa: E402
from app.vision import best_by_slug, split_full_crop  # noqa: E402

SUFFIXES = {".webp", ".jpg", ".jpeg", ".png", ".jfif", ".heic", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return label_rgb(opened)


class MemoryVisualSearch:
    """Drop-in for app.vision.VisualSearch backed by a NumPy matrix (exact search)."""

    def __init__(self, encoder, slugs, hashes, matrix, sources):
        self.encoder = encoder
        self.slugs = np.asarray(slugs)
        self.hashes = list(hashes)
        self.matrix = matrix.astype(np.float32)
        self.sources = list(sources)
        self.excluded: set = set()

    def _scores(self, image):
        vector = np.asarray(self.encoder.encode([image])[0], dtype=np.float32)
        scores = self.matrix @ vector
        if self.excluded:
            for index in self.excluded:
                scores[index] = -2.0
        return scores

    def _rows(self, scores, indices):
        return [
            {"slug": str(self.slugs[i]), "image_hash": self.hashes[i], "score": float(scores[i])}
            for i in indices
            if scores[i] > -1.5
        ]

    def search(self, image, limit=5):
        scores = self._scores(image)
        top = np.argsort(-scores)[: max(limit * 8, 16)]
        ranked = best_by_slug(self._rows(scores, top), limit)
        wanted = {item["slug"] for item in ranked}
        grouped = defaultdict(list)
        for i in np.flatnonzero(np.isin(self.slugs, list(wanted))):
            grouped[str(self.slugs[i])].extend(self._rows(scores, [i]))
        result = []
        for item in ranked:
            full_score, crop_score, best_view = split_full_crop(grouped.get(item["slug"], []))
            result.append({
                **item,
                "full_score": None if full_score is None else round(full_score, 4),
                "crop_score": None if crop_score is None else round(crop_score, 4),
                "best_view": best_view,
            })
        return result

    def score_slug(self, image, slug):
        scores = self._scores(image)
        rows = self._rows(scores, np.flatnonzero(self.slugs == slug))
        if not rows:
            return None
        full_score, crop_score, best_view = split_full_crop(rows)
        best = max(rows, key=lambda row: row["score"])
        return {**best, "full_score": full_score, "crop_score": crop_score, "best_view": best_view}


def gallery_files(catalog, images: Path | None, extra: Path | None):
    files = []
    index = {}
    if images and images.is_dir():
        for path in images.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, path)
    for wine in catalog:
        names = []
        if wine.image_name:
            names.append(Path(wine.image_name).name)
        for name in names:
            path = index.get(name)
            if path is not None:
                files.append((wine.slug, path, "catalog"))
                break
        if extra:
            for path in extra_files(extra, wine.slug):
                files.append((wine.slug, path, "extra"))
    return files


def build_gallery(encoder, files, cache_dir: Path, cache_key: str, batch=16):
    """Embeddings for full + label-crop views, cached per encoder configuration."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"gallery-{cache_key}.npz"
    cached = {}
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        cached = {key: data["vectors"][i] for i, key in enumerate(data["keys"])}
    slugs, hashes, vectors, sources = [], [], [], []
    pending = []
    for slug, path, kind in files:
        digest = sha256_file(path)
        for view, view_hash in (("full", digest), ("crop", crop_view_digest(digest))):
            key = f"{slug}|{view_hash}"
            if key in cached:
                slugs.append(slug); hashes.append(view_hash); vectors.append(cached[key]); sources.append(str(path))
            else:
                pending.append((slug, path, view, view_hash, key))
    started = time.time()
    done_keys = []
    new_vectors = []
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        images, meta = [], []
        for slug, path, view, view_hash, key in chunk:
            try:
                rgb = open_rgb(path)
            except Exception:
                continue
            if view == "crop":
                rgb = label_crop_if_useful(rgb)
                if rgb is None:
                    continue
            images.append(rgb)
            meta.append((slug, path, view_hash, key))
        if not images:
            continue
        for (slug, path, view_hash, key), vector in zip(meta, encoder.encode(images)):
            vector = np.asarray(vector, dtype=np.float32)
            slugs.append(slug); hashes.append(view_hash); vectors.append(vector); sources.append(str(path))
            done_keys.append(key); new_vectors.append(vector)
        if (start // batch) % 20 == 0:
            print(json.dumps({"gallery_encoded": start + len(chunk), "pending": len(pending),
                              "seconds": round(time.time() - started, 1)}), flush=True)
    if new_vectors:
        keys = list(cached) + done_keys
        all_vectors = list(cached.values()) + new_vectors
        np.savez(cache_path, keys=np.asarray(keys), vectors=np.stack(all_vectors))
    return slugs, hashes, np.stack(vectors), sources


# ---------------------------------------------------------------- synthetic queries

def _foreground(image: Image.Image) -> Image.Image:
    """RGBA packshot with the near-white studio background made transparent."""

    import cv2

    rgb = np.asarray(image.convert("RGB"))
    light = (rgb.min(axis=2) >= 232).astype(np.uint8)
    mask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), np.uint8)
    flood = light.copy()
    for x, y in ((0, 0), (rgb.shape[1] - 1, 0), (0, rgb.shape[0] - 1), (rgb.shape[1] - 1, rgb.shape[0] - 1)):
        if flood[y, x]:
            cv2.floodFill(flood, mask, (x, y), 2)
    alpha = np.where(flood == 2, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba = np.dstack([rgb, alpha])
    out = Image.fromarray(rgba, "RGBA")
    box = out.getbbox()
    return out.crop(box) if box else out


def _camera(image: Image.Image, rng: random.Random, seed: int) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.62, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.2))
    pixels = np.asarray(image).astype(np.float32)
    pixels *= np.array([rng.uniform(0.9, 1.1), 1, rng.uniform(0.88, 1.08)], dtype=np.float32)
    pixels += np.random.default_rng(seed).normal(0, rng.uniform(1, 6), pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if rng.random() < 0.45:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        x = rng.randint(int(width * 0.2), int(width * 0.8))
        band = max(6, int(width * rng.uniform(0.02, 0.06)))
        draw.ellipse((x - band, -height // 4, x + band, height * 5 // 4), fill=(255, 255, 255, rng.randint(40, 110)))
        image = Image.alpha_composite(image.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(band))).convert("RGB")
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    output = BytesIO()
    image.save(output, format="JPEG", quality=rng.randint(45, 88))
    return Image.open(BytesIO(output.getvalue())).convert("RGB")


def _perspective(image: Image.Image, rng: random.Random, fill) -> Image.Image:
    width, height = image.size
    coeffs = (1, rng.uniform(-0.08, 0.08), rng.uniform(-0.03, 0.03) * width,
              rng.uniform(-0.06, 0.06), 1, rng.uniform(-0.03, 0.03) * height,
              rng.uniform(-0.12, 0.12) / width, rng.uniform(-0.1, 0.1) / height)
    image = image.transform(image.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=fill)
    return image.rotate(rng.uniform(-8, 8), Image.Resampling.BICUBIC, fillcolor=fill)


def _cylinder(image: Image.Image, rng: random.Random) -> Image.Image:
    """Horizontal compression toward the edges, like a label wrapped on glass."""

    arr = np.asarray(image)
    height, width = arr.shape[:2]
    strength = rng.uniform(0.35, 0.8)
    xs = np.linspace(-1, 1, width)
    source = np.sin(xs * strength * np.pi / 2) / np.sin(strength * np.pi / 2)
    columns = np.clip(((source + 1) / 2 * (width - 1)).round().astype(int), 0, width - 1)
    shade = (0.78 + 0.22 * np.cos(xs * np.pi / 2 * strength))[None, :, None]
    warped = arr[:, columns].astype(np.float32) * shade
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))


BACKGROUNDS = [(58, 44, 36), (205, 196, 180), (120, 98, 76), (30, 30, 34), (170, 160, 150), (226, 222, 214)]


def render_shelf(target: Image.Image, neighbours, rng: random.Random, seed: int) -> Image.Image:
    height = 1400
    width = rng.randint(1000, 1200)
    background = BACKGROUNDS[rng.randrange(len(BACKGROUNDS))]
    canvas = Image.new("RGB", (width, height), background)
    noise = np.random.default_rng(seed + 1).normal(0, 10, (height // 20, width // 20, 3))
    texture = Image.fromarray(np.clip(np.asarray(background) + noise, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC)
    canvas.paste(texture)
    shelf_y = int(height * rng.uniform(0.86, 0.93))
    ImageDraw.Draw(canvas).rectangle((0, shelf_y, width, height), fill=tuple(max(0, c - 25) for c in background))
    bottle_height = int(height * rng.uniform(0.62, 0.8))
    placed = []
    centre = width // 2 + rng.randint(-60, 60)
    gap = rng.uniform(0.52, 0.72)
    target_fg = _foreground(target)
    scale = bottle_height / target_fg.height
    target_fg = target_fg.resize((max(1, int(target_fg.width * scale)), bottle_height), Image.Resampling.LANCZOS)
    step = int(target_fg.width * (1 + gap))
    for side, other in zip((-1, 1), neighbours):
        fg = _foreground(other)
        other_height = int(bottle_height * rng.uniform(0.9, 1.08))
        fg = fg.resize((max(1, int(fg.width * other_height / fg.height)), other_height), Image.Resampling.LANCZOS)
        x = centre + side * step - fg.width // 2
        placed.append((fg, x, shelf_y - fg.height))
    placed.append((target_fg, centre - target_fg.width // 2, shelf_y - target_fg.height))
    for fg, x, y in placed:
        canvas.paste(fg, (x, y), fg)
    canvas = _perspective(canvas, rng, background)
    return _camera(canvas, rng, seed)


def render_closeup(label: Image.Image, rng: random.Random, seed: int) -> Image.Image:
    fill = BACKGROUNDS[rng.randrange(len(BACKGROUNDS))]
    label = label.convert("RGB")
    label = _cylinder(label, rng)
    margin_x = int(label.width * rng.uniform(0.05, 0.35))
    margin_y = int(label.height * rng.uniform(0.05, 0.35))
    canvas = Image.new("RGB", (label.width + 2 * margin_x, label.height + 2 * margin_y), fill)
    canvas.paste(label, (margin_x, margin_y))
    scale = rng.uniform(700, 1300) / max(canvas.size)
    canvas = canvas.resize((max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))), Image.Resampling.LANCZOS)
    canvas = _perspective(canvas, rng, fill)
    return _camera(canvas, rng, seed)


def synthetic_queries(kind, files, count, seed):
    """Deterministic list of (slug, image, gallery_path) for one synthetic set."""

    catalog_files = [(slug, path) for slug, path, source in files if source == "catalog"]
    rng = random.Random(f"{seed}:{kind}")
    chosen = rng.sample(catalog_files, min(count, len(catalog_files)))
    for index, (slug, path) in enumerate(chosen):
        local = random.Random(f"{seed}:{kind}:{index}")
        try:
            image = open_rgb(path)
            if kind == "shelf":
                others = local.sample(catalog_files, 2)
                neighbours = [open_rgb(other) for _, other in others]
                query = render_shelf(image, neighbours, local, seed + index)
            else:
                label = label_crop_if_useful(image) or image
                query = render_closeup(label, local, seed + index)
        except Exception as error:  # a broken catalog photo should not stop the run
            print(json.dumps({"skip": str(path), "error": str(error)}), flush=True)
            continue
        yield f"{kind}/{index:04d}", slug, query, None


def real_queries(files):
    for slug, path, source in files:
        if source == "extra":
            yield f"real/{slug}/{path.name}", slug, open_rgb(path), path


# ---------------------------------------------------------------- evaluation

def image_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


def evaluate(recognizer, visual, catalog, queries, bottle_detection=True):
    rows = []
    source_index = defaultdict(list)
    for i, source in enumerate(visual.sources):
        source_index[source].append(i)
    for query_id, truth, image, holdout in queries:
        visual.excluded = set(source_index.get(str(holdout), [])) if holdout else set()
        started = time.time()
        result = recognizer.recognize(image_bytes(image), include_candidates=True)
        seconds = time.time() - started
        predicted = best_slug(result)
        top5 = [item["slug"] for item in (result.get("ranking") or {}).get("top5") or []]
        truth_wine, predicted_wine = catalog.get(truth), catalog.get(predicted) if predicted else None
        rows.append({
            "id": query_id,
            "truth": truth,
            "predicted": predicted,
            "status": result.get("status"),
            "top1": predicted == truth,
            "top5": truth in top5[:5],
            "rank": (top5.index(truth) + 1) if truth in top5 else None,
            "family_confusion": bool(predicted and predicted != truth and same_label_family(truth_wine, predicted_wine)),
            "score": result.get("confidence"),
            "query_view": (result.get("recognition") or {}).get("query_view"),
            "seconds": round(seconds, 3),
        })
    visual.excluded = set()
    return rows


def summarize(rows):
    by_set = defaultdict(list)
    for row in rows:
        by_set[row["id"].split("/")[0]].append(row)
    summary = {}
    for name, items in sorted(by_set.items()):
        n = len(items)
        matched = [row for row in items if row["status"] == "matched"]
        summary[name] = {
            "n": n,
            "top1": round(sum(row["top1"] for row in items) / n, 4),
            "top5": round(sum(row["top5"] for row in items) / n, 4),
            "mrr5": round(sum(1 / row["rank"] for row in items if row["rank"]) / n, 4),
            "matched_share": round(len(matched) / n, 4),
            "matched_precision": round(sum(row["top1"] for row in matched) / len(matched), 4) if matched else None,
            "family_confusions": sum(row["family_confusion"] for row in items),
            "median_seconds": round(float(np.median([row["seconds"] for row in items])), 3),
            "status": dict(Counter(row["status"] for row in items)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=settings.catalog_csv)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("backend/data/eval-cache"))
    parser.add_argument("--sets", default="real,shelf,closeup")
    parser.add_argument("--per-set", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--no-bottle-detector", action="store_true")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    register_decoders()
    catalog = load_catalog(args.catalog)
    extra = args.extra or (args.catalog.parent / "extra-labels")
    files = gallery_files(catalog, args.images, extra if extra.is_dir() else None)
    print(json.dumps({"wines": catalog.size, "gallery_files": len(files)}), flush=True)

    from app.recognition import Recognizer
    from app.vision import ImageEncoder, encoder_cache_key

    encoder = ImageEncoder()
    slugs, hashes, matrix, sources = build_gallery(encoder, files, args.cache, encoder_cache_key())
    visual = MemoryVisualSearch(encoder, slugs, hashes, matrix, sources)
    # Build without CV so the Recognizer does not load a second encoder or touch the DB.
    os.environ["CV_ENABLED"] = "false"
    recognizer = Recognizer(catalog, settings)
    os.environ["CV_ENABLED"] = "true"
    recognizer.visual = visual
    recognizer.visual_status = "ready"
    if args.no_bottle_detector:
        recognizer._bottle_detector_error = RuntimeError("disabled for benchmark")

    wanted = [name.strip() for name in args.sets.split(",") if name.strip()]
    rows = []
    for name in wanted:
        queries = real_queries(files) if name == "real" else synthetic_queries(name, files, args.per_set, args.seed)
        started = time.time()
        part = evaluate(recognizer, visual, catalog, queries)
        rows.extend(part)
        print(json.dumps({"set": name, "n": len(part), "seconds": round(time.time() - started, 1),
                          **summarize(part).get(name, {})}, ensure_ascii=False), flush=True)
    report = {
        "label": args.label,
        "encoder": encoder_cache_key(),
        "env": {key: value for key, value in sorted(os.environ.items()) if key.startswith(("CV_", "OCR_", "RERANK_"))},
        "gallery_vectors": int(matrix.shape[0]),
        "summary": summarize(rows),
        "rows": rows,
    }
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
