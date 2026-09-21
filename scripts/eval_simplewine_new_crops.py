"""Local 53-photo eval against a frozen gallery. Does not start Postgres."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HOME", str(Path("backend/data/models").resolve()))
os.environ.setdefault("CV_DEVICE", "cpu")
os.environ.setdefault("CV_THREADS", "4")

from app.import_catalog import load_catalog
from app.label_detection import crop_label, detect_label
from app.label_ocr import load_references, pipeline_version, read_label
from app.label_signals import blend_candidates, crop_color_features
from app.ranking import distinct_margin, is_visual_match, same_label_family
from app.settings import settings
from app.vision import ImageEncoder


def load_gallery(path: Path):
    meta = json.loads((path / "gallery.json").read_text())
    gallery = np.load(path / "gallery.npy")
    slugs = list(dict.fromkeys(row["slug"] for row in meta["rows"]))
    ids = {slug: index for index, slug in enumerate(slugs)}
    row_ids = np.array([ids[row["slug"]] for row in meta["rows"]])
    kinds = np.array([row["kind"] for row in meta["rows"]])
    return slugs, row_ids, kinds, gallery


def view_scores(gallery, vector, slugs, row_ids, kinds):
    similarity = gallery @ np.asarray(vector, dtype=np.float32)
    views = {}
    for kind in ("full", "crop"):
        scores = np.full(len(slugs), -1.0, dtype=np.float32)
        mask = kinds == kind
        if mask.any():
            np.maximum.at(scores, row_ids[mask], similarity[mask])
        views[kind] = scores
    return views


def decide(ranked, catalog):
    best = ranked[0]
    wine = catalog.get(best["slug"])
    runner = catalog.get(ranked[1]["slug"]) if len(ranked) > 1 else None
    family_tie = same_label_family(wine, runner)
    margin = distinct_margin(ranked, catalog)
    matched = bool(wine) and is_visual_match(
        best["score"],
        margin,
        threshold=0.75,
        min_margin=0.04,
        clear_match=0.85,
        family_tie=family_tie,
        corroborated=bool(best.get("ocr_delta")),
    )
    status = "matched" if matched else ("uncertain" if best["score"] >= 0.65 else "unknown")
    return status, margin, family_tie


def main():
    root = Path("Датасет/yandex-real-eval")
    cases = list(csv.DictReader((root.parent / "yandex-real-eval.csv").open(encoding="utf-8")))
    catalog = load_catalog(settings.catalog_csv)
    references = load_references("backend/data/catalog-ocr.json", settings.ocr_languages)
    query_cache = Path("backend/data/label-revision-query-cache")
    query_cache.mkdir(exist_ok=True)
    galleries = {
        "old_crops": Path("backend/data/ablation-2026-09-20"),
        "new_crops": Path("backend/data/label-revision-gallery"),
    }
    loaded = {name: load_gallery(path) for name, path in galleries.items()}
    encoder = None
    version = pipeline_version()
    rows_out = []
    for index, case in enumerate(cases, 1):
        path = root / case["file"]
        image = Image.open(path)
        detection = detect_label(image)
        crop = crop_label(image, detection)
        key = hashlib.sha256(path.read_bytes() + version.encode() + b"-crop").hexdigest()
        full_key = hashlib.sha256(path.read_bytes() + version.encode() + b"-full").hexdigest()
        vector_path = query_cache / (key + ".npy")
        full_path = query_cache / (full_key + ".npy")
        if encoder is None and (not vector_path.exists() or not full_path.exists()):
            encoder = ImageEncoder()
        if vector_path.exists():
            vector = np.load(vector_path)
        else:
            vector = encoder.encode([crop])[0]
            np.save(vector_path, vector)
        if full_path.exists():
            full_vector = np.load(full_path)
        else:
            full_vector = encoder.encode([image])[0]
            np.save(full_path, full_vector)
        text = read_label(crop)
        color_features = crop_color_features(crop)
        record = {
            "file": case["file"],
            "title": case["title"],
            "expected": case["expected_slug"],
            "method": detection.method,
            "ocr_text": text,
            "print_tone": color_features.get("print_tone"),
        }
        for name, (slugs, row_ids, kinds, gallery) in loaded.items():
            crop_views = view_scores(gallery, vector, slugs, row_ids, kinds)
            full_views = view_scores(gallery, full_vector, slugs, row_ids, kinds)
            merged = {
                kind: np.maximum(crop_views[kind], full_views[kind])
                for kind in ("full", "crop")
            }
            crop_only = [
                {"slug": slugs[i], "score": float(crop_views["crop"][i]), "full_score": float(crop_views["full"][i]), "crop_score": float(crop_views["crop"][i])}
                for i in np.argsort(-crop_views["crop"], kind="stable")[:8]
            ]
            both = [
                {
                    "slug": slugs[i],
                    "score": float(max(merged["full"][i], merged["crop"][i])),
                    "full_score": float(merged["full"][i]),
                    "crop_score": float(merged["crop"][i]),
                }
                for i in np.argsort(-np.maximum(merged["full"], merged["crop"]), kind="stable")[:8]
            ]
            ranked = blend_candidates(both, catalog, color_features, text, True, references)
            ranked_crop = blend_candidates(crop_only, catalog, color_features, text, True, references)
            status, margin, family_tie = decide(ranked, catalog)
            top = ranked[0]
            record[name] = {
                "got": top["slug"],
                "hit": top["slug"] == case["expected_slug"],
                "status": status,
                "score": round(top["score"], 4),
                "siglip": top.get("siglip"),
                "ocr_delta": top.get("ocr_delta"),
                "color_delta": top.get("color_delta"),
                "margin": margin,
                "family_tie": family_tie,
                "crop_only_top1": crop_only[0]["slug"],
                "crop_only_hit": crop_only[0]["slug"] == case["expected_slug"],
                "crop_blend_top1": ranked_crop[0]["slug"],
                "crop_blend_hit": ranked_crop[0]["slug"] == case["expected_slug"],
                "both_top1": both[0]["slug"],
                "both_hit": both[0]["slug"] == case["expected_slug"],
                "top5": [item["slug"] for item in ranked[:5]],
                "top5_hit": any(item["slug"] == case["expected_slug"] for item in ranked[:5]),
            }
        rows_out.append(record)
        new = record["new_crops"]
        print(
            f"{index:02d}/53 {'OK' if new['hit'] else '  '} {new['status']:10} {new['score']:.4f}  {new['got']}  <- {case['expected_slug']}",
            flush=True,
        )

    def summarize(key, field="hit"):
        hits = sum(row[key][field] for row in rows_out)
        statuses = Counter(row[key]["status"] for row in rows_out)
        scores = [row[key]["score"] for row in rows_out]
        return {
            "top1": hits,
            "top1_pct": round(100 * hits / len(rows_out), 1),
            "top5": sum(row[key]["top5_hit"] for row in rows_out),
            "top5_pct": round(100 * sum(row[key]["top5_hit"] for row in rows_out) / len(rows_out), 1),
            "crop_only_top1": sum(row[key]["crop_only_hit"] for row in rows_out),
            "crop_blend_top1": sum(row[key].get("crop_blend_hit") for row in rows_out),
            "both_top1": sum(row[key]["both_hit"] for row in rows_out),
            "status": dict(statuses),
            "mean_score": round(sum(scores) / len(scores), 4),
            "ocr_bonus": sum(1 for row in rows_out if (row[key].get("ocr_delta") or 0) > 0),
        }

    summary = {name: summarize(name) for name in galleries}
    previous = json.loads(Path("reports/recognition/label-revision-2026-09-21/simplewine-53-local-new-crops.json").read_text())
    live = json.loads(Path("reports/recognition/label-revision-2026-09-21/simplewine-53-live.json").read_text())
    dest = Path("reports/recognition/label-revision-2026-09-21/simplewine-53-after-worst10.json")
    payload = {
        "summary": summary,
        "pipeline": version,
        "previous_local": previous.get("summary"),
        "previous_live": live.get("summary"),
        "rows": rows_out,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"pipeline": version, "references": len(references), **summary}, ensure_ascii=False, indent=2), flush=True)
    print("\nCOMPARE top-1 / 53", flush=True)
    print(f"  live (server, old jpg)           {live['summary']['top1']:2d}/53  {live['summary']['top1_pct']}%", flush=True)
    print(f"  local prev, old gallery          {previous['summary']['old_crops']['top1']:2d}/53  {previous['summary']['old_crops']['top1_pct']}%", flush=True)
    print(f"  local prev, v2 gallery           {previous['summary']['new_crops']['top1']:2d}/53  {previous['summary']['new_crops']['top1_pct']}%", flush=True)
    print(f"  now old gallery  merge+color     {summary['old_crops']['top1']:2d}/53  {summary['old_crops']['top1_pct']}%  crop-only {summary['old_crops']['crop_only_top1']}  crop+blend {summary['old_crops']['crop_blend_top1']}", flush=True)
    print(f"  now patched v2   merge+color     {summary['new_crops']['top1']:2d}/53  {summary['new_crops']['top1_pct']}%  crop-only {summary['new_crops']['crop_only_top1']}  crop+blend {summary['new_crops']['crop_blend_top1']}", flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
