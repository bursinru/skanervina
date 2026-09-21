"""Probe crop variants for the 10 worst SimpleWine queries. Local gallery only."""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("HF_HOME", str(Path("backend/data/models").resolve()))
os.environ.setdefault("CV_DEVICE", "cpu")
os.environ.setdefault("CV_THREADS", "4")

from app.label_detection import LabelDetection, crop_label, detect_label, label_rgb
from app.label_ocr import pipeline_version
from app.vision import ImageEncoder

CASES = [
    ("31-sira.jpg", "sira"),
    ("07-chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-premium-krasn.jpg", "chateau-de-talu-yuzhnaya-vertikal-kaberne-fran-premium-krasnoe-suhoe-143"),
    ("01-loco-cimbali-oranzh-muskat-oranzhevoe-suhoe-127.jpg", "loco-cimbali-oranzh-muskat-oranzhevoe-suhoe-127"),
    ("34-nikolaev-i-synovya-flamingo-malbek-rozovoe-suhoe-125.jpg", "nikolaev-i-synovya-flamingo-malbek-rozovoe-suhoe-125"),
    ("09-loco-cimbali-risling-beloe-suhoe-125.jpg", "loco-cimbali-risling-beloe-suhoe-125"),
    ("41-risling.jpg", "risling"),
    ("25-usadba-mezyb-mezyb-merlo-krasnoe-suhoe-145.jpg", "usadba-mezyb-mezyb-merlo-krasnoe-suhoe-145"),
    ("10-roze-1.jpg", "roze-1"),
    ("48-usadba-markoth-kyuve-blan-shardone-beloe-suhoe-12.jpg", "usadba-markoth-kyuve-blan-shardone-beloe-suhoe-12"),
    ("23-loco-cimbali-rkatsiteli-oranzhevoe-suhoe-13.jpg", "loco-cimbali-rkatsiteli-oranzhevoe-suhoe-13"),
]


def load_gallery(path: Path):
    meta = json.loads((path / "gallery.json").read_text())
    gallery = np.load(path / "gallery.npy")
    slugs = list(dict.fromkeys(row["slug"] for row in meta["rows"]))
    ids = {slug: i for i, slug in enumerate(slugs)}
    row_ids = np.array([ids[row["slug"]] for row in meta["rows"]])
    kinds = np.array([row["kind"] for row in meta["rows"]])
    return slugs, row_ids, kinds, gallery


def rank(gallery, vector, slugs, row_ids, kinds, expected, mode="crop"):
    sim = gallery @ np.asarray(vector, dtype=np.float32)
    if mode == "both":
        scores = np.full(len(slugs), -1.0, dtype=np.float32)
        for kind in ("full", "crop"):
            part = np.full(len(slugs), -1.0, dtype=np.float32)
            mask = kinds == kind
            np.maximum.at(part, row_ids[mask], sim[mask])
            scores = np.maximum(scores, part)
    else:
        scores = np.full(len(slugs), -1.0, dtype=np.float32)
        mask = kinds == mode
        np.maximum.at(scores, row_ids[mask], sim[mask])
    order = np.argsort(-scores, kind="stable")
    top = slugs[order[0]]
    exp_i = slugs.index(expected) if expected in slugs else None
    exp_rank = int(np.where(order == exp_i)[0][0] + 1) if exp_i is not None else None
    exp_score = float(scores[exp_i]) if exp_i is not None else None
    return top, float(scores[order[0]]), exp_rank, exp_score, [slugs[i] for i in order[:3]]


def variants(image):
    rgb = label_rgb(image)
    det = detect_label(rgb)
    auto = crop_label(rgb, det)
    bbox = rgb.crop((round(det.bbox[0]*rgb.width), round(det.bbox[1]*rgb.height), round(det.bbox[2]*rgb.width), round(det.bbox[3]*rgb.height)))
    no_contour = crop_label(rgb, replace(det, contour=None))
    no_quad = crop_label(rgb, replace(det, quad=None))
    pad = list(det.bbox)
    pad[0] = max(0, pad[0]-0.04); pad[1] = max(0, pad[1]-0.04)
    pad[2] = min(1, pad[2]+0.04); pad[3] = min(1, pad[3]+0.04)
    padded = rgb.crop((round(pad[0]*rgb.width), round(pad[1]*rgb.height), round(pad[2]*rgb.width), round(pad[3]*rgb.height)))
    lower = rgb.crop((0, int(rgb.height*0.45), rgb.width, rgb.height))
    mid = rgb.crop((int(rgb.width*0.15), int(rgb.height*0.35), int(rgb.width*0.85), int(rgb.height*0.95)))
    cat = detect_label(rgb, catalog=True)
    catalog_crop = crop_label(rgb, cat)
    return {
        "auto": auto,
        "bbox": bbox,
        "no_contour": no_contour,
        "no_quad": no_quad,
        "padded": padded,
        "full": rgb,
        "lower_half": lower,
        "center_lower": mid,
        "catalog_mode": catalog_crop,
    }, det, cat


def main():
    root = Path("Датасет/yandex-real-eval")
    preview = Path("reports/recognition/label-revision-2026-09-21/worst10")
    preview.mkdir(parents=True, exist_ok=True)
    galleries = {
        "new": load_gallery(Path("backend/data/label-revision-gallery")),
        "old": load_gallery(Path("backend/data/ablation-2026-09-20")),
    }
    cache = Path("backend/data/label-revision-query-cache")
    cache.mkdir(exist_ok=True)
    encoder = ImageEncoder()
    version = pipeline_version()
    report = []
    for file, expected in CASES:
        image = Image.open(root / file)
        vars_, det, cat = variants(image)
        print(f"\n=== {file} expected={expected} auto={det.method} bbox={tuple(round(x,3) for x in det.bbox)} catalog={cat.method}", flush=True)
        row = {"file": file, "expected": expected, "auto_method": det.method, "variants": {}}
        for name, im in vars_.items():
            thumb = im.copy(); thumb.thumbnail((360, 480))
            thumb.save(preview / f"{Path(file).stem}--{name}.jpg", quality=88)
            raw = hashlib.sha256(im.tobytes() + version.encode() + name.encode()).hexdigest()
            npy = cache / f"var-{raw}.npy"
            if npy.exists():
                vector = np.load(npy)
            else:
                vector = encoder.encode([im])[0]
                np.save(npy, vector)
            for gname, packed in galleries.items():
                slugs, row_ids, kinds, gallery = packed
                top, score, exp_rank, exp_score, top3 = rank(gallery, vector, slugs, row_ids, kinds, expected, mode="crop")
                both_top, both_score, both_rank, both_exp, _ = rank(gallery, vector, slugs, row_ids, kinds, expected, mode="both")
                hit = top == expected
                key = f"{gname}/{name}"
                row["variants"][key] = {
                    "hit": hit, "top": top, "score": round(score,4), "exp_rank": exp_rank,
                    "exp_score": None if exp_score is None else round(exp_score,4), "top3": top3,
                    "both_hit": both_top == expected, "both_top": both_top, "both_rank": both_rank,
                }
                mark = "OK" if hit else f"#{exp_rank}"
                if gname == "new":
                    print(f"  {name:13} crop={mark:4} both={'OK' if both_top==expected else '#'+str(both_rank):4} top={top[:42]}  s={score:.3f}", flush=True)
        report.append(row)
    dest = preview / "probe.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # which variant wins most on new gallery
    names = list(vars_.keys())
    print("\nWINS new gallery:")
    for name in names:
        hits = sum(1 for r in report if r["variants"][f"new/{name}"]["hit"])
        print(f"  {name:13} {hits}/10")
    print("wrote", dest)


if __name__ == "__main__":
    main()
