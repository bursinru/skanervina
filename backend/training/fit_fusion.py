"""Fit the visual + label-text fusion (conditional logit over candidate wines).

Input: features.jsonl (PP-OCR lines per photo), visual_all.jsonl (fused SigLIP
score of every catalog wine per photo), catalog_ppocr.jsonl (PP-OCR lines of
every catalog photo), labels.json ({path, expected, set}) and catalog.json,
dumped inside the recognition container. Output: backend/app/fusion.json and
backend/app/catalog_photo_text.json.

    python backend/training/fit_fusion.py --data <dir> [--write]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.fusion import FEATURES, PHOTO_TEXT_PATH, WEIGHTS_PATH, candidates, features  # noqa: E402
from app.label_text import TextIndex, catalog_documents, ocr_words, reference_documents  # noqa: E402

SOURCES = ("ocr_label", "ocr_bottle")


def load(data: Path):
    catalog = [SimpleNamespace(**row) for row in json.loads((data / "catalog.json").read_text())]
    index = TextIndex(catalog_documents(catalog), catalog)
    photo_index = None
    photo_path = data / "catalog_ppocr.jsonl"
    if photo_path.exists():
        entries = [json.loads(line) for line in photo_path.read_text().splitlines() if line.strip()]
        photo_index = TextIndex(reference_documents(entries))
    labels = {row["path"]: row for row in json.loads((data / "labels.json").read_text())}
    visual = {}
    for line in (data / "visual_all.jsonl").read_text().splitlines():
        row = json.loads(line)
        visual[row["path"]] = row["scores"]
    queries = []
    for line in (data / "features.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["path"] not in visual:
            continue
        words = []
        for source in SOURCES:
            payload = row.get(source)
            if payload:
                words += ocr_words(payload["lines"], payload["size"])
        queries.append({
            "path": row["path"],
            "set": labels[row["path"]]["set"],
            "expected": row["expected"],
            "baseline": row.get("base_top1"),
            "visual": visual[row["path"]],
            "text": index.score(words),
            "photo": photo_index.score(words) if photo_index else None,
        })
    return queries


def matrices(queries, visual_k, text_n):
    out = []
    for query in queries:
        slugs = candidates(query["visual"], (query["text"], query["photo"]), visual_k, text_n)
        rows = features(slugs, query["visual"], query["text"], query["photo"])
        target = slugs.index(query["expected"]) if query["expected"] in slugs else None
        out.append((slugs, np.array([rows[slug] for slug in slugs], dtype=float), target))
    return out


def loss(data, w, l2):
    total = 0.0
    usable = [(x, t) for _, x, t in data if t is not None and len(x) > 1]
    for x, t in usable:
        z = x @ w
        total += np.log(np.exp(z - z.max()).sum()) + z.max() - z[t]
    return total / len(usable)


def fit(data, l2=0.05, steps=3000, lr=0.05):
    usable = [(x, t) for _, x, t in data if t is not None and len(x) > 1]
    stacked = np.vstack([x for x, _ in usable])
    scale = stacked.std(axis=0)
    scale[scale == 0] = 1.0
    w = np.zeros(stacked.shape[1])
    w[0] = 1.0  # start from the visual order
    m, v = np.zeros_like(w), np.zeros_like(w)
    for step in range(1, steps + 1):
        grad = l2 * w
        for x, t in usable:
            z = (x / scale) @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            grad += ((x / scale).T @ p - (x[t] / scale)) / len(usable)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        w -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    return w / scale


def top1(data, w):
    return sum(t is not None and int(np.argmax(x @ w)) == t for _, x, t in data)


def predictions(data, w):
    """(probability of the top-1, top-1 is right) per photo."""
    out = []
    for _, x, t in data:
        z = x @ w
        p = np.exp(z - z.max())
        p /= p.sum()
        best = int(np.argmax(z))
        out.append((float(p[best]), t is not None and best == t))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--visual-k", type=int, default=20)
    parser.add_argument("--text-n", type=int, default=10)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    queries = load(args.data)
    data = matrices(queries, args.visual_k, args.text_n)
    n = len(queries)
    visual_only = sum(max(q["visual"], key=q["visual"].get) == q["expected"] for q in queries)
    baseline = sum(q["baseline"] == q["expected"] for q in queries)
    recall = sum(t is not None for _, _, t in data)
    print(f"photos {n}  current pipeline {baseline}  visual-only {visual_only}  candidate recall {recall}")

    sets = sorted({q["set"] for q in queries})
    held = 0
    for name in sets:
        train = [d for d, q in zip(data, queries) if q["set"] != name]
        test = [d for d, q in zip(data, queries) if q["set"] == name]
        w = fit(train, args.l2)
        got = top1(test, w)
        held += got
        print(f"  leave-set-out {name}: {got}/{len(test)}")
    print(f"leave-set-out total {held}/{n} = {held / n:.1%}")

    rng = random.Random(7)
    totals = []
    held_out = []
    for repeat in range(3):
        order = list(range(n))
        rng.shuffle(order)
        got = 0
        for fold in range(5):
            test_ids = set(order[fold::5])
            w = fit([d for i, d in enumerate(data) if i not in test_ids], args.l2)
            test = [d for i, d in enumerate(data) if i in test_ids]
            got += top1(test, w)
            held_out += predictions(test, w)
        totals.append(got)
    print(f"5-fold CV x3: {totals} mean {np.mean(totals) / n:.1%}")
    match_probability = 0.8
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [right for p, right in held_out if p >= threshold]
        precision = sum(kept) / len(kept) if kept else 0.0
        print(f"  p >= {threshold}: {len(kept) / len(held_out):.0%} of photos, top-1 right {precision:.1%}")
        if precision >= 0.95 and threshold < match_probability:
            match_probability = threshold

    w = fit(data, args.l2)
    print(f"in-sample {top1(data, w)}/{n}  loss {loss(data, w, args.l2):.4f}  (6000 steps: {loss(data, fit(data, args.l2, steps=6000), args.l2):.4f})")
    for name, value in zip(FEATURES, w):
        print(f"  {name:12s} {value:+.3f}")
    if args.write:
        WEIGHTS_PATH.write_text(json.dumps({
            "features": list(FEATURES),
            "weights": [round(float(value), 5) for value in w],
            "visual_k": args.visual_k,
            "text_n": args.text_n,
            "l2": args.l2,
            "match_probability": match_probability,
            "trained_on": n,
        }, indent=2) + "\n")
        print("wrote", WEIGHTS_PATH)
        photo_path = args.data / "catalog_ppocr.jsonl"
        if photo_path.exists():
            entries = [json.loads(line) for line in photo_path.read_text().splitlines() if line.strip()]
            photo = {
                slug: [[text, round(weight, 3)] for text, weight in words if weight >= 0.05]
                for slug, words in sorted(reference_documents(entries).items())
            }
            PHOTO_TEXT_PATH.write_text(json.dumps(photo, ensure_ascii=False, separators=(",", ":")))
            print("wrote", PHOTO_TEXT_PATH, len(photo), "wines")


if __name__ == "__main__":
    main()
