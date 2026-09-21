"""Ablation on real phone photos: crop/full/merge × OCR on/off.

Hits a running recognition API (debug modes). No ground-truth slugs in filenames;
the report compares agreement, match rate and scores.

    python3 scripts/eval_real_photos.py \
      --folder 'Датасет/Реальные фото' \
      --url http://127.0.0.1:3000 \
      --out reports/recognition/real-100.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import median

SUFFIXES = {".webp", ".jpg", ".jpeg", ".png"}

CONFIGS = [
    ("merge_ocr", "combined", "1"),
]


def post_image(url: str, path: Path, mode: str, ocr: str, timeout: int) -> dict:
    boundary = "----skanervinaeval"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        "Content-Type: image/webp\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/recognize",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Scanner-Debug": "1",
            "X-Scanner-Mode": mode,
            "X-Scanner-OCR": ocr,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def compact(result: dict) -> dict:
    rec = result.get("recognition") or {}
    top5 = ((result.get("ranking") or {}).get("top5") or rec.get("candidates") or [])[:5]
    slim = []
    for item in top5:
        slim.append(
            {
                "slug": item.get("slug"),
                "name": item.get("name"),
                "winery": item.get("winery"),
                "score": item.get("score"),
                "siglip": item.get("siglip"),
                "ocr_delta": item.get("ocr_delta"),
                "full_score": item.get("full_score"),
                "crop_score": item.get("crop_score"),
                "best_view": item.get("best_view"),
            }
        )
    return {
        "status": result.get("status"),
        "slug": result.get("slug") or (slim[0]["slug"] if slim else None),
        "score": rec.get("similarity") or result.get("confidence"),
        "margin": rec.get("margin"),
        "method": rec.get("method"),
        "ocr": rec.get("ocr"),
        "ocr_delta": rec.get("ocr_delta"),
        "ocr_text": (rec.get("ocr_text") or "")[:180],
        "ocr_corroborated": rec.get("ocr_corroborated"),
        "detect": (rec.get("label_detection") or {}).get("method"),
        "query_view": rec.get("query_view"),
        "crop_quality": rec.get("crop_quality"),
        "timings_ms": rec.get("timings_ms"),
        "top5": slim,
        "error": None,
    }


def summarize(rows: list[dict], key: str) -> dict:
    statuses = Counter(row["configs"][key]["status"] for row in rows if row["configs"][key].get("status"))
    scores = [float(row["configs"][key]["score"] or 0) for row in rows if row["configs"][key].get("score") is not None]
    errors = sum(1 for row in rows if row["configs"][key].get("error"))
    return {
        "n": len(rows),
        "errors": errors,
        "status": dict(statuses),
        "matched": statuses.get("matched", 0),
        "uncertain": statuses.get("uncertain", 0),
        "unknown": statuses.get("unknown", 0),
        "matched_pct": round(100 * statuses.get("matched", 0) / max(1, len(rows)), 1),
        "mean_score": round(sum(scores) / max(1, len(scores)), 4),
        "median_score": round(median(scores), 4) if scores else None,
        "ocr_bonus": sum(1 for row in rows if (row["configs"][key].get("ocr_delta") or 0) > 0),
    }


def agreement(rows: list[dict], a: str, b: str) -> dict:
    same = 0
    diffs = []
    for row in rows:
        left = row["configs"][a].get("slug")
        right = row["configs"][b].get("slug")
        if left and right and left == right:
            same += 1
        elif left or right:
            diffs.append({"file": row["file"], a: left, b: right})
    return {"same": same, "same_pct": round(100 * same / max(1, len(rows)), 1), "diff_n": len(diffs), "diffs": diffs[:25]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    files = sorted(
        path for path in args.folder.iterdir() if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if args.limit:
        files = files[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.out.exists():
        try:
            previous = json.loads(args.out.read_text(encoding="utf-8"))
            done = {row["file"]: row for row in previous.get("rows") or []}
        except Exception:
            done = {}
    rows = []
    total = len(files) * len(CONFIGS)
    step = 0
    started = time.time()
    for path in files:
        record = done.get(path.name) or {"file": path.name, "configs": {}}
        for name, mode, ocr in CONFIGS:
            step += 1
            if name in record.get("configs") and not record["configs"][name].get("error"):
                continue
            t0 = time.time()
            try:
                result = post_image(args.url, path, mode, ocr, args.timeout)
                row = compact(result)
            except Exception as exc:
                row = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "slug": None, "score": None}
            row["elapsed_s"] = round(time.time() - t0, 2)
            record.setdefault("configs", {})[name] = row
            print(
                f"{step}/{total} {path.name} {name} {row.get('status')} {row.get('score')} {row.get('slug')}",
                flush=True,
            )
        rows.append(record)
        payload = {"rows": rows, "partial": True}
        args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary = {name: summarize(rows, name) for name, _, _ in CONFIGS}
    names = [name for name, _, _ in CONFIGS]
    pairs = {}
    if "crop_noocr" in names and "full_noocr" in names:
        pairs["crop vs full (no OCR)"] = agreement(rows, "crop_noocr", "full_noocr")
    if "crop_ocr" in names and "crop_noocr" in names:
        pairs["crop OCR vs crop no OCR"] = agreement(rows, "crop_ocr", "crop_noocr")
    if "full_ocr" in names and "full_noocr" in names:
        pairs["full OCR vs full no OCR"] = agreement(rows, "full_ocr", "full_noocr")
    if "merge_ocr" in names and "merge_noocr" in names:
        pairs["merge OCR vs merge no OCR"] = agreement(rows, "merge_ocr", "merge_noocr")
    if "merge_ocr" in names and "crop_ocr" in names:
        pairs["merge OCR vs crop OCR"] = agreement(rows, "merge_ocr", "crop_ocr")
    if "merge_ocr" in names and "full_ocr" in names:
        pairs["merge OCR vs full OCR"] = agreement(rows, "merge_ocr", "full_ocr")
    report = {
        "folder": str(args.folder),
        "url": args.url,
        "n": len(rows),
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "agreement": {key: {k: v for k, v in value.items() if k != "diffs"} | {"sample_diffs": value["diffs"][:8]} for key, value in pairs.items()},
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "agreement": report["agreement"]}, ensure_ascii=False, indent=2), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
