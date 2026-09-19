"""Benchmark automatic label recognition in image, OCR and combined modes.

The three public query photos have no published expected slugs, so their
accuracy is intentionally reported as unavailable.  The catalogue reference
images have known slugs and are reported as a smoke-test accuracy set; they
are not a substitute for a labelled real-world test set.

Run against a CV-enabled API with an administrator token:

    PYTHONPATH=backend backend/.venv-cv/bin/python scripts/benchmark_recognition.py
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time
from urllib.parse import unquote, urlparse

import httpx
from dotenv import dotenv_values

from app.catalog import WineCatalog
from app.settings import settings


MODES = {
    "image_full": "Изображение · весь кадр",
    "image_auto": "Изображение · автоматический crop",
    "image_auto_enhanced": "Изображение · автоматический crop + enhancement",
    "ocr_full": "OCR · весь кадр",
    "ocr_auto": "OCR · автоматический crop",
    "ocr_auto_enhanced": "OCR · автоматический crop + enhancement",
    "combined": "Совместный · image + OCR на автоматическом crop",
}

# Evaluation-only boxes.  They are not used by the application and are kept
# here solely to measure whether the automatic detector found the label area.
LABEL_BOXES = {
    "019c68d0.jpg": (0.25, 0.35, 0.79, 0.83),
    "02eef911.webp": (0.26, 0.19, 0.71, 0.79),
    "096ca74e.jpg": (0.21, 0.24, 0.83, 0.73),
}


def bbox_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return round(intersection / union, 4) if union else 0.0


def percentile(values, value):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * value
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    fraction = rank - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * fraction, 1)


def summarize(samples, expected=None):
    latencies = [sample["wall_ms"] for sample in samples]
    server_totals = [sample.get("recognition", {}).get("timings_ms", {}).get("total") for sample in samples]
    server_totals = [value for value in server_totals if isinstance(value, (int, float))]
    detection = [sample.get("recognition", {}).get("timings_ms", {}).get("label_detection") for sample in samples]
    detection = [value for value in detection if isinstance(value, (int, float))]
    visual = [sample.get("recognition", {}).get("timings_ms", {}).get("visual") for sample in samples]
    visual = [value for value in visual if isinstance(value, (int, float))]
    ocr = [sample.get("recognition", {}).get("timings_ms", {}).get("ocr") for sample in samples]
    ocr = [value for value in ocr if isinstance(value, (int, float))]
    last = samples[-1]
    metrics = last.get("recognition", {})
    result = {
        "status": last.get("status"),
        "slug": last.get("slug"),
        "name": (last.get("wine") or {}).get("name"),
        "confidence": last.get("confidence"),
        "similarity": metrics.get("similarity"),
        "ocr_text": metrics.get("ocr_text"),
        "candidate_slugs": [item.get("slug") for item in metrics.get("candidates", [])],
        "label_detection": metrics.get("label_detection"),
        "latency_ms": {
            "median": round(median(latencies), 1),
            "p95": percentile(latencies, 0.95),
        },
        "server_ms": {
            "total_median": round(median(server_totals), 1) if server_totals else None,
            "label_detection_median": round(median(detection), 1) if detection else None,
            "visual_median": round(median(visual), 1) if visual else None,
            "ocr_median": round(median(ocr), 1) if ocr else None,
        },
        "samples": samples,
    }
    if expected is not None:
        result["expected_slug"] = expected
        candidates = result["candidate_slugs"]
        rank = candidates.index(expected) + 1 if expected in candidates else None
        result["correct"] = last.get("slug") == expected
        result["top3_correct"] = rank is not None and rank <= 3
        result["reciprocal_rank"] = round(1 / rank, 4) if rank else 0.0
    return result


def mode_summary(rows, expected_key="expected_slug"):
    entries = [row for row in rows if row.get(expected_key) is not None]
    known = [row for row in entries if row.get("correct") is not None]
    statuses = Counter(row.get("status") for row in rows)
    latencies = [row["latency_ms"]["median"] for row in rows]
    correct = sum(bool(row.get("correct")) for row in known)
    top3 = sum(bool(row.get("top3_correct")) for row in known)
    mrr = sum(float(row.get("reciprocal_rank", 0)) for row in known)
    crop_ious = [row["label_detection_iou"] for row in rows if row.get("label_detection_iou") is not None]
    return {
        "cases": len(rows),
        "status_counts": dict(statuses),
        "unknown_rate": round(statuses.get("unknown", 0) / len(rows), 4) if rows else None,
        "latency_ms": {
            "median": round(median(latencies), 1) if latencies else None,
            "p95": percentile(latencies, 0.95),
        },
        "accuracy": {
            "ground_truth_available": bool(known),
            "labelled_cases": len(known),
            "top1": round(correct / len(known), 4) if known else None,
            "top3": round(top3 / len(known), 4) if known else None,
            "mrr": round(mrr / len(known), 4) if known else None,
        },
        "label_detection": {
            "ground_truth_available": bool(crop_ious),
            "mean_iou": round(sum(crop_ious) / len(crop_ious), 4) if crop_ious else None,
        },
    }


def load_query_labels(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    raise SystemExit("--labels must point to a JSON object: {\"filename.jpg\": \"expected-slug\"}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--output", type=Path, default=Path("reports/recognition/benchmark-modes.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--labels", type=Path, help="Optional JSON mapping query filename to expected slug")
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        raise SystemExit("--repeats must be between 1 and 20")

    local_env = dotenv_values("backend/.env.local")
    token = os.getenv("SCANNER_ADMIN_TOKEN") or local_env.get("SCANNER_ADMIN_TOKEN")
    if not token:
        raise SystemExit("Set SCANNER_ADMIN_TOKEN")
    labels = load_query_labels(args.labels)
    catalog = WineCatalog.from_csv(settings.catalog_csv, settings.image_base_url)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "repeats_per_case": args.repeats,
        "modes": MODES,
        "notes": [
            "Production combined mode uses the automatic crop and runs image retrieval plus OCR.",
            "The three supplied query photos do not include public ground-truth slugs; query accuracy is null unless --labels is supplied.",
            "Reference accuracy uses catalogue self-retrieval images and is a smoke test, not independent real-world accuracy.",
            "Requests are sequential; latency includes HTTP and response serialization.",
        ],
        "queries": [],
        "references": [],
    }

    def request(data, filename, mime, mode):
        started = time.perf_counter()
        response = client.post(
            "/v1/recognize",
            files={"image": (filename, data, mime)},
            headers={"X-Scanner-Admin": token, "X-Scanner-Mode": mode, "X-Scanner-Benchmark": "1"},
        )
        response.raise_for_status()
        result = response.json()
        return {"wall_ms": round((time.perf_counter() - started) * 1000, 1), **result}

    with httpx.Client(base_url=args.url, timeout=90, headers={"X-Scanner-Admin": token}) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        report["health"] = health.json()
        query_dir = Path("Датасет/eval/queries")
        for path in sorted(query_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif"}:
                continue
            import mimetypes

            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            data = path.read_bytes()
            case = {
                "file": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "expected_slug": labels.get(path.name),
                "modes": {},
            }
            for mode in MODES:
                samples = [request(data, path.name, mime, mode) for _ in range(args.repeats)]
                case["modes"][mode] = summarize(samples, labels.get(path.name))
                detected_box = case["modes"][mode].get("label_detection", {}).get("bbox")
                if path.name in LABEL_BOXES and detected_box:
                    case["modes"][mode]["label_detection_iou"] = bbox_iou(detected_box, LABEL_BOXES[path.name])
                print(
                    json.dumps(
                        {"file": path.name, "mode": mode, "ms": case["modes"][mode]["latency_ms"]["median"], "slug": case["modes"][mode]["slug"]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            report["queries"].append(case)

        images = Path("Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads")
        self_check = Path("backend/data/catalog-self-check.json")
        if self_check.exists():
            import mimetypes

            for item in json.loads(self_check.read_text(encoding="utf-8")):
                wine = catalog.get(item["expected"])
                if not wine:
                    continue
                filename = unquote(Path(urlparse(wine.direct_image_url or "").path).name) or wine.image_name
                image_path = images / filename
                if not image_path.is_file():
                    continue
                data = image_path.read_bytes()
                mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                case = {"file": str(image_path), "expected_slug": wine.slug, "modes": {}}
                for mode in MODES:
                    samples = [request(data, image_path.name, mime, mode) for _ in range(args.repeats)]
                    case["modes"][mode] = summarize(samples, wine.slug)
                report["references"].append(case)

    report["summary"] = {
        "queries": {
            mode: mode_summary([case["modes"][mode] for case in report["queries"]])
            for mode in MODES
        },
        "references": {
            mode: mode_summary([case["modes"][mode] for case in report["references"]])
            for mode in MODES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
