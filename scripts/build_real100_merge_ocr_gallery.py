"""Rebuild Merge+OCR rows with query crop and catalog bottle/label stills."""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SUFFIXES = {".webp", ".jpg", ".jpeg", ".png"}


def post_image(url: str, path: Path, timeout: int) -> dict:
    boundary = "----skanervinagallery"
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
            "X-Scanner-Mode": "combined",
            "X-Scanner-OCR": "1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def write_jpeg(dest: Path, b64: str | None) -> str | None:
    if not b64:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    return dest.name


def fetch_url(dest: Path, url: str | None, timeout: int) -> str | None:
    if not url:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "skanervina-eval"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            dest.write_bytes(response.read())
        return dest.name
    except Exception:
        return None


def pct(value) -> str | None:
    if value is None:
        return None
    return round(float(value) * 100, 1)


def compact(result: dict, stem: str, out: Path, timeout: int) -> dict:
    rec = result.get("recognition") or {}
    top = ((result.get("ranking") or {}).get("top5") or rec.get("candidates") or [])[:5]
    compared = (rec.get("compared_with") or [{}])[0]
    query_crop = write_jpeg(out / "crops" / f"{stem}.jpg", rec.get("crop_jpeg_base64"))
    catalog_label = write_jpeg(out / "catalog-label" / f"{stem}.jpg", compared.get("label_jpeg_base64"))
    catalog_bottle = fetch_url(out / "catalog-bottle" / f"{stem}.jpg", compared.get("image_url") or (top[0].get("image_url") if top else None), timeout)
    status = result.get("status")
    understood = "карточка" if status == "matched" else ("slug без карточки" if status == "uncertain" else "не понял")
    return {
        "file": None,
        "status": status,
        "understood": understood,
        "slug": result.get("slug") or (top[0].get("slug") if top else None),
        "name": (top[0].get("name") if top else None) or compared.get("name"),
        "winery": (top[0].get("winery") if top else None) or compared.get("winery"),
        "score_pct": pct(rec.get("similarity") or result.get("confidence")),
        "bottle_pct": pct(compared.get("full_score") or (top[0].get("full_score") if top else None)),
        "label_pct": pct(compared.get("crop_score") or (top[0].get("crop_score") if top else None)),
        "best_view": compared.get("best_view") or (top[0].get("best_view") if top else None),
        "margin": rec.get("margin"),
        "detect": (rec.get("label_detection") or {}).get("method"),
        "ocr": rec.get("ocr"),
        "ocr_delta": rec.get("ocr_delta"),
        "ocr_text": (rec.get("ocr_text") or "").replace("\n", " ").strip()[:220],
        "ocr_corroborated": rec.get("ocr_corroborated"),
        "query_crop": query_crop,
        "catalog_bottle": catalog_bottle,
        "catalog_label": catalog_label,
        "top5": [
            {
                "slug": item.get("slug"),
                "name": item.get("name"),
                "score_pct": pct(item.get("score")),
                "bottle_pct": pct(item.get("full_score")),
                "label_pct": pct(item.get("crop_score")),
            }
            for item in top
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    files = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.json"
    done = {}
    if rows_path.exists():
        try:
            done = {row["file"]: row for row in json.loads(rows_path.read_text())["rows"]}
        except Exception:
            done = {}
    rows = []
    for index, path in enumerate(files, 1):
        if path.name in done and done[path.name].get("query_crop"):
            rows.append(done[path.name])
            print(f"{index}/{len(files)} skip {path.name}", flush=True)
            continue
        stem = path.stem
        try:
            result = post_image(args.url, path, args.timeout)
            row = compact(result, stem, args.out, args.timeout)
        except Exception as exc:
            row = {"status": "error", "understood": "ошибка", "error": f"{type(exc).__name__}: {exc}"}
        row["file"] = path.name
        rows.append(row)
        rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"{index}/{len(files)} {path.name} {row.get('status')} {row.get('score_pct')} {row.get('slug')}", flush=True)
        time.sleep(0.05)
    rows_path.write_text(json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", rows_path, flush=True)


if __name__ == "__main__":
    main()
