#!/usr/bin/env python3
"""Enrich the local wine CSV with public data from vino-svoe.ru.

The source CSV is never overwritten. The default output is a sibling file with
"_enriched" in its name. Requests are deliberately throttled and use only
public HTML pages. The scraper keeps source URLs and a run timestamp next to
every imported field.
"""

import argparse
import csv
import json
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


BASE_URL = "https://vino-svoe.ru"
WINES_URL = f"{BASE_URL}/wines"
IMAGE_BASE_URL = (
    "https://api.vino-svoe.ru/v1/img/str-api/800/800/resize/uploads/"
)
USER_AGENT = "svoe-vino-scanner-data-research/0.1"
DEFAULT_INPUT = Path("Датасет/strapi_output0709.csv")
DEFAULT_OUTPUT = Path("Датасет/strapi_output0709_enriched.csv")
DEFAULT_REPORT = Path("Датасет/strapi_output0709_enriched_report.json")


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def fetch_html(url: str, timeout: int = 45, retries: int = 5) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def image_url_from_source(source_url: str) -> str:
    path = unquote(urlparse(source_url).path)
    match = re.search(r"/uploads/([^/]+)$", path)
    if not match:
        return ""
    return IMAGE_BASE_URL + quote(match.group(1), safe="")


def first_image_url(node) -> str:
    image = node.select_one("img")
    if not image:
        return ""
    source = image.get("src") or ""
    if not source:
        srcset = image.get("srcset") or ""
        source = srcset.split(",")[-1].strip().split(" ")[0]
    return source


def parse_rating(value: str) -> str:
    match = re.search(r"(?:Народный рейтинг\s*)?([0-5](?:[.,]\d+)?)", value)
    return match.group(1).replace(",", ".") if match else ""


def parse_list_page(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    records: List[Dict[str, str]] = []
    seen = set()
    for item in soup.select("a.wine-item"):
        href = item.get("href") or ""
        if not href.startswith("/wines/"):
            continue
        slug = href.rstrip("/").rsplit("/", 1)[-1]
        if not slug or slug in seen:
            continue
        seen.add(slug)
        title = clean_text(item.select_one(".wine-item__title").get_text(" ", strip=True)) if item.select_one(".wine-item__title") else ""
        manufacturer = clean_text(item.select_one(".wine-item__manufacturer").get_text(" ", strip=True)) if item.select_one(".wine-item__manufacturer") else ""
        rating_node = item.select_one(".wine-item-rating__text")
        rating = parse_rating(rating_node.get_text(" ", strip=True) if rating_node else "")
        source_image = first_image_url(item)
        records.append(
            {
                "slug": slug,
                "title": title,
                "manufacturer": manufacturer,
                "public_rating": rating,
                "image_url": image_url_from_source(source_image),
                "source_url": urljoin(BASE_URL, href),
            }
        )
    return records


def parse_page_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    pages = []
    for link in soup.select(".core-pagination__list a[href]"):
        match = re.search(r"[?&]page=(\d+)", link.get("href") or "")
        if match:
            pages.append(int(match.group(1)))
    return max(pages, default=1)


def labeled_details(soup: BeautifulSoup) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for block in soup.select(".wine-detail-info__detail"):
        label_node = block.select_one(".wine-detail-info__detail-label")
        if not label_node:
            continue
        label = clean_text(label_node.get_text(" ", strip=True))
        values = [clean_text(node.get_text(" ", strip=True)) for node in block.select(".wine-detail-info__detail-value")]
        if label and values and label not in result:
            result[label] = values
    return result


def labeled_cards(soup: BeautifulSoup) -> Dict[str, str]:
    result: Dict[str, str] = {}
    selectors = ".wine-hero-block__card, .wine-mobile-hero-block__card"
    for block in soup.select(selectors):
        label_node = block.select_one(".wine-hero-block__card-label")
        value_node = block.select_one(".wine-hero-block__card-value")
        if not label_node or not value_node:
            continue
        label = clean_text(label_node.get_text(" ", strip=True))
        value = clean_text(value_node.get_text(" ", strip=True))
        if label and value and label not in result:
            result[label] = value
    return result


def parse_detail_page(html: str, record: Dict[str, str]) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    result = dict(record)
    title_node = soup.select_one(".wine-main-title-block__title")
    manufacturer_node = soup.select_one(".wine-main-title-block__manufacturer")
    rating_node = soup.select_one(".wine-main-title-block__rating-text")
    if title_node:
        result["title"] = clean_text(title_node.get_text(" ", strip=True))
    if manufacturer_node:
        result["manufacturer"] = clean_text(manufacturer_node.get_text(" ", strip=True))
    if rating_node:
        result["public_rating"] = parse_rating(rating_node.get_text(" ", strip=True))

    details = labeled_details(soup)
    category_values = details.get("Категория и цвет", [])
    result["region"] = (details.get("Регион") or [""])[0]
    result["grapes"] = (details.get("Сорта винограда") or [""])[0]
    result["category"] = category_values[0] if category_values else ""
    result["color"] = category_values[1] if len(category_values) > 1 else ""

    cards = labeled_cards(soup)
    result["temperature"] = cards.get("Температура подачи", "")
    result["alcohol"] = cards.get("Крепость вина", "")
    result["dishes_json"] = json.dumps(
        list(dict.fromkeys(clean_text(node.get_text(" ", strip=True)) for node in soup.select(".wine-dish-item__name") if clean_text(node.get_text(" ", strip=True)))),
        ensure_ascii=False,
    )
    description_node = soup.select_one(".wine-page__description")
    result["description"] = description_node.get_text("\n", strip=True) if description_node else ""
    bottle_node = soup.select_one(".wine-hero-block__bottle, .wine-mobile-hero-block__bottle-image")
    if bottle_node:
        result["image_url"] = image_url_from_source(bottle_node.get("src") or first_image_url(bottle_node))

    # The detail page currently exposes public rating, but no numeric
    # Roskachestvo value in the rendered card. Keep the field empty rather
    # than confusing the two rating systems.
    result["quality_rating"] = ""
    result["status"] = "ok"
    return result


def fetch_detail(record: Dict[str, str], delay: float) -> Tuple[str, Dict[str, str]]:
    if delay > 0:
        time.sleep(delay)
    try:
        return record["slug"], parse_detail_page(fetch_html(record["source_url"]), record)
    except Exception as error:
        failed = dict(record)
        failed["status"] = "detail_fetch_error"
        failed["error"] = str(error)
        return record["slug"], failed


def load_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def enrich_rows(rows: Iterable[Dict[str, str]], site_records: Dict[str, Dict[str, str]], scraped_at: str) -> List[Dict[str, str]]:
    extra_fields = [
        "svoe_vino_source_url",
        "svoe_vino_public_rating",
        "svoe_vino_quality_rating",
        "svoe_vino_region",
        "svoe_vino_grapes",
        "svoe_vino_category",
        "svoe_vino_color",
        "svoe_vino_temperature",
        "svoe_vino_alcohol",
        "svoe_vino_dishes_json",
        "svoe_vino_description",
        "svoe_vino_image_url",
        "svoe_vino_scraped_at",
        "svoe_vino_scrape_status",
    ]
    enriched: List[Dict[str, str]] = []
    for original in rows:
        row = dict(original)
        site = site_records.get((row.get("Slug") or "").strip())
        row.update({field: "" for field in extra_fields})
        if not site:
            row["svoe_vino_scraped_at"] = scraped_at
            row["svoe_vino_scrape_status"] = "not_found_in_site"
        else:
            row.update(
                {
                    "svoe_vino_source_url": site.get("source_url", ""),
                    "svoe_vino_public_rating": site.get("public_rating", ""),
                    "svoe_vino_quality_rating": site.get("quality_rating", ""),
                    "svoe_vino_region": site.get("region", ""),
                    "svoe_vino_grapes": site.get("grapes", ""),
                    "svoe_vino_category": site.get("category", ""),
                    "svoe_vino_color": site.get("color", ""),
                    "svoe_vino_temperature": site.get("temperature", ""),
                    "svoe_vino_alcohol": site.get("alcohol", ""),
                    "svoe_vino_dishes_json": site.get("dishes_json", "[]"),
                    "svoe_vino_description": site.get("description", ""),
                    "svoe_vino_image_url": site.get("image_url", ""),
                    "svoe_vino_scraped_at": scraped_at,
                    "svoe_vino_scrape_status": site.get("status", "ok"),
                }
            )
        enriched.append(row)
    return enriched


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pages", type=int, default=0, help="Number of catalog pages; 0 reads all pages reported by the site")
    parser.add_argument("--max-wines", type=int, default=0, help="Limit detail pages for a trial run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds before each detail request")
    parser.add_argument("--skip-details", action="store_true", help="Only collect catalog list pages")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("Refusing to overwrite the source CSV. Choose a different --output.")
    fieldnames, original_rows = load_rows(args.input)
    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    first_page = parse_list_page(fetch_html(WINES_URL))
    if not first_page:
        raise SystemExit("The catalog page returned no wine cards.")
    page_count = args.pages or parse_page_count(fetch_html(WINES_URL))
    site_records = {record["slug"]: record for record in first_page}
    for page in range(2, page_count + 1):
        time.sleep(args.delay)
        page_records = parse_list_page(fetch_html(f"{WINES_URL}?page={page}"))
        site_records.update({record["slug"]: record for record in page_records})
        if page == 2 or page % 10 == 0 or page == page_count:
            print(f"list pages: {page}/{page_count}, wines: {len(site_records)}", flush=True)

    records_for_details = list(site_records.values())
    if args.max_wines:
        records_for_details = records_for_details[: args.max_wines]
    if not args.skip_details:
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(fetch_detail, record, args.delay) for record in records_for_details]
            for future in as_completed(futures):
                slug, detail = future.result()
                site_records[slug] = detail
                completed += 1
                if completed == 1 or completed % 50 == 0 or completed == len(futures):
                    print(f"detail pages: {completed}/{len(futures)}", flush=True)

    enriched = enrich_rows(original_rows, site_records, scraped_at)
    output_fields = list(fieldnames)
    for field in [
        "svoe_vino_source_url", "svoe_vino_public_rating", "svoe_vino_quality_rating",
        "svoe_vino_region", "svoe_vino_grapes", "svoe_vino_category", "svoe_vino_color",
        "svoe_vino_temperature", "svoe_vino_alcohol", "svoe_vino_dishes_json",
        "svoe_vino_description", "svoe_vino_image_url", "svoe_vino_scraped_at",
        "svoe_vino_scrape_status",
    ]:
        if field not in output_fields:
            output_fields.append(field)
    write_csv(args.output, output_fields, enriched)

    matched = sum(1 for row in enriched if row.get("svoe_vino_source_url"))
    rated = sum(1 for row in enriched if row.get("svoe_vino_public_rating"))
    detail_errors = sum(1 for record in site_records.values() if record.get("status") == "detail_fetch_error")
    report = {
        "source": WINES_URL,
        "scraped_at": scraped_at,
        "input": str(args.input),
        "output": str(args.output),
        "catalog_pages_read": page_count,
        "site_wines_seen": len(site_records),
        "input_rows": len(original_rows),
        "input_rows_matched_by_slug": matched,
        "input_rows_with_public_rating": rated,
        "detail_errors": detail_errors,
        "note": "The site currently exposes public rating in the wine card. No numeric Roskachestvo value was present in the detail HTML, so that field is kept empty.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
