#!/usr/bin/env python3
"""Collect wine-review photos from public Otzovik pages.

The scraper is intentionally conservative: it follows only links discovered
on the requested category/product pages, uses one request at a time, obeys
robots.txt when it is available, stops on a block/CAPTCHA response, and keeps
the source URL next to every downloaded image.  The source CSV/datasets in the
repository are never modified.

Example:

    python3 scripts/scrape_otzovik_wines.py \
        --output-dir Датасет/otzovik_wines \
        --delay 1.2

For a small smoke run use ``--max-products 2 --max-reviews 5``.  Use
``--no-download-images`` when only the mapping and source URLs are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import socket
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup


BASE_URL = "https://otzovik.com"
CATEGORY_URL = f"{BASE_URL}/food/alcohol/wines/"
USER_AGENT = "otzovik-wine-review-photo-dataset/0.1 (+public-research; respectful-rate-limit)"
IMAGE_HOST = "i.otzovik.com"
REVIEW_URL_RE = re.compile(r"^/review_(\d+)\.html/?$")
PRODUCT_URL_RE = re.compile(r"^/reviews/([^/]+)/?$")
PRODUCT_PAGINATION_URL_RE = re.compile(r"^/reviews/([^/]+)/([0-9]+)/?$")
GALLERY_URL_RE = re.compile(r"^/reviews/([^/]+)/gallery(?:/(\d+))?/?$")
CATEGORY_URL_RE = re.compile(r"^/food/alcohol/wines(?:/(\d+))?/?$")
THUMB_SUFFIX_RE = re.compile(r"_(?:s|m|xs|thumb)(\.[a-z0-9]+)$", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def normalize_url(url: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, url))
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"otzovik.com", "www.otzovik.com", IMAGE_HOST}:
        return ""
    scheme = "https"
    host = parsed.netloc.lower()
    if host == "www.otzovik.com":
        host = "otzovik.com"
    return f"{scheme}://{host}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")


def safe_slug(value: str, fallback: str = "unknown") -> str:
    value = unquote(value).lower()
    value = re.sub(r"[^a-z0-9а-яё]+", "-", value, flags=re.IGNORECASE).strip("-")
    return value[:120] or fallback


def html_encoding(headers) -> str:
    content_type = headers.get_content_charset() if headers else None
    return content_type or "utf-8"


def response_is_blocked(status: int, content: bytes = b"") -> bool:
    if status in {403, 429, 503}:
        return True
    sample = content[:200_000].lower()
    return any(marker in sample for marker in (b"captcha", b"access denied", b"too many requests"))


class Fetcher:
    def __init__(self, delay: float, timeout: float, retries: int, user_agent: str):
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.user_agent = user_agent
        self.last_request_at = 0.0

    def _throttle(self) -> None:
        remaining = self.delay - (time.monotonic() - self.last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str, accept: str = "*/*", referer: str = "") -> Tuple[bytes, object]:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            self._throttle()
            request = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": accept,
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
                    **({"Referer": referer} if referer else {}),
                },
            )
            self.last_request_at = time.monotonic()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    content = response.read()
                    if response_is_blocked(response.status, content):
                        raise RuntimeError(f"site block/CAPTCHA detected for {url} (HTTP {response.status})")
                    return content, response.headers
            except HTTPError as error:
                body = error.read(32_000) if error.fp else b""
                if response_is_blocked(error.code, body):
                    raise RuntimeError(f"site block/CAPTCHA detected for {url} (HTTP {error.code})") from error
                last_error = error
            except (URLError, TimeoutError, socket.timeout, OSError, RuntimeError) as error:
                last_error = error
                if isinstance(error, RuntimeError) and "block/CAPTCHA" in str(error):
                    raise
            if attempt + 1 < self.retries:
                time.sleep(min(30.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"could not fetch {url}: {last_error}")

    def html(self, url: str, referer: str = "") -> str:
        data, headers = self.get(url, accept="text/html,application/xhtml+xml", referer=referer)
        return data.decode(html_encoding(headers), errors="replace")


def same_site_path(url: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, url))
    return parsed.path or "/"


def is_category_url(url: str) -> bool:
    return bool(CATEGORY_URL_RE.match(same_site_path(url)))


def is_product_url(url: str) -> bool:
    return bool(PRODUCT_URL_RE.match(same_site_path(url)))


def is_product_page_url(url: str) -> bool:
    path = same_site_path(url)
    return bool(PRODUCT_URL_RE.match(path) or PRODUCT_PAGINATION_URL_RE.match(path))


def is_gallery_url(url: str) -> bool:
    return bool(GALLERY_URL_RE.match(same_site_path(url)))


def gallery_url_for(product_url: str, page: int = 1) -> str:
    slug = product_slug(product_url)
    suffix = "" if page <= 1 else f"{page}/"
    return f"{BASE_URL}/reviews/{slug}/gallery/{suffix}"


def product_slug(url: str) -> str:
    path = same_site_path(url)
    match = PRODUCT_URL_RE.match(path) or PRODUCT_PAGINATION_URL_RE.match(path) or GALLERY_URL_RE.match(path)
    return match.group(1) if match else safe_slug(same_site_path(url))


def review_id(url: str) -> str:
    match = REVIEW_URL_RE.match(same_site_path(url))
    return match.group(1) if match else safe_slug(url)


def is_review_url(url: str) -> bool:
    return bool(REVIEW_URL_RE.match(same_site_path(url)))


def image_url_candidate(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(urljoin(BASE_URL, value))
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host != IMAGE_HOST or not path.startswith("/objects/"):
        return ""
    if "/avatar/" in path:
        return ""
    return prefer_full_image(normalize_url(value))


def prefer_full_image(url: str) -> str:
    parsed = urlparse(url)
    path = THUMB_SUFFIX_RE.sub(r"\1", parsed.path)
    return parsed._replace(path=path, query="").geturl()


def iter_srcset(value: str) -> Iterator[str]:
    for item in (value or "").split(","):
        candidate = item.strip().split(" ", 1)[0]
        if candidate:
            yield candidate


def collect_object_images(root) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()
    for node in root.find_all(["img", "source", "a"]):
        values = [node.get("href", ""), node.get("src", ""), node.get("data-src", ""), node.get("data-original", "")]
        values.extend(iter_srcset(node.get("srcset", "")))
        for value in values:
            candidate = image_url_candidate(value)
            if candidate and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    return urls


def extract_review_images(soup: BeautifulSoup) -> List[str]:
    """Return full-size user photos from a review, skipping icons and avatars."""

    roots = []
    for selector in (
        ".review-photos",
        ".review-photo",
        ".item-photos",
        ".review-photo-list",
        '[itemprop="reviewBody"]',
        ".review-body",
        ".review__body",
        ".review-item__content",
        ".review-content",
        "article",
        "main",
    ):
        roots.extend(soup.select(selector))
    if not roots:
        roots = [soup]

    urls: List[str] = []
    seen: Set[str] = set()
    for root in roots:
        for candidate in collect_object_images(root):
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    return urls


def extract_gallery_images(html: str, page_url: str) -> List[Dict[str, str]]:
    """Photos people uploaded, listed on the product gallery tab."""

    soup = BeautifulSoup(html, "html.parser")
    records: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for node in soup.select("a[href], img[src], img[data-src]"):
        review_href = normalize_url(urljoin(page_url, node.get("href", "")))
        linked_review = review_href if is_review_url(review_href) else ""
        values = [node.get("href", ""), node.get("src", ""), node.get("data-src", ""), node.get("data-original", "")]
        values.extend(iter_srcset(node.get("srcset", "")))
        if node.name == "a":
            for child in node.find_all("img"):
                values.extend([child.get("src", ""), child.get("data-src", ""), child.get("data-original", "")])
                values.extend(iter_srcset(child.get("srcset", "")))
        for value in values:
            image_url = image_url_candidate(value)
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            records.append({"source_image_url": image_url, "review_url": linked_review})
    return records


def extract_gallery_page_links(html: str, page_url: str) -> List[str]:
    product = product_slug(page_url)
    result: List[str] = []
    seen: Set[str] = set()
    for anchor in BeautifulSoup(html, "html.parser").select("a[href]"):
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if is_gallery_url(url) and product_slug(url) == product and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def find_title(soup: BeautifulSoup, selectors: Sequence[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            value = clean_text(node.get_text(" ", strip=True))
            if value:
                return value
    return ""


def extract_product_links(html: str, page_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    records: Dict[str, Dict[str, str]] = {}
    header_links = soup.select("h1 a[href], h2 a[href], h3 a[href], h4 a[href]")
    candidates = header_links or soup.select("a[href]")
    for anchor in candidates:
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if not is_product_url(url):
            continue
        title = clean_text(anchor.get_text(" ", strip=True))
        if not title or title.lower() in {"добавить отзыв", "читать отзывы"} or "добавить отзыв" in title.lower():
            continue
        slug = product_slug(url)
        parent = anchor.parent
        context = clean_text(parent.get_text(" ", strip=True)) if parent else ""
        rating = ""
        rating_match = re.search(r"(?<!\d)([0-5](?:[.,]\d+)?)(?!\d)", context)
        if rating_match:
            rating = rating_match.group(1).replace(",", ".")
        count_match = re.search(r"(\d+)\s+отзыв", context, flags=re.IGNORECASE)
        record = records.setdefault(slug, {
            "product_slug": slug,
            "product_name": title,
            "product_url": url,
            "catalog_rating": rating,
            "catalog_review_count": count_match.group(1) if count_match else "",
        })
        if not record["product_name"]:
            record["product_name"] = title
    return list(records.values())


def extract_category_links(html: str, page_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    result: List[str] = []
    seen: Set[str] = set()
    for anchor in soup.select("a[href]"):
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if is_category_url(url) and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def extract_product_page_links(html: str, page_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    result: List[str] = []
    seen: Set[str] = set()
    for anchor in soup.select("a[href]"):
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if is_product_page_url(url) and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def extract_review_links(html: str, page_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    result: List[str] = []
    seen: Set[str] = set()
    for anchor in soup.select("a[href]"):
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if is_review_url(url) and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def extract_pagination_product_pages(html: str, page_url: str) -> List[str]:
    product = product_slug(page_url)
    result: List[str] = []
    for anchor in BeautifulSoup(html, "html.parser").select("a[href]"):
        url = normalize_url(urljoin(page_url, anchor.get("href", "")))
        if is_product_page_url(url) and product_slug(url) == product and url != page_url:
            result.append(url)
    return list(dict.fromkeys(result))


@dataclass
class Product:
    product_slug: str
    product_name: str
    product_url: str
    catalog_rating: str = ""
    catalog_review_count: str = ""
    review_count_scraped: int = 0
    image_count: int = 0
    scrape_status: str = "ok"
    error: str = ""


@dataclass
class Review:
    review_id: str
    product_slug: str
    product_name: str
    review_url: str
    review_title: str = ""
    review_date: str = ""
    review_rating: str = ""
    image_count: int = 0
    scrape_status: str = "ok"
    error: str = ""


def parse_review(html: str, url: str, product: Product) -> Review:
    soup = BeautifulSoup(html, "html.parser")
    title = find_title(soup, ["h1", ".review-title", ".review__title", "[itemprop='name']"])
    date = find_title(soup, ["time[datetime]", "time", ".review-date", ".review__date"])
    rating = ""
    for selector in ("[itemprop='ratingValue']", ".rating", ".review-rating", ".review__rating"):
        node = soup.select_one(selector)
        if node:
            match = re.search(r"(?:^|\s)([1-5](?:[.,]\d+)?)(?:\s|$)", clean_text(node.get_text(" ", strip=True)))
            if match:
                rating = match.group(1).replace(",", ".")
                break
    return Review(
        review_id=review_id(url),
        product_slug=product.product_slug,
        product_name=product.product_name,
        review_url=url,
        review_title=title,
        review_date=date,
        review_rating=rating,
        image_count=len(extract_review_images(soup)),
    )


def extension_for(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) if content_type else None
    return guessed if guessed in IMAGE_EXTENSIONS else ".bin"


def download_image(fetcher: Fetcher, image_url: str, destination: Path, referer: str) -> Dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        data, headers = fetcher.get(image_url, accept="image/avif,image/webp,image/apng,image/*,*/*;q=0.8", referer=referer)
        digest = hashlib.sha256(data).hexdigest()
        extension = extension_for(image_url, headers.get("Content-Type", ""))
        final_path = destination.with_name(destination.name + extension)
        if not final_path.exists():
            final_path.write_bytes(data)
        return {
            "local_path": str(final_path),
            "sha256": digest,
            "bytes": len(data),
            "content_type": headers.get("Content-Type", ""),
            "download_status": "ok",
            "error": "",
        }
    except Exception as error:  # keep one bad image from losing the rest of the dataset
        return {
            "local_path": "",
            "sha256": "",
            "bytes": 0,
            "content_type": "",
            "download_status": "error",
            "error": str(error),
        }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_robots(fetcher: Fetcher, ignore_robots: bool) -> None:
    if ignore_robots:
        return
    try:
        content, _ = fetcher.get(f"{BASE_URL}/robots.txt", accept="text/plain")
        parser = RobotFileParser()
        parser.set_url(f"{BASE_URL}/robots.txt")
        parser.parse(content.decode("utf-8", errors="replace").splitlines())
        for path in ("/food/alcohol/wines/", "/reviews/example/", "/reviews/example/gallery/", "/review_1.html"):
            if not parser.can_fetch(USER_AGENT, urljoin(BASE_URL, path)):
                raise RuntimeError(f"robots.txt disallows this path: {path}")
    except RuntimeError:
        raise
    except Exception as error:
        print(f"warning: robots.txt could not be checked ({error}); continuing", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("Датасет/otzovik_wines"))
    parser.add_argument("--start-url", default=CATEGORY_URL)
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests")
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-category-pages", type=int, default=0, help="0 = follow all discovered category pages")
    parser.add_argument("--max-products", type=int, default=0, help="0 = all discovered wines")
    parser.add_argument("--max-reviews", type=int, default=0, help="0 = all reviews in selected wines")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all review images")
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--ignore-robots", action="store_true", help="only use if you have independently verified permission")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not is_category_url(args.start_url):
        raise SystemExit("--start-url must be an Otzovik wine category URL")
    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(args.delay, args.timeout, args.retries, USER_AGENT)
    check_robots(fetcher, args.ignore_robots)
    scraped_at = now_iso()

    products_by_slug: Dict[str, Product] = {}
    category_queue = [normalize_url(args.start_url)]
    category_seen: Set[str] = set()
    while category_queue and (not args.max_category_pages or len(category_seen) < args.max_category_pages):
        page_url = category_queue.pop(0)
        if page_url in category_seen:
            continue
        category_seen.add(page_url)
        html = fetcher.html(page_url)
        for record in extract_product_links(html, page_url):
            products_by_slug.setdefault(record["product_slug"], Product(**record))
        for next_url in extract_category_links(html, page_url):
            if next_url not in category_seen and next_url not in category_queue:
                category_queue.append(next_url)
        print(f"category pages: {len(category_seen)}, wines: {len(products_by_slug)}", flush=True)

    products = list(products_by_slug.values())
    if args.max_products:
        products = products[: args.max_products]
    reviews: List[Review] = []
    image_rows: List[Dict[str, object]] = []
    image_counter = 0
    for product_index, product in enumerate(products, start=1):
        try:
            product_pages = [product.product_url]
            product_page_seen: Set[str] = set()
            review_urls: List[str] = []
            while product_pages:
                product_page_url = product_pages.pop(0)
                if product_page_url in product_page_seen:
                    continue
                product_page_seen.add(product_page_url)
                product_html = fetcher.html(product_page_url, referer=CATEGORY_URL)
                detail_name = find_title(BeautifulSoup(product_html, "html.parser"), ["h1", ".product-title", ".review-product-title"])
                if detail_name and "отзыв" not in detail_name.lower():
                    product.product_name = detail_name.removesuffix(" - отзывы").strip()
                review_urls.extend(extract_review_links(product_html, product_page_url))
                for next_url in extract_pagination_product_pages(product_html, product_page_url):
                    if next_url not in product_page_seen and next_url not in product_pages:
                        product_pages.append(next_url)
                if args.max_reviews and len(dict.fromkeys(review_urls)) >= args.max_reviews:
                    break
            review_urls = list(dict.fromkeys(review_urls))
            if args.max_reviews:
                review_urls = review_urls[: args.max_reviews]
            seen_image_urls: Set[str] = set()

            def add_image(image_url: str, review: Optional[Review], referer: str, image_index: int) -> None:
                nonlocal image_counter
                if image_url in seen_image_urls:
                    return
                if args.max_images and image_counter >= args.max_images:
                    return
                seen_image_urls.add(image_url)
                image_counter += 1
                review_key = review.review_id if review else "gallery"
                image_id = hashlib.sha1(f"{review_key}|{image_url}".encode()).hexdigest()[:16]
                folder = f"review_{review.review_id}" if review else "gallery"
                relative_base = Path("images") / product.product_slug / folder / image_id
                download_meta: Dict[str, object] = {
                    "local_path": "",
                    "sha256": "",
                    "bytes": 0,
                    "content_type": "",
                    "download_status": "skipped",
                    "error": "",
                }
                if not args.no_download_images:
                    download_meta = download_image(fetcher, image_url, output_dir / relative_base, referer)
                    if download_meta["local_path"]:
                        download_meta["local_path"] = str(Path(download_meta["local_path"]).relative_to(output_dir))
                image_rows.append({
                    "image_id": image_id,
                    "product_slug": product.product_slug,
                    "product_name": product.product_name,
                    "product_url": product.product_url,
                    "review_id": review.review_id if review else "",
                    "review_url": review.review_url if review else referer,
                    "review_title": review.review_title if review else "gallery",
                    "image_index": image_index,
                    "source_image_url": image_url,
                    "collected_at": scraped_at,
                    **download_meta,
                })

            for review_url in review_urls:
                review_html = fetcher.html(review_url, referer=product.product_url)
                review = parse_review(review_html, review_url, product)
                image_urls = extract_review_images(BeautifulSoup(review_html, "html.parser"))
                review.image_count = len(image_urls)
                reviews.append(review)
                for image_index, image_url in enumerate(image_urls):
                    add_image(image_url, review, review_url, image_index)
                    if args.max_images and image_counter >= args.max_images:
                        break
                if args.max_images and image_counter >= args.max_images:
                    break

            if not args.max_images or image_counter < args.max_images:
                gallery_pages = [gallery_url_for(product.product_url)]
                gallery_seen: Set[str] = set()
                gallery_index = 0
                try:
                    while gallery_pages:
                        gallery_page_url = gallery_pages.pop(0)
                        if gallery_page_url in gallery_seen:
                            continue
                        gallery_seen.add(gallery_page_url)
                        gallery_html = fetcher.html(gallery_page_url, referer=product.product_url)
                        for item in extract_gallery_images(gallery_html, gallery_page_url):
                            linked_review = next((item_review for item_review in reviews if item_review.review_url == item["review_url"]), None)
                            add_image(item["source_image_url"], linked_review, item["review_url"] or gallery_page_url, gallery_index)
                            gallery_index += 1
                            if args.max_images and image_counter >= args.max_images:
                                break
                        if args.max_images and image_counter >= args.max_images:
                            break
                        for next_url in extract_gallery_page_links(gallery_html, gallery_page_url):
                            if next_url not in gallery_seen and next_url not in gallery_pages:
                                gallery_pages.append(next_url)
                except Exception as gallery_error:
                    print(f"warning: gallery skipped for {product.product_slug}: {gallery_error}", file=sys.stderr)

            product.review_count_scraped = len(review_urls)
            product.image_count = sum(item["product_slug"] == product.product_slug for item in image_rows)
        except Exception as error:
            product.scrape_status = "error"
            product.error = str(error)
        print(f"products: {product_index}/{len(products)}, reviews: {len(reviews)}, images: {len(image_rows)}", flush=True)
        if args.max_images and image_counter >= args.max_images:
            break

    product_fields = list(asdict(Product("", "", "")).keys())
    review_fields = list(asdict(Review("", "", "", "")).keys())
    image_fields = [
        "image_id", "product_slug", "product_name", "product_url", "review_id", "review_url",
        "review_title", "image_index", "source_image_url", "local_path", "sha256", "bytes",
        "content_type", "download_status", "error", "collected_at",
    ]
    write_csv(output_dir / "products.csv", (asdict(item) for item in products), product_fields)
    write_csv(output_dir / "reviews.csv", (asdict(item) for item in reviews), review_fields)
    write_csv(output_dir / "images.csv", image_rows, image_fields)
    report = {
        "source": CATEGORY_URL,
        "start_url": args.start_url,
        "scraped_at": scraped_at,
        "category_pages_read": len(category_seen),
        "products_seen": len(products),
        "reviews_seen": len(reviews),
        "images_seen": len(image_rows),
        "images_downloaded": sum(row["download_status"] == "ok" for row in image_rows),
        "images_failed": sum(row["download_status"] == "error" for row in image_rows),
        "products_with_errors": sum(product.scrape_status != "ok" for product in products),
        "image_mapping": "images.csv maps every image to product_slug/product_name and review_id/review_url",
        "copyright_note": "Review photos are user-contributed content. Verify Otzovik terms, image licenses, and intended use before redistribution or model training.",
    }
    (output_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
