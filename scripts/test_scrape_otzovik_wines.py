#!/usr/bin/env python3
"""Small offline parser checks for the Otzovik dataset collector."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scrape_otzovik_wines import (  # noqa: E402
    CATEGORY_URL,
    extract_category_links,
    extract_gallery_images,
    extract_product_links,
    extract_review_images,
    extract_review_links,
)
from bs4 import BeautifulSoup  # noqa: E402


def main() -> None:
    category = """
    <h3><a href='/reviews/alpha-wine/'>Вино Alpha</a></h3>
    <a href='/reviews/alpha-wine/'>1 отзыв</a>
    <a href='/food/alcohol/wines/2/'>2</a>
    """
    products = extract_product_links(category, CATEGORY_URL)
    assert products == [{
        "product_slug": "alpha-wine",
        "product_name": "Вино Alpha",
        "product_url": "https://otzovik.com/reviews/alpha-wine/",
        "catalog_rating": "",
        "catalog_review_count": "",
    }], products
    assert extract_category_links(category, CATEGORY_URL) == ["https://otzovik.com/food/alcohol/wines/2/"]

    review = """
    <main class='review-body'>
      <h1>Хорошее вино</h1>
      <div class='review-photos'>
        <a href='https://i.otzovik.com/objects/a/1/abc.webp'><img src='https://i.otzovik.com/objects/a/1/abc_s.webp'></a>
      </div>
      <img src='https://i2024.otzovik.com/objects/a/1/year.webp'>
      <img src='https://i2024.otzovik.com/2024/07/avatar/1.webp'>
      <img src='/assets/logo.svg'>
    </main>
    <a href='/review_123.html'>Читать весь отзыв</a>
    """
    soup = BeautifulSoup(review, "html.parser")
    assert extract_review_links(review, CATEGORY_URL) == ["https://otzovik.com/review_123.html"]
    assert extract_review_images(soup) == [
        "https://i.otzovik.com/objects/a/1/abc.webp",
        "https://i2024.otzovik.com/objects/a/1/year.webp",
    ]

    gallery = """
    <a href='/review_123.html'><img src='https://i.otzovik.com/objects/a/1/abc_s.webp'></a>
    <a href='/reviews/alpha-wine/gallery/2/'>2</a>
    """
    assert extract_gallery_images(gallery, "https://otzovik.com/reviews/alpha-wine/gallery/") == [{
        "source_image_url": "https://i.otzovik.com/objects/a/1/abc.webp",
        "review_url": "https://otzovik.com/review_123.html",
    }]
    print("offline Otzovik parser checks: OK")


if __name__ == "__main__":
    main()
