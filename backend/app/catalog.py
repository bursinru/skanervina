import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import quote


TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def normalize(value: object) -> str:
    """Normalize Russian/Latin catalog text for OCR-friendly matching."""
    text = str(value or "").lower().replace("ё", "е")
    return " ".join(TOKEN_RE.findall(text))


def tokens(value: object) -> Set[str]:
    return set(normalize(value).split())


def split_grapes(value: object) -> List[str]:
    return [item.strip() for item in re.split(r"[,;]", str(value or "")) if item.strip()]


def optional_float(value: object) -> Optional[float]:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_dishes(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return split_grapes(text)
    return [str(item).strip() for item in parsed if str(item).strip()]


@dataclass(frozen=True)
class CatalogWine:
    slug: str
    name: str
    winery: str
    category: str
    region: str
    grapes: List[str]
    description: str
    image_name: str
    image_base_url: str
    direct_image_url: Optional[str] = None
    public_rating: Optional[float] = None
    quality_rating: Optional[float] = None
    color: str = ""
    temperature: str = ""
    alcohol: str = ""
    dishes: Optional[List[str]] = None
    source_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dishes is None:
            object.__setattr__(self, "dishes", [])

    @property
    def search_text(self) -> str:
        return normalize(
            " ".join(
                [self.name, self.winery, self.category, self.region, " ".join(self.grapes)]
            )
        )

    @property
    def search_tokens(self) -> Set[str]:
        return tokens(self.search_text)

    @property
    def image_url(self) -> Optional[str]:
        if self.direct_image_url:
            return self.direct_image_url
        if not self.image_name:
            return None
        return self.image_base_url + quote(self.image_name, safe="")

    def to_card(self) -> Dict[str, object]:
        return {
            "slug": self.slug,
            "name": self.name,
            "winery": self.winery,
            "category": self.category,
            "region": self.region,
            "grapes": self.grapes,
            "image_url": self.image_url,
            "description": self.description,
            "public_rating": self.public_rating,
            "quality_rating": self.quality_rating,
            "color": self.color,
            "temperature": self.temperature,
            "alcohol": self.alcohol,
            "dishes": self.dishes,
            "source_url": self.source_url,
            "source": "Каталог «Своё Вино»",
        }


@dataclass(frozen=True)
class SearchMatch:
    wine: CatalogWine
    score: float
    matched_tokens: List[str]


class WineCatalog:
    def __init__(self, wines: Iterable[CatalogWine]):
        self._wines: Dict[str, CatalogWine] = {wine.slug: wine for wine in wines}
        self._index: Dict[str, Set[str]] = {}
        for wine in self._wines.values():
            for token in wine.search_tokens:
                self._index.setdefault(token, set()).add(wine.slug)

    @classmethod
    def from_csv(cls, path: Path, image_base_url: str) -> "WineCatalog":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return cls.from_rows(csv.DictReader(handle), image_base_url)

    @classmethod
    def from_rows(
        cls, rows: Iterable[Dict[str, object]], image_base_url: str
    ) -> "WineCatalog":
        wines: List[CatalogWine] = []
        seen_slugs: Set[str] = set()
        for row in rows:
            slug = str(row.get("Slug") or "").strip()
            name = str(row.get("Название вина") or "").strip()
            if not slug or not name or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            site_image_url = str(row.get("svoe_vino_image_url") or "").strip()
            wines.append(
                CatalogWine(
                    slug=slug,
                    name=name,
                    winery=str(row.get("Винодельня") or "").strip(),
                    category=str(
                        row.get("svoe_vino_category") or row.get("Категория") or ""
                    ).strip(),
                    region=str(
                        row.get("svoe_vino_region") or row.get("Регион") or ""
                    ).strip(),
                    grapes=split_grapes(
                        row.get("svoe_vino_grapes") or row.get("Сорт винограда")
                    ),
                    description=str(
                        row.get("svoe_vino_description") or row.get("Описание") or ""
                    ).strip(),
                    image_name=str(row.get("Название фото") or "").strip(),
                    image_base_url=image_base_url,
                    direct_image_url=site_image_url or None,
                    public_rating=optional_float(row.get("svoe_vino_public_rating")),
                    quality_rating=optional_float(row.get("svoe_vino_quality_rating")),
                    color=str(row.get("svoe_vino_color") or "").strip(),
                    temperature=str(row.get("svoe_vino_temperature") or "").strip(),
                    alcohol=str(row.get("svoe_vino_alcohol") or "").strip(),
                    dishes=parse_dishes(row.get("svoe_vino_dishes_json")),
                    source_url=str(row.get("svoe_vino_source_url") or "").strip() or None,
                )
            )
        return cls(wines)

    @property
    def size(self) -> int:
        return len(self._wines)

    def get(self, slug: str) -> Optional[CatalogWine]:
        return self._wines.get(slug)

    def search(self, query: str, limit: int = 5) -> List[SearchMatch]:
        query_normalized = normalize(query)
        query_tokens = set(query_normalized.split())
        if not query_tokens:
            return []

        indexed_slugs: Set[str] = set()
        for token in query_tokens:
            indexed_slugs.update(self._index.get(token, set()))
        candidates = (
            [self._wines[slug] for slug in indexed_slugs]
            if indexed_slugs
            else list(self._wines.values())
        )

        matches: List[SearchMatch] = []
        for wine in candidates:
            matched = sorted(query_tokens.intersection(wine.search_tokens))
            overlap = len(matched) / len(query_tokens)
            sequence = difflib.SequenceMatcher(
                None, query_normalized, wine.search_text
            ).ratio()
            score = min(1.0, overlap * 0.75 + sequence * 0.25)
            matches.append(SearchMatch(wine, score, matched))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]
