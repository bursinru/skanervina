"""Russian alternatives from other wineries and a rule-based sommelier."""
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .catalog import CatalogWine, WineCatalog, normalize

_HASHED_PHOTO = re.compile(r"_[0-9a-f]{8,}\.(webp|jpe?g|png)$", re.I)


def has_catalog_photo(wine: CatalogWine) -> bool:
    name = urlparse(wine.image_url or "").path.rsplit("/", 1)[-1]
    return bool(_HASHED_PHOTO.search(name))


OCCASIONS = {
    "meat": {
        "label": "Мясо",
        "dishes": {"мясо", "стейк", "гриль", "баранина", "говядина", "шашлык"},
        "colors": {"red"},
        "reason": "Подходит к мясу",
    },
    "fish": {
        "label": "Рыба",
        "dishes": {"рыба", "морепродукты", "устриц", "креветк"},
        "colors": {"white", "sparkling", "rose"},
        "reason": "К рыбе",
    },
    "cheese": {
        "label": "Сыр",
        "dishes": {"сыр", "сыры"},
        "colors": {"white", "red", "sparkling"},
        "reason": "К сыру",
    },
    "dessert": {
        "label": "Десерт",
        "dishes": {"десерт", "выпечк", "фрукты", "шоколад"},
        "colors": {"white", "red", "rose"},
        "reason": "К десерту",
    },
    "aperitif": {
        "label": "Без еды",
        "dishes": {"закуск", "аперитив"},
        "colors": {"sparkling", "white", "rose"},
        "reason": "Без еды",
    },
}

VALID_COLORS = {"any", "white", "red", "rose", "sparkling"}
VALID_SWEETNESS = {"any", "dry", "semi_dry", "semi_sweet", "sweet"}
_ALCOHOL_RE = re.compile(r"(\d+[.,]?\d*)")
_LIGHT_RED = ("пино", "гаме", "санджовез", "барбера", "гренаш", "гарнач", "цвейгельт")
_HEAVY_RED = ("каберне", "сира", "шираз", "мальбек", "саперави")


def _winery_key(value: str) -> str:
    return normalize(value)


def alternative_score(source: CatalogWine, other: CatalogWine) -> float:
    if other.slug == source.slug:
        return -1
    if _winery_key(other.winery) and _winery_key(other.winery) == _winery_key(source.winery):
        return -1
    score = 0.0
    source_grapes = {normalize(item) for item in source.grapes if normalize(item)}
    other_grapes = {normalize(item) for item in other.grapes if normalize(item)}
    score += 3 * len(source_grapes & other_grapes)
    if source.color and other.color and normalize(source.color) == normalize(other.color):
        score += 2
    if source.category and other.category and normalize(source.category) == normalize(other.category):
        score += 1.5
    if source.region and other.region and normalize(source.region) != normalize(other.region):
        score += 0.4
    if other.public_rating:
        score += min(other.public_rating, 5) / 10
    return score


def alternatives(catalog: WineCatalog, wine: CatalogWine, limit: int = 3) -> List[Dict[str, object]]:
    scored = [
        (alternative_score(wine, other), other)
        for other in catalog
        if alternative_score(wine, other) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    pool = [item for _, item in scored[:24]]
    with_photo = [item for item in pool if has_catalog_photo(item)]
    without_photo = [item for item in pool if not has_catalog_photo(item)]
    return [item.to_card() for item in (with_photo + without_photo)[:limit]]


def _wine_blob(wine: CatalogWine) -> str:
    return normalize(" ".join([wine.color, wine.category, wine.name, " ".join(wine.grapes or [])]))


def wine_color_key(wine: CatalogWine) -> str:
    blob = _wine_blob(wine)
    if "розов" in blob:
        return "rose"
    if "оранж" in blob:
        return "orange"
    if "красн" in blob:
        return "red"
    if "бел" in blob:
        return "white"
    if "игрист" in blob or "brut" in blob or "брют" in blob or "шампан" in blob:
        return "sparkling"
    return "unknown"


def wine_is_sparkling(wine: CatalogWine) -> bool:
    blob = _wine_blob(wine)
    return "игрист" in blob or "brut" in blob or "брют" in blob or "шампан" in blob or "spumante" in blob


def wine_sweetness_key(wine: CatalogWine) -> str:
    blob = _wine_blob(wine)
    if "полуслад" in blob:
        return "semi_sweet"
    if "полусух" in blob:
        return "semi_dry"
    if "сладк" in blob:
        return "sweet"
    if "сух" in blob or "брют" in blob or "brut" in blob:
        return "dry"
    return "unknown"


def _alcohol_pct(wine: CatalogWine) -> Optional[float]:
    match = _ALCOHOL_RE.search(wine.alcohol or "")
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _grape_blob(wine: CatalogWine) -> str:
    return normalize(" ".join(wine.grapes or []) + " " + wine.name)


def sommelier_hint(occasion: str, color: str = "any", sweetness: str = "any") -> str:
    spec = OCCASIONS.get(occasion) or {}
    color = color if color in VALID_COLORS else "any"
    sweetness = sweetness if sweetness in VALID_SWEETNESS else "any"
    if occasion == "meat" and color in {"any", "red"} and sweetness in {"any", "dry"}:
        return "К мясу подойдут насыщенные сухие красные вина с выразительным вкусом."
    if occasion == "meat" and color == "white":
        return "К мясу белое тоже бывает уместно — лучше плотные выдержанные белые, чем самые лёгкие."
    if occasion == "fish" and color == "red":
        return "Красное к рыбе не запрет: лучше лёгкие красные без тяжёлого дуба."
    if occasion == "fish" and color in {"any", "white", "sparkling"}:
        return "К рыбе подойдут свежие белые и игристые без тяжёлого дуба."
    if occasion == "dessert" and sweetness in {"any", "sweet", "semi_sweet"}:
        return "К десерту выше поднимаем сладкие и полусладкие, чтобы вино не казалось кислым."
    if occasion == "dessert" and sweetness == "dry":
        return "Сухое к десерту может казаться резче — оставляем его в подборке, но слаще обычно лучше."
    if occasion == "cheese":
        return "К сыру подходят и свежие белые, и мягкие красные — смотрим и блюда, и стиль."
    if occasion == "aperitif":
        return "Без еды хорошо заходят игристое, лёгкое белое или розовое."
    return f"Подбираем вина к сочетанию «{spec.get('label', occasion)}»."


def _recommend_reason(spec: Dict[str, object], wine: CatalogWine, _color: str) -> str:
    grape = _grape_blob(wine)
    if spec.get("label") == "Рыба" and wine_color_key(wine) == "red":
        if any(item in grape for item in _LIGHT_RED):
            return "Лёгкое красное к рыбе"
        return "Красное к рыбе"
    if spec.get("label") == "Десерт" and wine_sweetness_key(wine) in {"sweet", "semi_sweet"}:
        return "Сладость к десерту"
    return str(spec.get("reason") or "Рекомендация сомелье")


def _score_color(wine: CatalogWine, wanted: str, occasion_colors: set) -> float:
    actual = wine_color_key(wine)
    sparkling = wine_is_sparkling(wine)
    if wanted == "sparkling":
        return 4.0 if sparkling else 0.7
    if wanted == "white":
        if actual == "white":
            return 4.0 + (0.4 if sparkling else 0)
        if actual == "orange":
            return 1.6
        return 0.7
    if wanted in {"red", "rose"}:
        return 4.0 if actual == wanted else 0.7
    if actual in occasion_colors or (sparkling and "sparkling" in occasion_colors):
        return 1.8
    if actual == "orange" and "white" in occasion_colors:
        return 0.8
    return 0.4


def _score_sweetness(wine: CatalogWine, wanted: str, occasion: str) -> float:
    actual = wine_sweetness_key(wine)
    if wanted != "any":
        return 4.0 if actual == wanted else 0.8
    if occasion == "dessert":
        return {"sweet": 3.4, "semi_sweet": 3.0, "semi_dry": 1.1, "dry": 0.35, "unknown": 0.6}.get(actual, 0.6)
    if occasion == "meat" and actual == "dry":
        return 1.1
    if occasion == "aperitif" and actual in {"dry", "semi_dry", "unknown"}:
        return 0.6
    return 0.3


def sommelier_reply(
    catalog: WineCatalog,
    occasion: str,
    current: Optional[CatalogWine] = None,
    *,
    color: str = "any",
    sweetness: str = "any",
    grape: str = "",
    region: str = "",
    alcohol_min: Optional[float] = None,
    alcohol_max: Optional[float] = None,
    limit: int = 8,
) -> Dict[str, object]:
    spec = OCCASIONS.get(occasion)
    if not spec:
        return {"error": "unknown_occasion", "occasions": {key: value["label"] for key, value in OCCASIONS.items()}}
    color = color if color in VALID_COLORS else "any"
    sweetness = sweetness if sweetness in VALID_SWEETNESS else "any"
    grape_q = normalize(grape)
    region_q = normalize(region)
    dish_needles = spec["dishes"]
    occasion_colors = set(spec["colors"])
    scored = []
    for wine in catalog:
        if current and wine.slug == current.slug:
            continue
        score = 0.0
        dish_blob = normalize(" ".join(wine.dishes or []))
        if any(needle in dish_blob for needle in dish_needles):
            score += 3.6
        score += _score_color(wine, color, occasion_colors)
        score += _score_sweetness(wine, sweetness, occasion)
        grape_blob = _grape_blob(wine)
        if occasion == "fish" and wine_color_key(wine) == "red":
            if any(item in grape_blob for item in _LIGHT_RED):
                score += 2.4
            elif any(item in grape_blob for item in _HEAVY_RED):
                score += 0.35
            else:
                score += 0.9
        if occasion == "meat" and wine_color_key(wine) == "red" and any(item in grape_blob for item in _HEAVY_RED):
            score += 0.8
        if grape_q:
            score += 2.2 if grape_q in grape_blob else 0.2
        if region_q:
            score += 2.0 if region_q in normalize(wine.region) else 0.2
        abv = _alcohol_pct(wine)
        if alcohol_min is not None or alcohol_max is not None:
            if abv is None:
                score += 0.15
            elif (alcohol_min is None or abv >= alcohol_min) and (alcohol_max is None or abv <= alcohol_max):
                score += 1.6
            else:
                score += 0.15
        if wine.public_rating:
            score += min(wine.public_rating, 5) / 8
        scored.append((score, wine))
    scored.sort(key=lambda item: (item[0], 1 if has_catalog_photo(item[1]) else 0), reverse=True)
    wines = []
    for score, wine in scored[: max(1, limit)]:
        card = wine.to_card()
        card["recommend_reason"] = _recommend_reason(spec, wine, color)
        card["sommelier_score"] = round(score, 3)
        wines.append(card)
    top = scored[0][0] if scored else 0
    close = sum(1 for score, _ in scored if score >= top * 0.82)
    return {
        "occasion": occasion,
        "label": spec["label"],
        "color": color,
        "sweetness": sweetness,
        "hint": sommelier_hint(occasion, color, sweetness),
        "found": max(len(wines), min(close, 48)),
        "wines": wines,
    }
