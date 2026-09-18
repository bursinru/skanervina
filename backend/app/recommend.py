"""Russian alternatives from other wineries and a rule-based sommelier."""
from typing import Dict, List, Optional

from .catalog import CatalogWine, WineCatalog, normalize, tokens


OCCASIONS = {
    "meat": {
        "label": "Мясо",
        "dishes": {"мясо", "стейк", "гриль", "баранина", "говядина", "шашлык"},
        "colors": {"красное", "красное сухое"},
        "hint": "К мясу обычно лучше красное с телом. Ниже — российские вина с похожим профилем.",
    },
    "fish": {
        "label": "Рыба и морепродукты",
        "dishes": {"рыба", "морепродукты", "устриц", "креветк"},
        "colors": {"белое", "белое сухое", "игристое"},
        "hint": "К рыбе — белое или игристое без тяжёлого дуба.",
    },
    "cheese": {
        "label": "Сыр",
        "dishes": {"сыр", "сыры"},
        "colors": {"белое", "красное", "игристое"},
        "hint": "К сыру подходят и свежие белые, и мягкие красные. Смотрите совпадение блюд в каталоге.",
    },
    "dessert": {
        "label": "Десерт",
        "dishes": {"десерт", "выпечк", "фрукты", "шоколад"},
        "colors": {"сладкое", "белое сладкое", "красное сладкое"},
        "hint": "К десерту нужен остаточный сахар, иначе вино покажется кислым.",
    },
    "aperitif": {
        "label": "Просто выпить",
        "dishes": {"закуск", "аперитив"},
        "colors": {"игристое", "белое", "розовое"},
        "hint": "Для аперитива — игристое или лёгкое белое из российского каталога.",
    },
}


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
    ranked = sorted(
        catalog,
        key=lambda other: alternative_score(wine, other),
        reverse=True,
    )
    picked = [item.to_card() for item in ranked if alternative_score(wine, item) > 0][:limit]
    return picked


def sommelier_reply(catalog: WineCatalog, occasion: str, current: Optional[CatalogWine] = None) -> Dict[str, object]:
    spec = OCCASIONS.get(occasion)
    if not spec:
        return {"error": "unknown_occasion", "occasions": {key: value["label"] for key, value in OCCASIONS.items()}}
    dish_needles = spec["dishes"]
    color_needles = {normalize(item) for item in spec["colors"]}
    scored = []
    for wine in catalog:
        if current and wine.slug == current.slug:
            continue
        score = 0.0
        blob = normalize(" ".join(wine.dishes or []) + " " + wine.category + " " + wine.color)
        if any(needle in blob for needle in dish_needles):
            score += 3
        if color_needles & tokens(wine.color + " " + wine.category):
            score += 2
        if wine.public_rating:
            score += min(wine.public_rating, 5) / 8
        if score > 0:
            scored.append((score, wine))
    scored.sort(key=lambda item: item[0], reverse=True)
    wines = [wine.to_card() for _, wine in scored[:3]]
    return {
        "occasion": occasion,
        "label": spec["label"],
        "hint": spec["hint"],
        "wines": wines,
    }
