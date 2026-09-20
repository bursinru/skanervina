"""Ranking confidence without ground truth.

The TZ asks for an F1-style metric for top-1 and top-5 so the gap to neighbours
is visible in the API. True F1 needs labelled queries. Here F1 is the harmonic
mean of top-1 cosine similarity and margin to the k-th neighbour, scaled so a
0.15 gap counts as full separation.
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .catalog import CatalogWine, normalize, tokens


def _f1(precision: float, recall: float) -> float:
    precision = max(0.0, min(1.0, float(precision)))
    recall = max(0.0, min(1.0, float(recall)))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def same_label_family(left: Optional[CatalogWine], right: Optional[CatalogWine]) -> bool:
    """Same producer + grape (or shared name tokens): catalog SKUs of one wine line."""

    if left is None or right is None or left.slug == right.slug:
        return False
    if not left.winery or normalize(left.winery) != normalize(right.winery):
        return False
    grapes_left = {normalize(item) for item in left.grapes if normalize(item)}
    grapes_right = {normalize(item) for item in right.grapes if normalize(item)}
    if grapes_left and grapes_right and grapes_left & grapes_right:
        return True
    names_left, names_right = tokens(left.name), tokens(right.name)
    return len(names_left & names_right) >= 2


def distinct_margin(ranked: Sequence[Mapping[str, Any]], catalog) -> float:
    """Gap to the first neighbour that is not the same wine line."""

    if not ranked:
        return 0.0
    best = ranked[0]
    leader = catalog.get(best["slug"]) if catalog is not None else None
    for item in ranked[1:]:
        other = catalog.get(item.get("slug")) if catalog is not None else None
        if same_label_family(leader, other):
            continue
        return round(float(best["score"]) - float(item["score"]), 4)
    return round(float(best["score"]), 4)


def is_visual_match(
    score: float,
    margin: float,
    *,
    threshold: float,
    min_margin: float,
    clear_match: float,
    family_tie: bool = False,
    corroborated: bool = False,
    gap_floor: float = 0.70,
    gap_match: float = 0.06,
) -> bool:
    if score >= threshold and (margin >= min_margin or family_tie or corroborated or score >= clear_match):
        return True
    return score >= gap_floor and margin >= gap_match


def ranking_metrics(candidates: Sequence[Mapping[str, Any]], gap_scale: float = 0.15) -> Dict[str, Any]:
    top = [
        {"slug": str(item["slug"]), "score": round(float(item["score"]), 4)}
        for item in list(candidates)[:5]
        if item.get("slug") is not None
    ]
    scores = [item["score"] for item in top]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top5 = scores[4] if len(scores) > 4 else (scores[-1] if scores else 0.0)
    margin = round(top1 - top2, 4) if len(scores) > 1 else round(top1, 4)
    scale = gap_scale if gap_scale > 0 else 0.15
    return {
        "f1_top1": _f1(top1, margin / scale),
        "f1_top5": _f1(top1, (top1 - top5) / scale if scores else 0.0),
        "margin": margin,
        "top5": top,
        "note": "F1 without ground truth: harmonic mean of top-1 similarity and margin to the k-th neighbour (0.15 = full gap).",
    }


def best_slug(result: Mapping[str, Any]) -> Optional[str]:
    slug = result.get("slug")
    if isinstance(slug, str) and slug:
        return slug
    ranking = result.get("ranking") or {}
    top5: List[Mapping[str, Any]] = list(ranking.get("top5") or [])
    if top5 and top5[0].get("slug"):
        return str(top5[0]["slug"])
    candidates = (result.get("recognition") or {}).get("candidates") or []
    if candidates and candidates[0].get("slug"):
        return str(candidates[0]["slug"])
    return None
