"""Ranking confidence without ground truth.

The TZ asks for an F1-style metric for top-1 and top-5 so the gap to neighbours
is visible in the API. True F1 needs labelled queries. Here F1 is the harmonic
mean of top-1 cosine similarity and margin to the k-th neighbour, scaled so a
0.15 gap counts as full separation.
"""
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _f1(precision: float, recall: float) -> float:
    precision = max(0.0, min(1.0, float(precision)))
    recall = max(0.0, min(1.0, float(recall)))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


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
