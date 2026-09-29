"""Visual + label-text fusion: a softmax over candidate wines.

Candidates are the visual top-K and the text top-N. Each gets a few features;
weights come from a conditional logit fitted on labelled photos
(backend/training/fit_fusion.py) and live in fusion.json next to this module.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .label_text import LabelText

FEATURES = (
    "visual_gap",      # visual score minus the best visual score among candidates
    "visual_far",      # the part of the gap beyond 0.08: a clear label read can overrule it
    "text_share",      # text score / best text score among candidates
    "text_log",        # log(1 + IDF-weighted text score)
    "coverage",        # share of the wine's catalog identity found on the label
    "sweetness",       # +1 agree, -1 disagree, 0 unknown
    "colour",
    "year",
    "text_leader",     # best text score among candidates
    "contradiction",   # label weight a same-winery rival explains and this wine does not, / best text
    "text_margin",     # text leader only: its lead over the second text score, / best text
    "photo_text_share",  # words shared with the text read off the catalog photo, / best
    "photo_text_log",
)
WEIGHTS_PATH = Path(__file__).with_name("fusion.json")
# Words PP-OCR read off each catalog photo: {slug: [[text, weight], ...]}.
PHOTO_TEXT_PATH = Path(__file__).with_name("catalog_photo_text.json")


def load_weights(path: Path = WEIGHTS_PATH) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if tuple(payload.get("features") or ()) != FEATURES:
        return None
    return payload


def candidates(visual: Mapping[str, float], texts: Sequence[Optional[LabelText]], visual_k: int, text_n: int) -> List[str]:
    """Visual top-K plus the text top-N of every text channel."""

    found = sorted(visual, key=visual.get, reverse=True)[:visual_k]
    for text in texts:
        if text is not None:
            found += [slug for slug in text.top(text_n) if slug in visual]
    return list(dict.fromkeys(found))


def features(slugs: Sequence[str], visual: Mapping[str, float], text: Optional[LabelText], photo: Optional[LabelText] = None) -> Dict[str, List[float]]:
    best_visual = max(visual[slug] for slug in slugs)
    evidence = {slug: text.evidence(slug, slugs) for slug in slugs} if text is not None else {}
    ordered = sorted((item.score for item in evidence.values()), reverse=True)
    best_text = ordered[0] if ordered else 0.0
    second_text = ordered[1] if len(ordered) > 1 else 0.0
    photo_scores = {slug: photo.totals.get(slug, 0.0) for slug in slugs} if photo is not None else {}
    best_photo = max(photo_scores.values(), default=0.0)
    rows = {}
    for slug in slugs:
        item = evidence.get(slug)
        score = item.score if item else 0.0
        gap = visual[slug] - best_visual
        leader = best_text > 0 and score >= best_text
        rows[slug] = [
            gap,
            min(0.0, gap + 0.08),
            score / best_text if best_text > 0 else 0.0,
            math.log1p(score),
            item.coverage if item else 0.0,
            float(item.sweetness) if item else 0.0,
            float(item.colour) if item else 0.0,
            float(item.year) if item else 0.0,
            1.0 if leader else 0.0,
            item.contradiction / best_text if item and best_text > 0 else 0.0,
            (best_text - second_text) / best_text if leader else 0.0,
            photo_scores.get(slug, 0.0) / best_photo if best_photo > 0 else 0.0,
            math.log1p(photo_scores.get(slug, 0.0)),
        ]
    return rows


SIBLING_GAP = 0.05
SIBLING_WORDS = 2.0


def prefer_named_sibling(ranked: List[dict], visual: Mapping[str, float], text: Optional[LabelText]) -> List[dict]:
    """A close sibling of the same winery wins if only it explains label words.

    The ranker is linear and weighs all text alike; a grape or cuvée name that
    one bottle of a series carries and its neighbour lacks decides the pair.
    Cross-validated on 205 labelled photos: +5 top-1 over the ranker alone.
    """

    if text is None or len(ranked) < 2:
        return ranked
    lead = ranked[0]["slug"]
    winery = text.index.wineries.get(lead)
    if not winery:
        return ranked
    for position in (1, 2):
        if position >= len(ranked):
            break
        other = ranked[position]["slug"]
        if text.index.wineries.get(other) != winery or visual[lead] - visual[other] > SIBLING_GAP:
            continue
        own, rival = text.contributions.get(other, {}), text.contributions.get(lead, {})
        other_only = sum(value for word, value in own.items() if word not in rival)
        lead_only = sum(value for word, value in rival.items() if word not in own)
        if other_only >= SIBLING_WORDS and lead_only < SIBLING_WORDS / 3:
            moved = dict(ranked[position], sibling_override=True)
            return [moved] + [item for item in ranked if item["slug"] != other]
    return ranked


def rank(visual: Mapping[str, float], text: Optional[LabelText], weights: dict, photo: Optional[LabelText] = None) -> List[dict]:
    """Candidates by fused probability, best first."""

    slugs = candidates(visual, (text, photo), weights["visual_k"], weights["text_n"])
    if not slugs:
        return []
    rows = features(slugs, visual, text, photo)
    w = weights["weights"]
    logits = {slug: sum(a * b for a, b in zip(w, values)) for slug, values in rows.items()}
    top = max(logits.values())
    exp = {slug: math.exp(value - top) for slug, value in logits.items()}
    total = sum(exp.values())
    ranked = []
    for slug in sorted(slugs, key=lambda item: logits[item], reverse=True):
        item = text.evidence(slug) if text is not None else None
        ranked.append({
            "slug": slug,
            "probability": exp[slug] / total,
            "visual": visual[slug],
            "text_score": round(item.score, 3) if item else 0.0,
            "text_coverage": round(item.coverage, 3) if item else 0.0,
            "text_matched": list(item.matched) if item else [],
        })
    return prefer_named_sibling(ranked, visual, text)
