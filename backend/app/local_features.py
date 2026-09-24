"""Geometric verification of visual candidates with local features.

Global embeddings find wines that look alike; they struggle to tell apart two
labels of one producer's line. Counting RANSAC-consistent RootSIFT matches
between the query label and each candidate's catalog photo checks that the
same printed artwork is present, which separates such look-alikes.
"""
from collections import OrderedDict
from threading import Lock
from typing import Callable, Iterable, List, Mapping, Optional

import numpy as np
from PIL import Image

MAX_SIDE = 900
MAX_FEATURES = 1500
RATIO = 0.8
RANSAC_PX = 8.0


def _gray(image: Image.Image, max_side: int = MAX_SIDE) -> np.ndarray:
    image = image.convert("L")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    return np.asarray(image)


class LocalMatcher:
    def __init__(self, cache_size: int = 512):
        import cv2

        self.cv2 = cv2
        self.sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.cache: "OrderedDict[str, list]" = OrderedDict()
        self.cache_size = cache_size
        self.lock = Lock()

    def features(self, image: Image.Image):
        gray = _gray(image)
        # CLAHE keeps keypoints on dim shelf photos and glossy highlights comparable.
        gray = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 8:
            return None
        # RootSIFT: Hellinger kernel is more robust than L2 on raw SIFT.
        descriptors = descriptors / (np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors).astype(np.float32)
        points = np.float32([keypoint.pt for keypoint in keypoints])
        return points, descriptors

    def gallery(self, slug: str, loader: Callable[[str], Iterable[Image.Image]]):
        with self.lock:
            if slug in self.cache:
                self.cache.move_to_end(slug)
                return self.cache[slug]
        entries = []
        for image in loader(slug) or []:
            try:
                feats = self.features(image)
            except Exception:
                feats = None
            if feats is not None:
                entries.append(feats)
        with self.lock:
            self.cache[slug] = entries
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return entries

    def inliers(self, query, reference) -> int:
        if query is None or reference is None:
            return 0
        q_points, q_desc = query
        r_points, r_desc = reference
        if len(q_desc) < 8 or len(r_desc) < 8:
            return 0
        pairs = self.matcher.knnMatch(q_desc, r_desc, k=2)
        good = [m for m, n in (pair for pair in pairs if len(pair) == 2) if m.distance < RATIO * n.distance]
        if len(good) < 8:
            return 0
        source = q_points[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        target = r_points[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        _, mask = self.cv2.findHomography(source, target, self.cv2.RANSAC, RANSAC_PX)
        return int(mask.sum()) if mask is not None else 0

    def verify(self, query_images: List[Image.Image], slugs: List[str], loader) -> Mapping[str, int]:
        queries = [self.features(image) for image in query_images]
        result = {}
        for slug in slugs:
            best = 0
            for reference in self.gallery(slug, loader):
                for query in queries:
                    best = max(best, self.inliers(query, reference))
            result[slug] = best
        return result


def rerank(ranked: List[dict], inliers: Mapping[str, int], *, min_inliers: int = 12, dominance: float = 1.5,
           weight: float = 0.15, saturation: int = 60) -> List[dict]:
    """Add a bounded geometric bonus; a clearly dominant verified candidate moves to the top."""

    if not inliers:
        return ranked
    rows = []
    for item in ranked:
        count = int(inliers.get(item["slug"], 0))
        bonus = weight * min(1.0, count / saturation) if count >= min_inliers else 0.0
        rows.append({**item, "inliers": count, "geo_delta": round(bonus, 4), "score": min(1.0, float(item["score"]) + bonus)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    counts = sorted((row["inliers"] for row in rows), reverse=True)
    leader = max(rows, key=lambda row: row["inliers"])
    runner_up = counts[1] if len(counts) > 1 else 0
    if leader["inliers"] >= min_inliers and leader["inliers"] >= dominance * max(runner_up, 1) and rows[0] is not leader:
        rows.remove(leader)
        rows.insert(0, leader)
    return rows


def geometric_confirmed(ranked: List[dict], *, min_inliers: int = 20, dominance: float = 2.0) -> bool:
    if not ranked:
        return False
    top = int(ranked[0].get("inliers") or 0)
    others = [int(row.get("inliers") or 0) for row in ranked[1:]]
    return top >= min_inliers and top >= dominance * max(max(others, default=0), 1)
