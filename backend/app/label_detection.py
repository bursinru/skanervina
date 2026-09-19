"""Find the front paper label on a bottle photo.

The crop must cover the whole panel: brand line, vintage, illustration and
layout.  A tight box around a drawing or a centre-biased slice of glass
throws SigLIP at the wrong wine.
"""

from dataclasses import dataclass
from math import atan2, hypot
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


NormalizedBox = Tuple[float, float, float, float]
Quad = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class LabelDetection:
    bbox: NormalizedBox
    confidence: float
    method: str = "label_panel"
    quad: Optional[Quad] = None


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _mean(integral: np.ndarray, left: int, top: int, right: int, bottom: int) -> float:
    area = max(1, (right - left) * (bottom - top))
    return float(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    ) / area


def _paper_mask(pixels: np.ndarray) -> np.ndarray:
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    value = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.where(value > 1e-4, 1.0 - minimum / np.maximum(value, 1e-4), 0.0)
    # Cream, white and pale gold stock.  Dark glass and black UI drop out.
    light_paper = (value > 0.40) & (value < 0.985) & (saturation < 0.42)
    warm = (red + green) > (blue * 1.02)
    return light_paper & warm


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 127


def _clear_border(mask: np.ndarray) -> np.ndarray:
    """Drop cream walls that touch the photo frame; keep the interior panel."""

    height, width = mask.shape
    keep = mask.copy()
    stack = []
    for x in range(width):
        if keep[0, x]:
            stack.append((x, 0))
        if keep[height - 1, x]:
            stack.append((x, height - 1))
    for y in range(height):
        if keep[y, 0]:
            stack.append((0, y))
        if keep[y, width - 1]:
            stack.append((width - 1, y))
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= width or y >= height or not keep[y, x]:
            continue
        keep[y, x] = False
        stack.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return keep


def _frame_like(box: Sequence[int], width: int, height: int) -> bool:
    left, top, right, bottom = box
    area = (right - left) * (bottom - top) / max(1, width * height)
    touches = (
        int(left <= 2)
        + int(top <= 2)
        + int(right >= width - 2)
        + int(bottom >= height - 2)
    )
    return area > 0.48 or touches >= 3 or (right - left) / width > 0.82


def _label_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Return bounding boxes (left, top, right, bottom, area) on a small mask."""

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    next_id = 1
    for y in range(height):
        row = mask[y]
        prev = labels[y - 1] if y else None
        for x in range(width):
            if not row[x]:
                continue
            left = labels[y, x - 1] if x else 0
            up = prev[x] if prev is not None else 0
            if left and up:
                root_left, root_up = find(left), find(up)
                labels[y, x] = root_left
                if root_left != root_up:
                    parent[root_up] = root_left
            elif left:
                labels[y, x] = find(left)
            elif up:
                labels[y, x] = find(up)
            else:
                parent.append(next_id)
                labels[y, x] = next_id
                next_id += 1

    remap = {}
    boxes = {}
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if not label:
                continue
            root = find(label)
            mapped = remap.setdefault(root, len(remap) + 1)
            box = boxes.get(mapped)
            if box is None:
                boxes[mapped] = [x, y, x + 1, y + 1, 1]
            else:
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x + 1)
                box[3] = max(box[3], y + 1)
                box[4] += 1
    return [tuple(item) for item in boxes.values()]


def _candidate_score(
    edge: np.ndarray,
    gray_integral: np.ndarray,
    edge_integral: np.ndarray,
    paper_integral: np.ndarray,
    paper_total: float,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> float:
    width = right - left
    height = bottom - top
    margin_x = max(1, round(width * 0.035))
    margin_y = max(1, round(height * 0.035))

    border = (
        _mean(edge_integral, left, top, right, min(bottom, top + margin_y))
        + _mean(edge_integral, left, max(top, bottom - margin_y), right, bottom)
        + _mean(edge_integral, left, top, min(right, left + margin_x), bottom)
        + _mean(edge_integral, max(left, right - margin_x), top, right, bottom)
    ) / 4

    inner_left = min(right - 1, left + margin_x * 3)
    inner_top = min(bottom - 1, top + margin_y * 3)
    inner_right = max(inner_left + 1, right - margin_x * 3)
    inner_bottom = max(inner_top + 1, bottom - margin_y * 3)
    interior_edge = _mean(edge_integral, inner_left, inner_top, inner_right, inner_bottom)

    outer_left = max(0, left - margin_x * 2)
    outer_top = max(0, top - margin_y * 2)
    outer_right = min(edge.shape[1], right + margin_x * 2)
    outer_bottom = min(edge.shape[0], bottom + margin_y * 2)
    outer_area = max(1, (outer_right - outer_left) * (outer_bottom - outer_top))
    candidate_area = max(1, width * height)
    outside_area = max(1, outer_area - candidate_area)
    outer_mean = (
        _mean(gray_integral, outer_left, outer_top, outer_right, outer_bottom) * outer_area
        - _mean(gray_integral, left, top, right, bottom) * candidate_area
    ) / outside_area
    inside_mean = _mean(gray_integral, inner_left, inner_top, inner_right, inner_bottom)
    contrast = min(1.0, abs(inside_mean - outer_mean) * 2.2)

    width_frac = width / edge.shape[1]
    height_frac = height / edge.shape[0]
    area = width_frac * height_frac
    aspect = height_frac / max(width_frac, 1e-6)
    if 1.15 <= aspect <= 2.7:
        aspect_score = 1.0
    elif 0.9 <= aspect < 1.15:
        aspect_score = 0.78
    elif aspect < 0.62:
        aspect_score = 0.08
    else:
        aspect_score = 0.42
    if 0.05 <= area <= 0.36:
        area_score = 1.0
    elif area <= 0.5:
        area_score = 0.5
    elif area >= 0.68:
        area_score = 0.05
    else:
        area_score = 0.28

    paper_inside = _mean(paper_integral, left, top, right, bottom) * candidate_area
    coverage = paper_inside / max(paper_total, 1.0)
    density = paper_inside / candidate_area
    # Text and drawing add interior edges, but must not beat a full paper panel.
    text_layout = min(1.0, interior_edge * 2.4) * min(1.0, density + 0.15)
    paper = min(1.0, density * 1.15)
    leftover = max(0.0, 1.0 - coverage) if paper_total > candidate_area * 0.15 else 0.0

    score = (
        min(1.0, border * 3.1) * 0.18
        + contrast * 0.12
        + text_layout * 0.10
        + aspect_score * 0.16
        + area_score * 0.10
        + paper * 0.16
        + min(1.0, coverage) * 0.22
        - leftover * 0.18
    )
    return float(score)


def _expand_to_paper(mask: np.ndarray, box: Sequence[int]) -> Tuple[int, int, int, int]:
    height, width = mask.shape
    left, top, right, bottom = [int(value) for value in box]

    def accept(next_box: Tuple[int, int, int, int]) -> bool:
        return not _frame_like(next_box, width, height)

    for _ in range(24):
        grown = False
        step_x = max(1, round(width * 0.012))
        step_y = max(1, round(height * 0.012))
        if left > 0:
            strip = mask[top:bottom, max(0, left - step_x) : left]
            nxt = (max(0, left - step_x), top, right, bottom)
            if strip.size and float(strip.mean()) >= 0.28 and accept(nxt):
                left = nxt[0]
                grown = True
        if right < width:
            strip = mask[top:bottom, right : min(width, right + step_x)]
            nxt = (left, top, min(width, right + step_x), bottom)
            if strip.size and float(strip.mean()) >= 0.28 and accept(nxt):
                right = nxt[2]
                grown = True
        if top > 0:
            strip = mask[max(0, top - step_y) : top, left:right]
            nxt = (left, max(0, top - step_y), right, bottom)
            if strip.size and float(strip.mean()) >= 0.24 and accept(nxt):
                top = nxt[1]
                grown = True
        if bottom < height:
            strip = mask[bottom : min(height, bottom + step_y), left:right]
            nxt = (left, top, right, min(height, bottom + step_y))
            if strip.size and float(strip.mean()) >= 0.24 and accept(nxt):
                bottom = nxt[3]
                grown = True
        if not grown:
            break
    return left, top, right, bottom


def _cover_and_trim(mask: np.ndarray, box: Sequence[int]) -> Tuple[int, int, int, int]:
    """Snap to the cream panel: pull in missing brand line, drop empty glass."""

    height, width = mask.shape
    left, top, right, bottom = [int(value) for value in box]
    extra_x = max(10, round((right - left) * 0.4))
    extra_y = max(8, round((bottom - top) * 0.22))
    search_left = max(0, left - extra_x)
    search_top = max(0, top - extra_y)
    search_right = min(width, right + extra_x)
    search_bottom = min(height, bottom + extra_y)
    ys, xs = np.where(mask[search_top:search_bottom, search_left:search_right])
    if xs.size >= 60:
        left = search_left + int(xs.min())
        top = search_top + int(ys.min())
        right = search_left + int(xs.max()) + 1
        bottom = search_top + int(ys.max()) + 1

    def column_ok(x: int) -> bool:
        return float(mask[top:bottom, x].mean()) >= 0.1

    def row_ok(y: int) -> bool:
        return float(mask[y, left:right].mean()) >= 0.08

    while left < right - 4 and not column_ok(left):
        left += 1
    while right > left + 4 and not column_ok(right - 1):
        right -= 1
    while top < bottom - 4 and not row_ok(top):
        top += 1
    while bottom > top + 4 and not row_ok(bottom - 1):
        bottom -= 1
    return left, top, right, bottom


def _convex_hull(points: np.ndarray) -> List[Tuple[float, float]]:
    if points.size == 0:
        return []
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return [tuple(map(float, row)) for row in pts]

    def cross(origin, start, end) -> float:
        return (start[0] - origin[0]) * (end[1] - origin[1]) - (start[1] - origin[1]) * (end[0] - origin[0])

    lower: List[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[np.ndarray] = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.vstack([lower[:-1], upper[:-1]])
    return [tuple(map(float, row)) for row in hull]


def _simplify_quad(hull: Sequence[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    points = list(hull)
    if len(points) < 4:
        return None
    while len(points) > 4:
        drop = 0
        smallest = None
        count = len(points)
        for index in range(count):
            prev_pt = points[(index - 1) % count]
            point = points[index]
            next_pt = points[(index + 1) % count]
            area = abs(
                (point[0] - prev_pt[0]) * (next_pt[1] - prev_pt[1])
                - (point[1] - prev_pt[1]) * (next_pt[0] - prev_pt[0])
            )
            if smallest is None or area < smallest:
                smallest = area
                drop = index
        points.pop(drop)
    return points


def _order_quad(points: Sequence[Tuple[float, float]]) -> Quad:
    cx = sum(point[0] for point in points) / 4
    cy = sum(point[1] for point in points) / 4
    ordered = sorted(points, key=lambda point: atan2(point[1] - cy, point[0] - cx))
    start = min(range(4), key=lambda index: ordered[index][0] + ordered[index][1])
    ordered = ordered[start:] + ordered[:start]
    return (ordered[0], ordered[1], ordered[2], ordered[3])


def _quad_area(quad: Sequence[Tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(quad):
        nxt = quad[(index + 1) % 4]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) / 2


def _expand_quad(quad: Quad, pad: float, width: int, height: int) -> Quad:
    cx = sum(point[0] for point in quad) / 4
    cy = sum(point[1] for point in quad) / 4
    out = []
    for x, y in quad:
        out.append(
            (
                min(width - 1, max(0.0, x + (x - cx) * pad)),
                min(height - 1, max(0.0, y + (y - cy) * pad)),
            )
        )
    return _order_quad(out)


def _quad_from_mask(mask: np.ndarray, box: Sequence[int]) -> Optional[Quad]:
    height, width = mask.shape
    left, top, right, bottom = [int(value) for value in box]
    if right - left < 12 or bottom - top < 16:
        return None
    ys, xs = np.where(mask[top:bottom, left:right])
    if xs.size < 80:
        return None
    step = max(1, xs.size // 900)
    points = np.stack((xs[::step] + left, ys[::step] + top), axis=1)
    simplified = _simplify_quad(_convex_hull(points))
    if not simplified:
        return None
    quad = _order_quad(simplified)
    area = _quad_area(quad)
    box_area = max(1, (right - left) * (bottom - top))
    if area < box_area * 0.45:
        return None
    sides = [hypot(quad[i][0] - quad[(i + 1) % 4][0], quad[i][1] - quad[(i + 1) % 4][1]) for i in range(4)]
    if min(sides) < 12:
        return None
    signs = []
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        cx, cy = quad[(i + 2) % 4]
        signs.append((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    if any(value == 0 for value in signs) or (min(signs) < 0) == (max(signs) > 0):
        return None
    width_top = hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    height_side = hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1])
    aspect = height_side / max(width_top, 1e-6)
    if aspect < 0.7 or aspect > 3.4:
        return None
    return _expand_quad(quad, 0.04, width, height)


def _perspective_coeffs(source: Sequence[Tuple[float, float]], dest: Sequence[Tuple[float, float]]):
    matrix = []
    for (x, y), (u, v) in zip(dest, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    try:
        return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(source, dtype=np.float64).reshape(8))
    except np.linalg.LinAlgError:
        return None


def _warp_quad(image: Image.Image, quad: Quad) -> Optional[Image.Image]:
    width, height = image.size
    source = [(x * width, y * height) for x, y in quad]
    out_w = max(32, int(round(max(hypot(source[1][0] - source[0][0], source[1][1] - source[0][1]), hypot(source[2][0] - source[3][0], source[2][1] - source[3][1])))))
    out_h = max(32, int(round(max(hypot(source[3][0] - source[0][0], source[3][1] - source[0][1]), hypot(source[2][0] - source[1][0], source[2][1] - source[1][1])))))
    scale = min(1.0, 1024 / max(out_w, out_h))
    out_w = max(32, int(out_w * scale))
    out_h = max(32, int(out_h * scale))
    dest = ((0.0, 0.0), (out_w - 1.0, 0.0), (out_w - 1.0, out_h - 1.0), (0.0, out_h - 1.0))
    coeffs = _perspective_coeffs(source, dest)
    if coeffs is None:
        return None
    return image.transform((out_w, out_h), Image.Transform.PERSPECTIVE, coeffs.tolist(), Image.Resampling.BICUBIC)


def _grid_boxes(width: int, height: int) -> Iterable[Tuple[int, int, int, int]]:
    widths = (0.18, 0.24, 0.32, 0.42, 0.52, 0.64)
    heights = (0.32, 0.42, 0.52, 0.62, 0.74)
    centres_x = (0.22, 0.32, 0.42, 0.50, 0.58, 0.68, 0.78)
    centres_y = (0.28, 0.38, 0.48, 0.58, 0.68)
    for box_width in widths:
        for box_height in heights:
            for centre_x in centres_x:
                for centre_y in centres_y:
                    left = max(0, round((centre_x - box_width / 2) * width))
                    top = max(0, round((centre_y - box_height / 2) * height))
                    right = min(width, round((centre_x + box_width / 2) * width))
                    bottom = min(height, round((centre_y + box_height / 2) * height))
                    if right - left < width * 0.12 or bottom - top < height * 0.2:
                        continue
                    yield left, top, right, bottom


def detect_label(image: Image.Image) -> LabelDetection:
    """Return a padded bbox around the likely front label, not a drawing crop."""

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((720, 960), Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    gray = pixels.mean(axis=2)
    paper = _paper_mask(pixels)
    height, width = gray.shape
    small_size = (max(48, width // 6), max(64, height // 6))
    small_paper = np.array(
        Image.fromarray(paper.astype(np.uint8) * 255).resize(small_size, Image.Resampling.NEAREST)
    ) > 127
    interior_small = _dilate(_clear_border(small_paper), 2)
    if float(interior_small.mean()) < 0.015:
        interior_small = _dilate(small_paper, 2)
    interior = np.array(
        Image.fromarray(interior_small.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    ) > 127
    filled = interior
    edge = np.zeros_like(gray)
    edge[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    edge[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    edge = np.asarray(
        Image.fromarray(np.clip(edge * 255, 0, 255).astype("uint8")).filter(
            ImageFilter.GaussianBlur(radius=0.7)
        ),
        dtype=np.float32,
    ) / 255.0
    gray_integral = _integral(gray)
    edge_integral = _integral(edge)
    paper_integral = _integral(interior.astype(np.float32))
    paper_total = float(interior.sum())

    def score_box(box: Sequence[int]) -> float:
        return _candidate_score(
            edge,
            gray_integral,
            edge_integral,
            paper_integral,
            paper_total,
            *box,
        )

    best_score = -1.0
    best_box = (round(width * 0.28), round(height * 0.22), round(width * 0.72), round(height * 0.78))
    method = "label_panel"

    scale_x = width / interior_small.shape[1]
    scale_y = height / interior_small.shape[0]
    for left_s, top_s, right_s, bottom_s, area in _label_components(interior_small):
        area_frac = area / interior_small.size
        if area_frac < 0.02 or area_frac > 0.45:
            continue
        box = (
            max(0, round(left_s * scale_x)),
            max(0, round(top_s * scale_y)),
            min(width, round(right_s * scale_x)),
            min(height, round(bottom_s * scale_y)),
        )
        if _frame_like(box, width, height):
            continue
        box = _expand_to_paper(filled, box)
        if _frame_like(box, width, height):
            continue
        if box[2] - box[0] < width * 0.12 or box[3] - box[1] < height * 0.2:
            continue
        score = score_box(box) + 0.06
        if score > best_score:
            best_score = score
            best_box = box
            method = "paper_panel"

    for box in _grid_boxes(width, height):
        if _frame_like(box, width, height):
            continue
        score = score_box(box)
        if score > best_score:
            best_score = score
            best_box = box
            method = "label_panel"

    best_box = _expand_to_paper(filled, best_box)
    best_box = _cover_and_trim(filled, best_box)
    if _frame_like(best_box, width, height):
        best_box = (
            round(width * 0.28),
            round(height * 0.22),
            round(width * 0.72),
            round(height * 0.78),
        )
        best_box = _cover_and_trim(filled, best_box)
    left, top, right, bottom = best_box
    pad_x = max(0.01, (right - left) / width * 0.03)
    pad_y = max(0.01, (bottom - top) / height * 0.03)
    result = (
        max(0.0, left / width - pad_x),
        max(0.0, top / height - pad_y),
        min(1.0, right / width + pad_x),
        min(1.0, bottom / height + pad_y),
    )
    if (result[2] - result[0]) * (result[3] - result[1]) > 0.82:
        result = (left / width, top / height, right / width, bottom / height)
    pixel_quad = _quad_from_mask(filled, (left, top, right, bottom))
    quad = None
    method_name = method
    if pixel_quad:
        quad = (
            (pixel_quad[0][0] / width, pixel_quad[0][1] / height),
            (pixel_quad[1][0] / width, pixel_quad[1][1] / height),
            (pixel_quad[2][0] / width, pixel_quad[2][1] / height),
            (pixel_quad[3][0] / width, pixel_quad[3][1] / height),
        )
        method_name = "paper_quad"
    return LabelDetection(result, round(max(0.0, min(1.0, best_score)), 4), method_name, quad)


def crop_label(image: Image.Image, detection: LabelDetection) -> Image.Image:
    """Crop a detected label, unwarping a paper quad when corners are reliable."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    if detection.quad:
        warped = _warp_quad(image, detection.quad)
        if warped is not None and warped.width >= 32 and warped.height >= 32:
            return warped
    left, top, right, bottom = detection.bbox
    return image.crop(
        (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )
    )


def enhance_label(image: Image.Image) -> Image.Image:
    """Create a mild OCR/visual variant without changing the crop geometry."""

    image = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=105, threshold=3))
