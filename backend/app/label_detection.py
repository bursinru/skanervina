"""Find the front paper label on a bottle photo.

The crop must cover the whole panel: brand line, vintage, illustration and
layout.  A tight box around a drawing or a centre-biased slice of glass
throws SigLIP at the wrong wine.
"""

from dataclasses import dataclass
from math import atan2, hypot
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


NormalizedBox = Tuple[float, float, float, float]
Quad = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def label_rgb(image):
    image = ImageOps.exif_transpose(image)
    if 'A' in image.getbands() or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA', image.size, 'white'), rgba).convert('RGB')
    return image.convert('RGB')


@dataclass(frozen=True)
class LabelDetection:
    bbox: NormalizedBox
    confidence: float
    method: str = "label_panel"
    quad: Optional[Quad] = None
    contour: Optional[Tuple[Tuple[float, float], ...]] = None


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


def _studio_background(pixels: np.ndarray) -> np.ndarray:
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    value = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.where(value > 1e-4, 1.0 - minimum / np.maximum(value, 1e-4), 0.0)
    return (value > 0.86) & (saturation < 0.10)


def _paper_mask(pixels: np.ndarray) -> np.ndarray:
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    value = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.where(value > 1e-4, 1.0 - minimum / np.maximum(value, 1e-4), 0.0)
    # Cream, white and pale gold stock.  Dark glass and black UI drop out.
    light_paper = (value > 0.40) & (value < 0.985) & (saturation < 0.42)
    warm = (red + green) > (blue * 1.02)
    return light_paper & warm


def _usable_label_box(pixels: np.ndarray, box: Sequence[int]) -> bool:
    """Drop dark glass, blurry shelf bottles and textureless cardboard."""

    height, width = pixels.shape[:2]
    left, top, right, bottom = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    region = pixels[top:bottom, left:right]
    gray = region.mean(axis=2)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 0.40 or std < 0.07:
        return False
    area = ((right - left) / width) * ((bottom - top) / height)
    centre_x = (left + right) / 2 / width
    if area < 0.06 and abs(centre_x - 0.5) > 0.20:
        return False
    width_frac = (right - left) / width
    height_frac = (bottom - top) / height
    if height_frac < width_frac and left > width * 0.40:
        paper = _paper_mask(pixels)
        if float(paper[top:bottom, :left].mean()) > 0.22:
            return False
    return True


def _amber_panel(pixels: np.ndarray):
    """Gold/orange labels printed on glass, not cream paper stock."""

    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    value = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.where(value > 1e-4, 1.0 - minimum / np.maximum(value, 1e-4), 0.0)
    amber = (red > green + 0.04) & (red > blue + 0.10) & (saturation > 0.32) & (value > 0.40) & (value < 0.90)
    height, width = amber.shape
    if float(amber.mean()) < 0.004:
        return None
    small_w, small_h = max(24, width // 8), max(32, height // 8)
    small = np.array(Image.fromarray(amber.astype(np.uint8) * 255).resize((small_w, small_h), Image.Resampling.NEAREST)) > 127
    small = _close(_erode(small, 1), 2)
    visited = np.zeros_like(small)
    best = None
    for sy, sx in zip(*np.where(small)):
        if visited[sy, sx]:
            continue
        stack = [(int(sx), int(sy))]
        points = []
        visited[sy, sx] = True
        while stack:
            x, y = stack.pop()
            points.append((x, y))
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < small_w and 0 <= ny < small_h and small[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        if len(points) < small_w * small_h * 0.012:
            continue
        pts = np.asarray(points)
        left_s, top_s = pts.min(axis=0)
        right_s, bottom_s = pts.max(axis=0) + 1
        bw, bh = right_s - left_s, bottom_s - top_s
        if bw < small_w * 0.08 or bh < small_h * 0.06:
            continue
        scale_x, scale_y = width / small_w, height / small_h
        box = (
            max(0, round(left_s * scale_x)),
            max(0, round(top_s * scale_y)),
            min(width, round(right_s * scale_x)),
            min(height, round(bottom_s * scale_y)),
        )
        area = ((box[2] - box[0]) / width) * ((box[3] - box[1]) / height)
        centre_x = (box[0] + box[2]) / 2 / width
        centre_y = (box[1] + box[3]) / 2 / height
        if area < 0.012 or area > 0.22 or centre_x < 0.28 or centre_x > 0.72 or centre_y < 0.40 or centre_y > 0.82:
            continue
        score = len(points) * (1 - abs(centre_x - 0.5))
        if best is None or score > best[0]:
            best = (score, box)
    if best is None:
        return None
    left, top, right, bottom = best[1]
    band = pixels[top:bottom, left:right]
    row_value = band.max(axis=2).mean(axis=1)
    keep = np.flatnonzero(row_value > max(0.45, float(row_value.max()) * 0.62))
    if keep.size:
        top, bottom = top + int(keep[0]), top + int(keep[-1]) + 1
    return left, top, right, bottom


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 127


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.filter(ImageFilter.MinFilter(size))) > 127


def _close(mask: np.ndarray, radius: int) -> np.ndarray:
    return _erode(_dilate(mask, radius), max(1, radius - 1))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _longest_run(flags: np.ndarray) -> Tuple[int, int]:
    best = (0, 0)
    start = None
    for index, on in enumerate(flags):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and len(flags) - start > best[1] - best[0]:
        best = (start, len(flags))
    return best


def _run_containing(flags: np.ndarray, index: int) -> Tuple[int, int]:
    if index < 0 or index >= len(flags) or not flags[index]:
        return _longest_run(flags)
    start = index
    while start > 0 and flags[start - 1]:
        start -= 1
    end = index + 1
    while end < len(flags) and flags[end]:
        end += 1
    return start, end


def _paper_band_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Front paper panel from the bottle base up, stopping before glass."""

    height, width = mask.shape
    row = _smooth(mask.mean(axis=1), max(7, height // 35))
    bottom = int(height) - 1
    while bottom > height * 0.35 and row[bottom] < 0.12:
        bottom -= 1
    if row[bottom] < 0.12:
        return None
    top = bottom
    min_span = int(height * (0.22 if height >= width * 1.7 else 0.12))
    max_span = int(height * (0.50 if height >= width * 1.7 else 0.55))
    while top > int(height * 0.06) and bottom - top < max_span:
        candidate = top - 1
        span = bottom - candidate
        if span >= min_span and row[candidate] < 0.25:
            y = candidate
            while y > 0 and row[y] < 0.30:
                y -= 1
            peak_end = y
            while y > 0 and row[y] >= 0.28:
                y -= 1
            if peak_end - y > height * 0.14:
                break
        if span >= min_span and row[candidate] < 0.14:
            break
        top = candidate
    probe_from = max(int(height * 0.08), top - height // 10)
    header_top = top
    y = top
    while y > probe_from and row[y - 1] < 0.30:
        y -= 1
    skipped = top - y
    if 0 < skipped <= max(16, height // 20):
        header_bottom = y
        while y > probe_from and row[y - 1] >= 0.30:
            y -= 1
        header_len = header_bottom - y
        if height * 0.05 < header_len < height * 0.16:
            header_top = y
    top = header_top
    if bottom - top < height * 0.08 or bottom - top > height * 0.62:
        return None
    band = mask[top:bottom]
    col = _smooth(band.mean(axis=0), max(5, width // 28))
    cpeak = float(col.max())
    if cpeak < 0.12:
        return None
    left, right = _run_containing(col >= max(0.08, cpeak * 0.22), int(col.argmax()))
    if right - left < width * 0.28:
        return None
    if left > width * 0.28:
        return None
    return left, top, right, bottom


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


def _frame_like(box: Sequence[int], width: int, height: int, *, allow_wide: bool = False) -> bool:
    left, top, right, bottom = box
    area = (right - left) * (bottom - top) / max(1, width * height)
    touches = (
        int(left <= 2)
        + int(top <= 2)
        + int(right >= width - 2)
        + int(bottom >= height - 2)
    )
    too_wide = (not allow_wide) and (right - left) / width > 0.82
    return area > 0.48 or touches >= 3 or too_wide


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


def _cover_and_trim(mask: np.ndarray, box: Sequence[int], extra: float = 0.4) -> Tuple[int, int, int, int]:
    """Snap to the cream panel: pull in missing brand line, drop empty glass."""

    height, width = mask.shape
    left, top, right, bottom = [int(value) for value in box]
    extra_x = max(8, round((right - left) * extra))
    extra_y = max(6, round((bottom - top) * extra * 0.6))
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
    # Keep boundary extrema on every row; strided interior sampling loses corners.
    points = []
    for y in np.unique(ys):
        row_x = np.flatnonzero(mask[y + top, left:right])
        points.extend(((row_x.min() + left, y + top), (row_x.max() + left, y + top)))
    points = np.asarray(points)
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

    def mostly_horizontal(start, end) -> bool:
        return abs(end[0] - start[0]) >= abs(end[1] - start[1]) * 1.6

    def mostly_vertical(start, end) -> bool:
        return abs(end[1] - start[1]) >= abs(end[0] - start[0]) * 1.6

    if not (
        mostly_horizontal(quad[0], quad[1])
        and mostly_horizontal(quad[3], quad[2])
        and mostly_vertical(quad[0], quad[3])
        and mostly_vertical(quad[1], quad[2])
    ):
        return None
    return _expand_quad(quad, 0.04, width, height)


def _panel_contour(mask: np.ndarray, box: Sequence[int]):
    """Follow the paper silhouette, filling printed holes without flattening its top.

    Only accept a coherent panel; uncertain segmentation keeps the original crop.
    The column envelope retains arched tops and die-cut edges.
    """
    left, top, right, bottom = map(int, box)
    region = mask[top:bottom, left:right]
    if not region.size:
        return None
    columns = []
    for x in range(region.shape[1]):
        ys = np.flatnonzero(region[:, x])
        if len(ys) >= max(6, region.shape[0] * .03):
            columns.append((x + left, int(ys[0]) + top, int(ys[-1]) + top))
    # Ignore isolated reflections beside the panel instead of joining them to it.
    runs = [[]]
    for column in columns:
        if runs[-1] and column[0] - runs[-1][-1][0] > 3:
            runs.append([])
        runs[-1].append(column)
    columns = max(runs, key=len)
    if len(columns) < max(12, region.shape[1] * .45):
        return None
    # Specular streaks on glass create narrow spikes above the paper. A median
    # envelope rejects those while retaining broad arches and the lower die-cut.
    radius = max(2, len(columns) // 20)
    tops = [c[1] for c in columns]
    columns = [(x, int(np.median(tops[max(0, i-radius):i+radius+1])), y1)
               for i, (x, y0, y1) in enumerate(columns)]
    stride = max(1, len(columns) // 64)
    samples = columns[::stride]
    if samples[-1] != columns[-1]:
        samples.append(columns[-1])
    points = [(x, y0) for x, y0, y1 in samples]
    points += [(x, y1) for x, y0, y1 in reversed(samples)]
    return tuple(points)


def _quad_matches_contour(quad, contour, size):
    """A curved/scalloped panel must never be forced into a projective rectangle."""
    if not quad or not contour:
        return False
    panel = Image.new("1", size)
    rectangle = Image.new("1", size)
    ImageDraw.Draw(panel).polygon(contour, fill=1)
    ImageDraw.Draw(rectangle).polygon(quad, fill=1)
    a, b = np.asarray(panel), np.asarray(rectangle)
    intersection = np.count_nonzero(a & b)
    return (intersection / max(1, np.count_nonzero(a)) > .98
            and intersection / max(1, np.count_nonzero(b)) > .90)


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


def _connected_panel(pixels, *, catalog=False):
    """Select one coherent paper component before tracing its silhouette.

    Opening severs narrow glass reflections; closing bridges text strokes.
    Background shelves must not be joined via column envelopes.
    """
    h, w = pixels.shape[:2]
    scale = min(1., 320 / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    paper = _paper_mask(pixels) | ((pixels.min(axis=2) > .85))
    if catalog:
        # Transparent packshots are often decoded onto pure white. That white
        # must not join the paper panel to the image frame.  Orange/ochre print
        # on the same sticker is still the panel, not glass.
        cream = paper & (pixels.min(axis=2) < .985) & (pixels.mean(axis=2) > .58)
        studio = float(_studio_background(pixels).mean())
        if studio >= 0.10:
            painted = (
                (pixels.mean(axis=2) > .18)
                & (pixels.mean(axis=2) < .72)
                & ~_studio_background(pixels)
            )
            paper = cream | (_dilate(cream, 12) & painted)
        else:
            paper = cream
    small = np.asarray(Image.fromarray(paper.astype('uint8') * 255).resize((sw, sh))) > 127
    if not catalog:
        # A table/wall can touch the label along one edge. Sever broad horizontal
        # background bands before connected components, rather than rejecting the
        # entire label as a component touching the frame.
        for row in small:
            if row[0]:
                end = next((i for i, value in enumerate(row) if not value), sw)
                row[:end] = False
            if row[-1]:
                start = next((i for i in range(sw - 1, -1, -1) if not row[i]), -1)
                row[start + 1:] = False
            left, right = _longest_run(row)
            if right - left > sw * .72:
                row[left:right] = False
    small = _dilate(_erode(small, 4), 4)
    small = _close(small, 2)
    visited = np.zeros_like(small)
    candidates = []
    for sy, sx in zip(*np.where(small)):
        if visited[sy, sx]:
            continue
        stack = [(int(sx), int(sy))]; points = []; visited[sy, sx] = True
        while stack:
            x, y = stack.pop(); points.append((x, y))
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < sw and 0 <= ny < sh and small[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx] = True; stack.append((nx,ny))
        if len(points) < sw * sh * .018:
            continue
        pts = np.asarray(points); l,t = pts.min(axis=0); r,b = pts.max(axis=0) + 1
        bw,bh = r-l,b-t
        density = len(points) / (bw*bh)
        if bw < sw*.12 or bh < sh*.12 or not .35 < bh/bw < 4.0 or density < .42:
            continue
        if t <= 1 or b >= sh-1 or (not catalog and (l <= 1 or r >= sw-1)):
            continue
        # A foreground panel is generally larger than neighboring shelf labels.
        center = abs((l+r)/2/sw-.5)
        score = len(points) * density * (1 - .65*center)
        candidates.append((score, pts, (l,t,r,b)))
    if not candidates:
        return None
    candidates.sort(key=lambda item:item[0], reverse=True)
    _, points, box = candidates[0]
    # Trace within this component's box; retain dark printed artwork inside it.
    mask = _close(paper, 3)
    l,t,r,b = box
    if catalog:
        # Narrow reflections above/below the paper are not part of the label.
        density = _smooth(small[t:b, l:r].mean(axis=1), 5)
        occupied = np.flatnonzero(density > max(.35, float(density.max()) * .55))
        if occupied.size and occupied[-1] - occupied[0] >= (b - t) * .4:
            t, b = t + int(occupied[0]), t + int(occupied[-1]) + 1
    # Recover dark printed artwork below the bright core, or the lower panel
    # after cutting a table connection. Follow its interior until dark glass.
    recovery = paper if catalog else _paper_mask(pixels)
    original_small = np.asarray(Image.fromarray(recovery.astype('uint8') * 255).resize((sw, sh))) > 127
    inset = max(1, (r - l) // 4)
    limit = min(sh - 1, b + (b - t))
    while b < limit and original_small[b, l + inset:r - inset].mean() > .55:
        b += 1
    if catalog:
        right_limit = min(sw - 1, r + max(4, (r - l) // 2), l + max(r - l, int(sw * 0.55)))
        while r < right_limit and original_small[t + max(1, (b - t) // 6): b - max(1, (b - t) // 6), min(sw - 1, r)].mean() > .18:
            r += 1
        left_limit = max(0, r - max(r - l, int(sw * 0.55)))
        while l > left_limit and original_small[t + max(1, (b - t) // 6): b - max(1, (b - t) // 6), l - 1].mean() > .18:
            l -= 1
    if not catalog:
        ceiling = max(0, t - max(3, (b - t) // 3))
        while t > ceiling and original_small[t - 1, l + inset:r - inset].mean() > .28:
            t -= 1
    box = (max(0,int((l-3)/sw*w)),max(0,int((t-3)/sh*h)),min(w,int((r+3)/sw*w)),min(h,int((b+3)/sh*h)))
    if catalog:
        left, top, right, bottom = box
        painted = (
            (pixels[:, :, 0] > pixels[:, :, 2] + 0.08)
            & (pixels[:, :, 0] > 0.28)
            & (pixels.mean(axis=2) < 0.78)
            & ~_studio_background(pixels)
        )
        mid_top = top + max(1, (bottom - top) // 6)
        mid_bottom = bottom - max(1, (bottom - top) // 6)
        right_limit = min(w, left + max(right - left, int(w * 0.72)))
        while right < right_limit and float(painted[mid_top:mid_bottom, min(w - 1, right)].mean()) > 0.04:
            right += 1
        band = np.zeros_like(mask)
        band[top:bottom, left:min(w, right + 2)] = True
        mask = mask | (painted & band)
        box = (left, top, right, bottom)
    return mask, box


def detect_label(image: Image.Image, *, catalog: bool = False) -> LabelDetection:
    """Return a padded bbox around the likely front label, not a drawing crop."""

    prepared = label_rgb(image)
    prepared.thumbnail((720, 960), Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    gray = pixels.mean(axis=2)
    paper = _paper_mask(pixels) & np.logical_not(_studio_background(pixels))
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
    closed = _close(paper, 6)
    if float(closed.mean()) < 0.012:
        closed = _close(_paper_mask(pixels), 6)
    filled = np.maximum(interior, closed)
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

    studio_frac = float(_studio_background(pixels).mean())
    tall_packshot = height >= width * 1.7 and (catalog or studio_frac >= 0.15)
    band = _paper_band_box(closed)
    if band and not _frame_like(band, width, height, allow_wide=True):
        score = score_box(band) + (0.45 if tall_packshot else 0.04)
        if score > best_score or tall_packshot:
            best_score = max(score, best_score)
            best_box = band
            method = "paper_band"
    if method != "paper_band" or not tall_packshot:
        for box in _grid_boxes(width, height):
            if _frame_like(box, width, height):
                continue
            score = score_box(box)
            if score > best_score:
                best_score = score
                best_box = box
                method = "label_panel"

    if method != "paper_band":
        best_box = _expand_to_paper(filled, best_box)
        best_box = _cover_and_trim(closed, best_box, extra=0.35)
        if _frame_like(best_box, width, height):
            best_box = (
                round(width * 0.28),
                round(height * 0.22),
                round(width * 0.72),
                round(height * 0.78),
            )
            best_box = _cover_and_trim(filled, best_box)
    scored_box, scored_method = best_box, method
    selected_panel = _connected_panel(pixels, catalog=catalog)
    if selected_panel and (catalog or _usable_label_box(pixels, selected_panel[1])):
        panel_mask, best_box = selected_panel
        method = "connected_paper"
    else:
        panel_mask = None
        best_box = scored_box
        method = scored_method
        if method == "paper_band":
            best_box = _cover_and_trim(closed, best_box, extra=0.12)
        wide_band = method == "paper_band" and (best_box[2] - best_box[0]) > width * 0.82
        if not catalog and (not _usable_label_box(pixels, best_box) or wide_band):
            amber = None if wide_band else _amber_panel(pixels)
            if amber is not None:
                best_box = amber
                method = "amber_panel"
            else:
                alt = detect_label(image, catalog=True)
                alt_box = (
                    round(alt.bbox[0] * width),
                    round(alt.bbox[1] * height),
                    round(alt.bbox[2] * width),
                    round(alt.bbox[3] * height),
                )
                alt_area = (alt.bbox[2] - alt.bbox[0]) * (alt.bbox[3] - alt.bbox[1])
                if _usable_label_box(pixels, alt_box) and alt_area < 0.72:
                    if not (wide_band and alt.bbox[0] >= 0.22):
                        return alt
                    return alt
                if float(pixels[max(0, best_box[1]):best_box[3], max(0, best_box[0]):best_box[2]].mean()) < 0.42:
                    return LabelDetection((0.0, 0.0, 1.0, 1.0), 0.12, "full_frame", None, None)
    left, top, right, bottom = best_box
    pad_x = max(0.01, (right - left) / width * 0.03)
    pad_y = max(0.01, (bottom - top) / height * 0.03)
    if method == "paper_band":
        pad_x = max(0.07, (right - left) / width * 0.08)
        pad_y = max(0.012, (bottom - top) / height * 0.02)
    result = (
        max(0.0, left / width - pad_x),
        max(0.0, top / height - pad_y),
        min(1.0, right / width + pad_x),
        min(1.0, bottom / height + pad_y),
    )
    if (result[2] - result[0]) * (result[3] - result[1]) > 0.82:
        result = (left / width, top / height, right / width, bottom / height)
    pixel_quad = None if (tall_packshot or method == "paper_band") else _quad_from_mask(panel_mask if panel_mask is not None else closed, (left, top, right, bottom))
    pixel_contour = _panel_contour(panel_mask, (left, top, right, bottom)) if panel_mask is not None else None
    if not _quad_matches_contour(pixel_quad, pixel_contour, (width, height)):
        pixel_quad = None
    contour = tuple((x / width, y / height) for x, y in pixel_contour) if pixel_contour else None
    quad = None
    method_name = method
    if pixel_quad:
        quad_h = max(point[1] for point in pixel_quad) - min(point[1] for point in pixel_quad)
        if height > width * 1.45 and quad_h > height * 0.55:
            pixel_quad = None
    if pixel_quad:
        quad = (
            (pixel_quad[0][0] / width, pixel_quad[0][1] / height),
            (pixel_quad[1][0] / width, pixel_quad[1][1] / height),
            (pixel_quad[2][0] / width, pixel_quad[2][1] / height),
            (pixel_quad[3][0] / width, pixel_quad[3][1] / height),
        )
        method_name = "paper_quad"
    return LabelDetection(result, round(max(0.0, min(1.0, best_score)), 4), method_name, quad, contour)


def crop_quality(image: Image.Image, detection: LabelDetection) -> dict:
    """Conservative geometry gate; scores are diagnostics, not probabilities."""
    left, top, right, bottom = detection.bbox
    width, height = max(0, right - left), max(0, bottom - top)
    area = width * height
    aspect = width * image.width / max(height * image.height, 1)
    reasons = []
    if area < 0.025:
        reasons.append("tiny_region")
    if aspect < 0.32 and area < 0.12:
        reasons.append("narrow_fragment")
    if aspect > 5 and area < 0.12:
        reasons.append("thin_fragment")
    if area < 0.06 and (top + bottom) / 2 < 0.40:
        reasons.append("small_upper_fragment")
    return {"usable": not reasons, "reasons": reasons,
            "area_fraction": round(area, 4), "aspect_ratio": round(aspect, 4)}


def crop_front_design(image: Image.Image, detection: LabelDetection) -> Image.Image:
    """Conservative printed-glass fallback below a rejected upper highlight.

    Keep the source pixels untouched: glass printing has no paper contour to
    mask or planar surface to rectify. The rejected region supplies only the
    horizontal position of the bottle, never the crop content.
    """
    image = label_rgb(image)
    left, top, right, bottom = detection.bbox
    centre = (left + right) / 2
    bounds = (
        max(0.0, centre - 0.32),
        min(0.72, max(0.38, bottom + 0.02)),
        min(1.0, centre + 0.22),
        0.84,
    )
    if bounds[1] >= bounds[3] or bounds[2] <= bounds[0]:
        return image
    return image.crop((
        round(bounds[0] * image.width), round(bounds[1] * image.height),
        round(bounds[2] * image.width), round(bounds[3] * image.height),
    ))


def crop_label(image: Image.Image, detection: LabelDetection, *, preserve_pixels: bool = False) -> Image.Image:
    """Crop a detected label, unwarping a paper quad when corners are reliable."""

    image = label_rgb(image)
    if detection.method == "full_frame":
        width, height = image.size
        return image.crop((round(width * 0.12), round(height * 0.32), round(width * 0.88), round(height * 0.98)))
    if detection.contour and not preserve_pixels:
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).polygon([(x * image.width, y * image.height) for x, y in detection.contour], fill=255)
        # Neutral background for embeddings and OCR, same silhouette as the UI.
        image = Image.composite(image, Image.new("RGB", image.size, "white"), mask)
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
