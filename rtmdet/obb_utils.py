import math

import torch
from shapely.geometry import Polygon
from torch import Tensor


def norm_angle(angle: Tensor) -> Tensor:
    """Normalize angle to [-pi/2, pi/2) via le90 convention."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def distance2obb(points: Tensor, distance: Tensor) -> Tensor:
    """Decode distance + angle prediction to oriented bounding boxes.

    Args:
        points: Grid centers, shape [N, 2] or [B, H, W, 2].
        distance: (left, top, right, bottom, angle), shape [N, 5] or [B, H, W, 5].

    Returns:
        Bboxes in (cx, cy, w, h, theta) format. Width >= height, theta in [-pi/2, pi/2).
    """
    lt = distance[..., :2]  # left, top
    rb = distance[..., 2:4]  # right, bottom
    angle = distance[..., 4]

    wh = lt + rb  # w, h
    offset_t = (rb - lt) / 2.0  # center offset in rotated frame

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    # Rotate offset by angle: [[cos, -sin], [sin, cos]] @ offset_t
    offset_x = offset_t[..., 0] * cos_a - offset_t[..., 1] * sin_a
    offset_y = offset_t[..., 0] * sin_a + offset_t[..., 1] * cos_a

    ctr_x = points[..., 0] + offset_x
    ctr_y = points[..., 1] + offset_y

    # Enforce w >= h (le90 convention) — swap w/h and shift angle by pi/2
    w = wh[..., 0]
    h = wh[..., 1]
    swap = h > w
    w, h = torch.where(swap, h, w), torch.where(swap, w, h)
    angle = torch.where(swap, norm_angle(angle + math.pi / 2), angle)

    angle = norm_angle(angle)

    return torch.stack([ctr_x, ctr_y, w, h, angle], dim=-1)


def obb2polygons(obb: Tensor) -> list[list[tuple[float, float]]]:
    """Convert OBB (cx, cy, w, h, theta) to polygon vertex coordinates.

    Args:
        obb: Shape [N, 5], on CPU.

    Returns:
        List of 4 (x, y) tuples per bbox, clockwise order.
    """
    cx, cy, w, h, theta = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    hw = w / 2.0
    hh = h / 2.0

    # Corners in local frame: top-left, top-right, bottom-right, bottom-left
    dx = torch.stack([-hw, hw, hw, -hw], dim=1)
    dy = torch.stack([-hh, -hh, hh, hh], dim=1)

    # Rotate and translate
    rx = dx * cos_t.unsqueeze(1) - dy * sin_t.unsqueeze(1)
    ry = dx * sin_t.unsqueeze(1) + dy * cos_t.unsqueeze(1)
    px = rx + cx.unsqueeze(1)
    py = ry + cy.unsqueeze(1)

    polygons: list[list[tuple[float, float]]] = []
    for i in range(px.shape[0]):
        poly = [(px[i, j].item(), py[i, j].item()) for j in range(4)]
        polygons.append(poly)
    return polygons


def rotated_nms(
    bboxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    max_num: int = 300,
) -> Tensor:
    """Rotated IoU-based NMS using shapely polygon intersection.

    Args:
        bboxes: Shape [N, 5], (cx, cy, w, h, theta), on CPU.
        scores: Shape [N,].
        iou_threshold: IoU threshold for suppression.
        max_num: Maximum detections to keep.

    Returns:
        Indices of kept detections, sorted by score descending.
    """
    if bboxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=bboxes.device)

    order = torch.argsort(scores, descending=True)
    bboxes_sorted = bboxes[order]

    polygons = obb2polygons(bboxes_sorted)
    polys = [Polygon(p) for p in polygons]

    keep: list[int] = []
    suppressed: set[int] = set()

    for i in range(len(polys)):
        if i in suppressed:
            continue
        if len(keep) >= max_num:
            break
        keep.append(int(order[i].item()))

        poly_i = polys[i]
        area_i = poly_i.area

        for j in range(i + 1, len(polys)):
            if j in suppressed:
                continue
            poly_j = polys[j]
            intersection = poly_i.intersection(poly_j)
            union = area_i + poly_j.area - intersection.area
            if union > 0:
                iou = intersection.area / union
                if iou > iou_threshold:
                    suppressed.add(j)

    return torch.tensor(keep, dtype=torch.long, device=bboxes.device)