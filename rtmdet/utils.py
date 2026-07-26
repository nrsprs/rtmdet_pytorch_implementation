import torch
from torch import Tensor
from typing import overload


@overload
def apply_factor(values: int, factor: float) -> int: ...
@overload
def apply_factor(values: list[int], factor: float) -> list[int]: ...


def apply_factor(values, factor: float):
    def _scale(v: int) -> int:
        return max(1, round(v * factor))

    if isinstance(values, int):
        return _scale(values)
    else:
        return [_scale(v) for v in values]


def generate_grid_centers(
    img_size: int,
    strides: list[int],
    device: torch.device = torch.device("cpu"),
) -> list[tuple[Tensor, Tensor]]:
    """Generate grid center coordinates for each pyramid level.

    Returns a list of (cx, cy) tensor pairs, one per level.
    Each tensor has shape [H, W] with pixel coordinates.
    """
    centers = []
    for stride in strides:
        feat_h = img_size // stride
        feat_w = img_size // stride
        ys = torch.arange(feat_h, dtype=torch.float32, device=device)
        xs = torch.arange(feat_w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cx = (grid_x + 0.5) * stride
        cy = (grid_y + 0.5) * stride
        centers.append((cx, cy))
    return centers


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape: tuple[int, int] | None = None,
) -> Tensor:
    """Decode distance prediction to corner-format bounding boxes.

    Args:
        points: Grid centers, shape [B, H, W, 2] or [N, 2].
        distance: Distances (left, top, right, bottom), shape [B, H, W, 4] or [N, 4].
        max_shape: Optional (H, W) for clamping.

    Returns:
        Bboxes in [x1, y1, x2, y2] format.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    if max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes
