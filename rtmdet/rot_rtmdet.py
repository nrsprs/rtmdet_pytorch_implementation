import math
from pathlib import Path
from typing import overload

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from rtmdet.backbone import CSPNext
from rtmdet.checkpoint_utils import (
    _download_if_needed,
    _safe_load_state_dict,
    extract_sub_state_dict,
    load_mmdet_checkpoint,
)
from rtmdet.config import RotRTMDetConfig
from rtmdet.constants import _ROTATED_PRETRAINED_URLS
from rtmdet.neck import CSPNeXtPAFPN
from rtmdet.obb_utils import distance2obb, rotated_nms
from rtmdet.rot_head import RotRTMDetHead
from rtmdet.typings import RotatedPresetName
from rtmdet.utils import generate_grid_centers


class RotRTMDet(nn.Module):
    def __init__(self, cfg: RotRTMDetConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CSPNext(cfg)
        self.neck = CSPNeXtPAFPN(cfg)
        self.head = RotRTMDetHead(cfg)
        # Cache grid centers for bbox decoding (computed once, moved to device at inference)
        self._grid_centers = generate_grid_centers(cfg.img_size, cfg.prior_strides, torch.device("cpu"))

    @staticmethod
    def _default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def from_preset(
        cls,
        name: RotatedPresetName,
        img_size: int | None = None,
        num_classes: int | None = None,
        pretrained: bool = False,
    ) -> "RotRTMDet":
        cfg = RotRTMDetConfig.from_preset(name)

        if img_size is not None:
            cfg.img_size = img_size
        if num_classes is not None:
            cfg.num_classes = num_classes

        model = cls(cfg)

        if pretrained:
            model._load_pretrained_weights(name)

        return model.to(cls._default_device())

    def _load_pretrained_weights(self, name: RotatedPresetName) -> None:
        url = _ROTATED_PRETRAINED_URLS[name]
        checkpoint_path = _download_if_needed(url)
        full_sd = load_mmdet_checkpoint(str(checkpoint_path), map_location="cpu")

        backbone_sd = extract_sub_state_dict(full_sd, "backbone.")
        neck_sd = extract_sub_state_dict(full_sd, "neck.")
        head_sd = extract_sub_state_dict(full_sd, "bbox_head.")

        _safe_load_state_dict(self.backbone, backbone_sd)
        _safe_load_state_dict(self.neck, neck_sd)
        _safe_load_state_dict(self.head, head_sd)

    def to_file(self, path: str | Path) -> None:
        """Save the model state dict and config metadata to a .pt file."""
        path = Path(path)
        assert self.cfg.preset_name, (
            "Cannot save model — preset_name is empty. "
            "Use from_preset() or set cfg.preset_name manually."
        )
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "name": self.cfg.preset_name,
                "num_classes": self.cfg.num_classes,
                "img_size": self.cfg.img_size,
            },
        }
        torch.save(checkpoint, str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> "RotRTMDet":
        """Load a model saved by `to_file()`, auto-detecting the architecture."""
        path = Path(path)
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = checkpoint["state_dict"]
        config = checkpoint["config"]

        name = config["name"]
        cfg = RotRTMDetConfig.from_preset(name)
        cfg.num_classes = config.get("num_classes", cfg.num_classes)
        cfg.img_size = config.get("img_size", cfg.img_size)

        model = cls(cfg)
        model.load_state_dict(sd, strict=True)
        return model.to(cls._default_device())

    def predict(self, path: str) -> tuple[Tensor, Tensor, Tensor]:
        """Run single-image inference and return post-processed rotated detections.

        Args:
            path: Image file path.

        Returns:
            (bboxes[N,5], scores[N], classes[N]) — OBB in (cx, cy, w, h, theta) format
        """
        return self._inference_from_path(path)

    def forward(
        self, x: Tensor, return_logits: bool = False
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]] | tuple[Tensor, Tensor, Tensor]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_scores, bbox_preds, angle_preds = self.head(feats)

        if not return_logits:
            return cls_scores, bbox_preds, angle_preds

        # Decode OBBs using distance2obb
        decoded_bboxes = []
        for level_idx, bbox_pred in enumerate(bbox_preds):
            stride = self.cfg.prior_strides[level_idx]

            cx, cy = self._grid_centers[level_idx]
            cx, cy = cx.to(bbox_pred.device), cy.to(bbox_pred.device)

            distances = (
                torch.exp(torch.clamp(bbox_pred, min=0)) if self.cfg.exp_on_reg
                else torch.clamp(bbox_pred, min=0)
            ) * stride  # [B, 4, H, W]
            angle = angle_preds[level_idx].permute(0, 2, 3, 1).squeeze(-1)  # [B, H, W]
            dist = distances.permute(0, 2, 3, 1)  # [B, H, W, 4]
            dist_with_angle = torch.cat([dist, angle.unsqueeze(-1)], dim=-1)  # [B, H, W, 5]

            points = torch.stack([cx, cy], dim=-1).unsqueeze(0).expand(
                distances.shape[0], -1, -1, -1
            )
            bboxes = distance2obb(points, dist_with_angle)  # [B, H, W, 5]
            decoded_bboxes.append(bboxes.flatten(1, 2))  # [B, H*W, 5]

        bboxes = torch.cat(decoded_bboxes, dim=1)  # [B, N_total, 5]
        cls = torch.cat([torch.sigmoid(s).flatten(2) for s in cls_scores], dim=2).permute(0, 2, 1)

        return bboxes, torch.zeros(0), cls

    @overload
    def __call__(
        self, image_input: str, return_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def __call__(
        self, image_input: Tensor, return_logits: bool = False
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]] | tuple[Tensor, Tensor, Tensor]: ...

    def __call__(
        self, image_input: str | Tensor, return_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]:
        if isinstance(image_input, str):
            return self._inference_from_path(image_input)

        return self.forward(image_input, return_logits=return_logits)

    def _inference_from_path(self, path: str) -> tuple[Tensor, Tensor, Tensor]:
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        target = self.cfg.img_size

        scale = target / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        padded = Image.new("RGB", (target, target), (0, 0, 0))
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        padded.paste(img, (pad_x, pad_y))

        tensor = torch.from_numpy(np.array(padded, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Normalize with ImageNet mean/std (required for pretrained weights)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        self.eval()
        feats = self.backbone(tensor)
        feats = self.neck(feats)
        cls_scores, bbox_preds, angle_preds = self.head(feats)

        bboxes, scores, class_ids = self._postprocess(cls_scores, bbox_preds, angle_preds)

        # Transform from padded target -> original image coordinates
        bboxes = bboxes.clone()
        bboxes[:, 0] = (bboxes[:, 0] - pad_x) / scale  # cx
        bboxes[:, 1] = (bboxes[:, 1] - pad_y) / scale  # cy
        bboxes[:, 2] = bboxes[:, 2] / scale  # w
        bboxes[:, 3] = bboxes[:, 3] / scale  # h
        # angle (bboxes[:, 4]) unchanged — rotation is scale-invariant
        bboxes[:, 0].clamp_(0, orig_w)
        bboxes[:, 1].clamp_(0, orig_h)

        return bboxes, scores, class_ids

    def _postprocess(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        angle_preds: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device

        decoded_bboxes = []
        cls_preds_list = []

        for level_idx, (bbox_pred, cls_score, angle_pred) in enumerate(
            zip(bbox_preds, cls_scores, angle_preds)
        ):
            stride = self.cfg.prior_strides[level_idx]

            # Grid centers for this level
            cx, cy = self._grid_centers[level_idx]
            cx, cy = cx.to(device), cy.to(device)

            # Decode raw distances: clamp(>=0) * stride -> pixel distances [B, 4, H, W]
            distances = (
                torch.exp(torch.clamp(bbox_pred, min=0)) if self.cfg.exp_on_reg
                else torch.clamp(bbox_pred, min=0)
            ) * stride

            # Build points [1, H, W, 2] and distances [B, H, W, 4]
            points = torch.stack([cx, cy], dim=-1).unsqueeze(0)
            dist = distances.permute(0, 2, 3, 1)

            # Concat angle into distance tensor [B, H, W, 5]
            angle = angle_pred.permute(0, 2, 3, 1).squeeze(-1)
            dist_with_angle = torch.cat([dist, angle.unsqueeze(-1)], dim=-1)

            # Decode to oriented bboxes [B, H, W, 5]
            bboxes = distance2obb(points, dist_with_angle)
            decoded_bboxes.append(bboxes.flatten(1, 2))

            # Classification: sigmoid -> [B, num_classes, H*W]
            cls_preds_list.append(torch.sigmoid(cls_score).flatten(2))

        # Concatenate across levels: [B, N_total, 5] and [B, num_classes, N_total]
        bboxes = torch.cat(decoded_bboxes, dim=1)
        scores = torch.cat(cls_preds_list, dim=2)

        # Single-image inference: take batch index 0
        bboxes = bboxes[0]  # [N, 5]
        scores = scores[0]  # [num_classes, N]

        # Get max score and class per bbox
        scores, class_ids = scores.max(dim=0)

        # Filter by score threshold
        keep = scores > self.cfg.score_threshold
        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Rotated NMS — falls back to axis-aligned NMS when angles are zero
        if bboxes.numel() > 0:
            keep = rotated_nms(bboxes.cpu(), scores.cpu(), self.cfg.nms_iou_threshold)
            keep = keep[: self.cfg.max_num_detections]
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        return bboxes.to(device), scores.to(device), class_ids.to(device)

    def draw_detections(
        self,
        image_input: str | Tensor,
        bboxes: Tensor,
        scores: Tensor,
        classes: Tensor,
    ) -> Image.Image:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = Image.fromarray(
                (image_input.detach().cpu().numpy() * 255).astype(np.uint8)
            )

        draw = ImageDraw.Draw(img)
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
            (255, 0, 128),
        ]

        for bbox, score, cls in zip(bboxes, scores, classes):
            cx, cy, w, h, theta = bbox.tolist()
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            hw, hh = w / 2, h / 2
            # Four corners: top-left, top-right, bottom-right, bottom-left
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            rotated = [
                (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
                for x, y in corners
            ]
            color = colors[int(cls) % len(colors)]
            draw.polygon(rotated, outline=color, width=2)
            draw.text(
                (int(cx) + 4, int(cy) + 4),
                f"cls {int(cls)} {score:.2f}",
                fill=color,
            )

        return img

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device