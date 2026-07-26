
from typing import overload

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torchvision.ops import nms

from rtmdet.backbone import CSPNext
from rtmdet.checkpoint_utils import (
    _download_if_needed,
    _safe_load_state_dict,
    extract_sub_state_dict,
    load_mmdet_checkpoint,
)
from rtmdet.config import RTMDetConfig
from rtmdet.constants import _PRETRAINED_URLS
from rtmdet.head import RTMDetHead
from rtmdet.neck import CSPNeXtPAFPN
from rtmdet.typings import PresetName
from rtmdet.utils import distance2bbox, generate_grid_centers


class RTMDet(nn.Module):
    def __init__(self, cfg: RTMDetConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CSPNext(cfg)
        self.neck = CSPNeXtPAFPN(cfg)
        self.head = RTMDetHead(cfg)
        # Cache grid centers for bbox decoding (computed once, moved to device at inference)
        self._grid_centers = generate_grid_centers(cfg.img_size, cfg.prior_strides, torch.device("cpu"))

    @classmethod
    def from_preset(
        cls,
        name: PresetName,
        img_size: int | None = None,
        num_classes: int | None = None,
        pretrained: bool = False,
    ) -> "RTMDet":
        cfg = RTMDetConfig.from_preset(name)

        if img_size is not None:
            cfg.img_size = img_size
        if num_classes is not None:
            cfg.num_classes = num_classes

        model = cls(cfg)

        if pretrained:
            model._load_pretrained_weights(name)

        return model

    def _load_pretrained_weights(self, name: PresetName) -> None:
        url = _PRETRAINED_URLS[name]
        checkpoint_path = _download_if_needed(url)
        full_sd = load_mmdet_checkpoint(str(checkpoint_path), map_location="cpu")

        backbone_sd = extract_sub_state_dict(full_sd, "backbone.")
        neck_sd = extract_sub_state_dict(full_sd, "neck.")
        head_sd = extract_sub_state_dict(full_sd, "bbox_head.")

        _safe_load_state_dict(self.backbone, backbone_sd)
        _safe_load_state_dict(self.neck, neck_sd)
        _safe_load_state_dict(self.head, head_sd)

    def forward(
        self, x: Tensor, return_logits: bool = False
    ) -> tuple[list[Tensor], list[Tensor]] | tuple[Tensor, Tensor, Tensor, Tensor]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_scores, bbox_preds = self.head(feats)

        if not return_logits:
            return cls_scores, bbox_preds

        # Decode bboxes using distance-based approach
        decoded_bboxes = []
        for level_idx, bbox_pred in enumerate(bbox_preds):
            stride = self.cfg.prior_strides[level_idx]

            cx, cy = self._grid_centers[level_idx]
            cx, cy = cx.to(bbox_pred.device), cy.to(bbox_pred.device)

            distances = torch.clamp(bbox_pred, min=0) * stride  # [B, 4, H, W]
            points = torch.stack([cx, cy], dim=-1).unsqueeze(0).expand(
                distances.shape[0], -1, -1, -1
            )
            dist = distances.permute(0, 2, 3, 1)  # [B, H, W, 4]
            bboxes = distance2bbox(
                points, dist, (self.cfg.img_size, self.cfg.img_size)
            )  # [B, H, W, 4]
            decoded_bboxes.append(bboxes.flatten(1, 2))  # [B, H*W, 4]

        bboxes = torch.cat(decoded_bboxes, dim=1)  # [B, N_total, 4]
        cls = torch.cat([torch.sigmoid(s).flatten(2) for s in cls_scores], dim=2).permute(0, 2, 1)

        return bboxes, torch.zeros(0), torch.zeros(0), cls

    @overload
    def __call__(
        self, image_input: str, return_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def __call__(
        self, image_input: Tensor, return_logits: bool = False
    ) -> tuple[list[Tensor], list[Tensor]] | tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def __call__(
        self, image_input: str | Tensor, return_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[list[Tensor], list[Tensor]] | tuple[Tensor, Tensor, Tensor, Tensor]:
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
        cls_scores, bbox_preds = self.head(feats)

        bboxes, scores, class_ids = self._postprocess(cls_scores, bbox_preds)

        # Transform from padded target -> original image coordinates
        bboxes = bboxes.clone()
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_x) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_y) / scale
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, orig_w)
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, orig_h)

        return bboxes, scores, class_ids

    def _postprocess(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device

        decoded_bboxes = []
        cls_preds_list = []

        for level_idx, (bbox_pred, cls_score) in enumerate(zip(bbox_preds, cls_scores)):
            stride = self.cfg.prior_strides[level_idx]

            # Grid centers for this level
            cx, cy = self._grid_centers[level_idx]
            cx, cy = cx.to(device), cy.to(device)

            # Decode raw distances: clamp(≥0) * stride -> pixel distances [B, 4, H, W]
            distances = torch.clamp(bbox_pred, min=0) * stride

            # Build points [1, H, W, 2] and distances [B, H, W, 4]
            points = torch.stack([cx, cy], dim=-1).unsqueeze(0)
            dist = distances.permute(0, 2, 3, 1)

            # Convert to corner-format bboxes [B, H, W, 4]
            bboxes = distance2bbox(points, dist, (self.cfg.img_size, self.cfg.img_size))
            decoded_bboxes.append(bboxes.flatten(1, 2))

            # Classification: sigmoid -> [B, num_classes, H*W]
            cls_preds_list.append(torch.sigmoid(cls_score).flatten(2))

        # Concatenate across levels: [B, N_total, 4] and [B, num_classes, N_total]
        bboxes = torch.cat(decoded_bboxes, dim=1)
        scores = torch.cat(cls_preds_list, dim=2)

        # Single-image inference: take batch index 0
        bboxes = bboxes[0]  # [N, 4]
        scores = scores[0]  # [num_classes, N]

        # Get max score and class per bbox
        scores, class_ids = scores.max(dim=0)

        # Filter by score threshold
        keep = scores > self.cfg.score_threshold
        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # NMS
        if bboxes.numel() > 0:
            keep = nms(bboxes, scores, self.cfg.nms_iou_threshold)
            keep = keep[: self.cfg.max_num_detections]
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        return bboxes, scores, class_ids

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

        for i, (bbox, score, cls) in enumerate(zip(bboxes, scores, classes)):
            x1, y1, x2, y2 = bbox.tolist()
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            color = colors[cls.item() % len(colors)]
            draw.rectangle(
                [int(x1), int(y1), int(x2), int(y2)], outline=color, width=2
            )
            draw.text(
                (int(x1) + 4, int(y1) + 4),
                f"cls {cls.item()} {score:.2f}",
                fill=color,
            )

        return img

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
