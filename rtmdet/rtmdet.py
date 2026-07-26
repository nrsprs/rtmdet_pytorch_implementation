from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch import Tensor
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


class RTMDet(nn.Module):
    def __init__(self, cfg: RTMDetConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CSPNext(cfg)
        self.neck = CSPNeXtPAFPN(cfg)
        self.head = RTMDetHead(cfg)

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
    ) -> Union[
        Tuple[List[Tensor], List[Tensor]],
        Tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_scores, bbox_preds = self.head(feats)

        if not return_logits:
            return cls_scores, bbox_preds

        cls = torch.cat([s.flatten(2) for s in cls_scores], dim=2).permute(0, 2, 1)
        bboxes = torch.cat(
            [p.flatten(2) for p in bbox_preds], dim=2
        ).permute(0, 2, 1)
        bboxes = torch.sigmoid(bboxes) * self.cfg.img_size
        return bboxes, torch.zeros(0), torch.zeros(0), cls

    def __call__(
        self, image_input: Union[str, Tensor], return_logits: bool = False
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if isinstance(image_input, str):
            return self._inference_from_path(image_input)

        return self.forward(image_input, return_logits=return_logits)

    def _inference_from_path(self, path: str) -> Tuple[Tensor, Tensor, Tensor]:
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
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        img_size = self.cfg.img_size

        bbox_preds_decoded = (
            torch.cat([torch.sigmoid(p).flatten(2) for p in bbox_preds], dim=2)
            * img_size
        )
        cls_preds = torch.cat([torch.sigmoid(s).flatten(2) for s in cls_scores], dim=2)

        bboxes = bbox_preds_decoded[0].permute(1, 0)
        scores = cls_preds[0]

        scores, class_ids = scores.max(dim=0)
        keep = scores > self.cfg.score_threshold
        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if bboxes.numel() > 0:
            keep = nms(bboxes, scores, self.cfg.nms_iou_threshold)
            keep = keep[: self.cfg.max_num_detections]
            bboxes = bboxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        return bboxes, scores, class_ids

    def draw_detections(
        self,
        image_input: Union[str, Tensor],
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
            x1, y1 = min(x1, x2), min(y1, y2)
            x2, y2 = max(x1, x2), max(y1, y2)
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