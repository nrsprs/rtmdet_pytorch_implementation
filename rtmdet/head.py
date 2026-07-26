from torch import Tensor, nn

from rtmdet.config import RTMDetConfig
from rtmdet.layers import ConvModule
from rtmdet.utils import apply_factor


class RTMDetHead(nn.Module):
    def __init__(self, cfg: RTMDetConfig):
        super().__init__()
        c = apply_factor(cfg.neck_out_channels, cfg.widen_factor)

        # Per-level towers
        cls_convs = []
        reg_convs = []

        # Per-level prediction heads
        rtm_cls = []
        rtm_reg = []

        for _ in range(cfg.head_num_levels):
            # each level has its own classification and regression towers
            cls_tower = nn.ModuleList(
                [
                    ConvModule(c_in=c, c_out=c, kernel_size=3, stride=1, padding=1)
                    for _ in range(cfg.head_num_stacked_convs)
                ]
            )
            reg_tower = nn.ModuleList(
                [
                    ConvModule(c_in=c, c_out=c, kernel_size=3, stride=1, padding=1)
                    for _ in range(cfg.head_num_stacked_convs)
                ]
            )

            cls_convs.append(cls_tower)
            reg_convs.append(reg_tower)

            rtm_cls.append(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=cfg.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            rtm_reg.append(
                nn.Conv2d(
                    in_channels=c, out_channels=4, kernel_size=1, stride=1, padding=0
                )
            )

        self.cls_convs: nn.ModuleList = nn.ModuleList(cls_convs)
        self.reg_convs: nn.ModuleList = nn.ModuleList(reg_convs)
        self.rtm_cls: nn.ModuleList = nn.ModuleList(rtm_cls)
        self.rtm_reg: nn.ModuleList = nn.ModuleList(rtm_reg)

    def forward(self, x: tuple[Tensor, ...]) -> tuple[list[Tensor], ...]:
        cls_scores, bbox_preds = [], []

        for i, feat in enumerate(x):
            # ---- classification path ----
            cls_feat = feat
            for layer in self.cls_convs[i]:  # type: ignore[attr-defined]
                cls_feat = layer(cls_feat)
            cls_scores.append(self.rtm_cls[i](cls_feat))

            # ---- regression path ----
            reg_feat = feat
            for layer in self.reg_convs[i]:  # type: ignore[attr-defined]
                reg_feat = layer(reg_feat)
            bbox_preds.append(self.rtm_reg[i](reg_feat))

        return cls_scores, bbox_preds
