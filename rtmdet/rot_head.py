from torch import Tensor, nn

from rtmdet.config import RotRTMDetConfig
from rtmdet.layers import ConvModule
from rtmdet.utils import apply_factor


class RotRTMDetHead(nn.Module):
    def __init__(self, cfg: RotRTMDetConfig):
        super().__init__()
        c = apply_factor(cfg.neck_out_channels, cfg.widen_factor)

        # Per-level towers
        cls_convs = []
        reg_convs = []
        ang_convs = []

        # Per-level prediction heads
        rtm_cls = []
        rtm_reg = []
        rtm_ang = []

        for _ in range(cfg.head_num_levels):
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
            # Angle tower: single ConvModule per level (matches mmrotate checkpoint)
            ang_tower = nn.ModuleList(
                [
                    ConvModule(c_in=c, c_out=c, kernel_size=3, stride=1, padding=1)
                ]
            )

            cls_convs.append(cls_tower)
            reg_convs.append(reg_tower)
            ang_convs.append(ang_tower)

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
            rtm_ang.append(
                nn.Conv2d(
                    in_channels=c, out_channels=1, kernel_size=1, stride=1, padding=0
                )
            )

        self.cls_convs: nn.ModuleList = nn.ModuleList(cls_convs)
        self.reg_convs: nn.ModuleList = nn.ModuleList(reg_convs)
        self.ang_convs: nn.ModuleList = nn.ModuleList(ang_convs)
        self.rtm_cls: nn.ModuleList = nn.ModuleList(rtm_cls)
        self.rtm_reg: nn.ModuleList = nn.ModuleList(rtm_reg)
        self.rtm_ang: nn.ModuleList = nn.ModuleList(rtm_ang)

    def forward(self, x: tuple[Tensor, ...]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        cls_scores, bbox_preds, angle_preds = [], [], []

        for i, feat in enumerate(x):
            # classification path
            cls_feat = feat
            for layer in self.cls_convs[i]:  # type: ignore
                cls_feat = layer(cls_feat)
            cls_scores.append(self.rtm_cls[i](cls_feat))

            # regression path
            reg_feat = feat
            for layer in self.reg_convs[i]:  # type: ignore
                reg_feat = layer(reg_feat)
            bbox_preds.append(self.rtm_reg[i](reg_feat))

            # angle path
            ang_feat = feat
            for layer in self.ang_convs[i]:  # type: ignore
                ang_feat = layer(ang_feat)
            angle_preds.append(self.rtm_ang[i](ang_feat))

        return cls_scores, bbox_preds, angle_preds