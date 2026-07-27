import torch

from rtmdet import RotRTMDet, RotRTMDetConfig
from rtmdet.backbone import CSPNext
from rtmdet.neck import CSPNeXtPAFPN
from rtmdet.rot_head import RotRTMDetHead


class TestRotRTMDetHead:
    def _neck_feats(self, cfg):
        backbone = CSPNext(cfg)
        neck = CSPNeXtPAFPN(cfg)
        return neck(backbone(torch.randn(1, 3, cfg.img_size, cfg.img_size)))

    def test_returns_three_lists(self, rot_tiny_model):
        feats = rot_tiny_model.backbone(torch.randn(1, 3, 640, 640))
        feats = rot_tiny_model.neck(feats)
        result = rot_tiny_model.head(feats)
        assert len(result) == 3
        cls_scores, bbox_preds, angle_preds = result
        assert len(cls_scores) == rot_tiny_model.cfg.head_num_levels
        assert len(bbox_preds) == rot_tiny_model.cfg.head_num_levels
        assert len(angle_preds) == rot_tiny_model.cfg.head_num_levels

    def test_cls_score_shape(self, rot_tiny_config):
        from rtmdet import RotRTMDet
        model = RotRTMDet(rot_tiny_config)
        feats = self._neck_feats(model.cfg)
        cls_scores, _, _ = model.head(feats)
        for s in cls_scores:
            assert s.shape[1] == model.cfg.num_classes

    def test_bbox_pred_shape(self, rot_tiny_model):
        feats = self._neck_feats(rot_tiny_model.cfg)
        _, bbox_preds, _ = rot_tiny_model.head(feats)
        for b in bbox_preds:
            assert b.shape[1] == 4

    def test_angle_pred_shape(self, rot_tiny_model):
        feats = self._neck_feats(rot_tiny_model.cfg)
        _, _, angle_preds = rot_tiny_model.head(feats)
        for a in angle_preds:
            assert a.shape[1] == 1

    def test_preserves_spatial_dims(self, rot_tiny_model):
        feats = rot_tiny_model.backbone(torch.randn(1, 3, 640, 640))
        feats = rot_tiny_model.neck(feats)
        cls_scores, bbox_preds, angle_preds = rot_tiny_model.head(feats)
        for i, f in enumerate(feats):
            assert cls_scores[i].shape[2:] == f.shape[2:]
            assert bbox_preds[i].shape[2:] == f.shape[2:]
            assert angle_preds[i].shape[2:] == f.shape[2:]

    def test_batch_passthrough(self, rot_tiny_model):
        backbone = CSPNext(rot_tiny_model.cfg)
        neck = CSPNeXtPAFPN(rot_tiny_model.cfg)
        feats = neck(backbone(torch.randn(3, 3, 640, 640)))
        cls_scores, bbox_preds, angle_preds = rot_tiny_model.head(feats)
        for s in cls_scores:
            assert s.shape[0] == 3
        for b in bbox_preds:
            assert b.shape[0] == 3
        for a in angle_preds:
            assert a.shape[0] == 3

    def test_custom_num_classes(self):
        cfg = RotRTMDetConfig.from_preset("tiny")
        cfg.num_classes = 10
        model = RotRTMDet(cfg)
        feats = self._neck_feats(model.cfg)
        cls_scores, _, _ = model.head(feats)
        for s in cls_scores:
            assert s.shape[1] == 10

    def test_all_presets(self, rot_all_configs):
        for cfg in rot_all_configs:
            head = RotRTMDetHead(cfg)
            feats = self._neck_feats(cfg)
            cls_scores, bbox_preds, angle_preds = head(feats)
            assert len(cls_scores) == cfg.head_num_levels
            assert len(bbox_preds) == cfg.head_num_levels
            assert len(angle_preds) == cfg.head_num_levels

    def test_stacked_convs_count(self, rot_tiny_model):
        for i in range(rot_tiny_model.cfg.head_num_levels):
            assert len(rot_tiny_model.head.cls_convs[i]) == rot_tiny_model.cfg.head_num_stacked_convs
            assert len(rot_tiny_model.head.reg_convs[i]) == rot_tiny_model.cfg.head_num_stacked_convs
            assert len(rot_tiny_model.head.ang_convs[i]) == 1