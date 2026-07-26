import torch

from rtmdet.head import RTMDetHead
from rtmdet.neck import CSPNeXtPAFPN


class TestRTMDetHead:
    def _neck_feats(self, cfg):
        from rtmdet.backbone import CSPNext
        backbone = CSPNext(cfg)
        neck = CSPNeXtPAFPN(cfg)
        return neck(backbone(torch.randn(1, 3, cfg.img_size, cfg.img_size)))

    def test_output_lists_length(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(1, 3, 640, 640))
        feats = tiny_model.neck(feats)
        cls_scores, bbox_preds = tiny_model.head(feats)
        assert len(cls_scores) == tiny_model.cfg.head_num_levels
        assert len(bbox_preds) == tiny_model.cfg.head_num_levels

    def test_cls_score_shape(self, tiny_config):
        from rtmdet import RTMDet
        model = RTMDet(tiny_config)
        feats = self._neck_feats(model.cfg)
        cls_scores, _ = model.head(feats)
        for s in cls_scores:
            assert s.shape[1] == model.cfg.num_classes

    def test_bbox_pred_shape(self, tiny_model):
        feats = self._neck_feats(tiny_model.cfg)
        _, bbox_preds = tiny_model.head(feats)
        for b in bbox_preds:
            assert b.shape[1] == 4

    def test_preserves_spatial_dims(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(1, 3, 640, 640))
        feats = tiny_model.neck(feats)
        cls_scores, bbox_preds = tiny_model.head(feats)
        for i, f in enumerate(feats):
            assert cls_scores[i].shape[2:] == f.shape[2:]
            assert bbox_preds[i].shape[2:] == f.shape[2:]

    def test_batch_passthrough(self, tiny_model):
        feats = self._neck_feats(tiny_model.cfg)
        # Re-create with batch=3
        from rtmdet.backbone import CSPNext
        backbone = CSPNext(tiny_model.cfg)
        neck = CSPNeXtPAFPN(tiny_model.cfg)
        feats = neck(backbone(torch.randn(3, 3, 640, 640)))
        cls_scores, _bbox_preds = tiny_model.head(feats)
        for s in cls_scores:
            assert s.shape[0] == 3

    def test_custom_num_classes(self, custom_model):
        feats = self._neck_feats(custom_model.cfg)
        cls_scores, _ = custom_model.head(feats)
        for s in cls_scores:
            assert s.shape[1] == custom_model.cfg.num_classes

    def test_all_presets(self, all_configs):
        for cfg in all_configs:
            head = RTMDetHead(cfg)
            feats = self._neck_feats(cfg)
            cls_scores, _bbox_preds = head(feats)
            assert len(cls_scores) == cfg.head_num_levels

    def test_stacked_convs_count(self, tiny_model):
        for i in range(tiny_model.cfg.head_num_levels):
            assert len(tiny_model.head.cls_convs[i]) == tiny_model.cfg.head_num_stacked_convs
            assert len(tiny_model.head.reg_convs[i]) == tiny_model.cfg.head_num_stacked_convs