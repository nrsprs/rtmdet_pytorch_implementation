import torch

from rtmdet.backbone import CSPNext
from rtmdet.neck import CSPNeXtPAFPN
from rtmdet.utils import apply_factor


class TestCSPNeXtPAFPN:
    def test_output_count(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(1, 3, 640, 640))
        outputs = tiny_model.neck(feats)
        assert len(outputs) == 3

    def test_uniform_channels(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(1, 3, 640, 640))
        outputs = tiny_model.neck(feats)
        ch = outputs[0].shape[1]
        for out in outputs[1:]:
            assert out.shape[1] == ch

    def test_preserves_spatial_dims(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(1, 3, 640, 640))
        outputs = tiny_model.neck(feats)
        assert outputs[0].shape[2:] == feats[0].shape[2:]
        assert outputs[1].shape[2:] == feats[1].shape[2:]
        assert outputs[2].shape[2:] == feats[2].shape[2:]

    def test_tiny_output_channels(self, tiny_config):
        neck = CSPNeXtPAFPN(tiny_config)
        ch = apply_factor(tiny_config.neck_out_channels, tiny_config.widen_factor)
        backbone = CSPNext(tiny_config)
        feats = neck(backbone(torch.randn(1, 3, tiny_config.img_size, tiny_config.img_size)))
        for out in feats:
            assert out.shape[1] == ch

    def test_batch_passthrough(self, tiny_model):
        feats = tiny_model.backbone(torch.randn(2, 3, 640, 640))
        outputs = tiny_model.neck(feats)
        for out in outputs:
            assert out.shape[0] == 2

    def test_all_presets(self, all_configs):
        for cfg in all_configs:
            backbone = CSPNext(cfg)
            neck = CSPNeXtPAFPN(cfg)
            feats = backbone(torch.randn(1, 3, cfg.img_size, cfg.img_size))
            outputs = neck(feats)
            assert len(outputs) == 3

    def test_custom_config(self, custom_config):
        backbone = CSPNext(custom_config)
        neck = CSPNeXtPAFPN(custom_config)
        feats = backbone(torch.randn(1, 3, custom_config.img_size, custom_config.img_size))
        outputs = neck(feats)
        assert len(outputs) == 3