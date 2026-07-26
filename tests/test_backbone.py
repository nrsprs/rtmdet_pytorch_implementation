import torch

from rtmdet.backbone import CSPNext


class TestCSPNext:
    def test_output_count(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        outputs = tiny_model.backbone(x)
        assert len(outputs) == 3

    def test_stride8_shape_tiny(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        p3, _, _ = tiny_model.backbone(x)
        assert p3.shape == (1, 96, 80, 80)

    def test_stride16_shape_tiny(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        _, p4, _ = tiny_model.backbone(x)
        assert p4.shape == (1, 192, 40, 40)

    def test_stride32_shape_tiny(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        _, _, p5 = tiny_model.backbone(x)
        assert p5.shape == (1, 384, 20, 20)

    def test_channel_progression(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = tiny_model.backbone(x)
        assert p4.shape[1] == 2 * p3.shape[1]
        assert p5.shape[1] == 2 * p4.shape[1]

    def test_batch_size_1(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        outputs = tiny_model.backbone(x)
        for out in outputs:
            assert out.shape[0] == 1

    def test_batch_size_4(self, tiny_model):
        x = torch.randn(4, 3, 640, 640)
        outputs = tiny_model.backbone(x)
        for out in outputs:
            assert out.shape[0] == 4

    def test_all_presets(self, all_configs):
        for cfg in all_configs:
            backbone = CSPNext(cfg)
            x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
            outputs = backbone(x)
            assert len(outputs) == 3

    def test_custom_img_size(self, custom_config):
        backbone = CSPNext(custom_config)
        x = torch.randn(1, 3, 256, 256)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2:] == (32, 32)
        assert p4.shape[2:] == (16, 16)
        assert p5.shape[2:] == (8, 8)

    def test_stem_output_shape(self, tiny_model):
        x = torch.randn(1, 3, 640, 640)
        stem_out = tiny_model.backbone.stem(x)
        assert stem_out.shape[1] == 24
        assert stem_out.shape[2] == 320
        assert stem_out.shape[3] == 320