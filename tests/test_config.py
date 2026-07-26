import pytest

from rtmdet.config import RTMDetConfig


class TestFromPreset:
    def test_from_preset_tiny(self):
        cfg = RTMDetConfig.from_preset("tiny")
        assert cfg.deepen_factor == pytest.approx(0.167)
        assert cfg.widen_factor == pytest.approx(0.375)
        assert cfg.exp_on_reg is False
        assert cfg.preset_name == "tiny"

    def test_from_preset_small(self):
        cfg = RTMDetConfig.from_preset("small")
        assert cfg.deepen_factor == pytest.approx(0.33)
        assert cfg.widen_factor == pytest.approx(0.5)
        assert cfg.exp_on_reg is False
        assert cfg.preset_name == "small"

    def test_from_preset_medium(self):
        cfg = RTMDetConfig.from_preset("medium")
        assert cfg.deepen_factor == pytest.approx(0.67)
        assert cfg.widen_factor == pytest.approx(0.75)
        assert cfg.exp_on_reg is True
        assert cfg.preset_name == "medium"

    def test_from_preset_large(self):
        cfg = RTMDetConfig.from_preset("large")
        assert cfg.deepen_factor == pytest.approx(1.0)
        assert cfg.widen_factor == pytest.approx(1.0)
        assert cfg.exp_on_reg is True
        assert cfg.preset_name == "large"

    @pytest.mark.parametrize("cfg", ["tiny", "small", "medium", "large"])
    def test_all_presets_defaults(self, cfg):
        cfg = RTMDetConfig.from_preset(cfg)
        assert cfg.num_classes == 80
        assert cfg.img_size == 640
        assert cfg.prior_strides == [8, 16, 32]
        assert cfg.score_threshold == 0.001
        assert cfg.nms_iou_threshold == 0.65
        assert cfg.max_num_detections == 300
        assert cfg.head_num_levels == 3
        assert cfg.head_num_stacked_convs == 2
        assert cfg.neck_out_channels == 256


class TestValidation:
    def test_deepen_factor_must_be_positive(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=0, widen_factor=1.0)
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=-1, widen_factor=1.0)

    def test_widen_factor_must_be_positive(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=0)
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=-1)

    def test_num_classes_must_be_positive(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, num_classes=0)

    def test_img_size_must_be_positive(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, img_size=0)

    def test_neck_out_channels_must_be_positive(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, neck_out_channels=0)

    def test_score_threshold_range(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, score_threshold=-0.1)
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, score_threshold=1.1)

    def test_nms_iou_threshold_range(self):
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, nms_iou_threshold=-0.1)
        with pytest.raises(Exception) as _:
            RTMDetConfig(deepen_factor=1.0, widen_factor=1.0, nms_iou_threshold=1.1)


class TestManualConstruction:
    def test_custom_config_manual(self):
        cfg = RTMDetConfig(deepen_factor=0.5, widen_factor=0.75, num_classes=10)
        assert cfg.deepen_factor == 0.5
        assert cfg.widen_factor == 0.75
        assert cfg.num_classes == 10
        assert cfg.preset_name == ""

    def test_preset_override_num_classes(self):
        cfg = RTMDetConfig.from_preset("tiny")
        cfg.num_classes = 5
        assert cfg.num_classes == 5

    def test_preset_override_img_size(self):
        cfg = RTMDetConfig.from_preset("tiny")
        cfg.img_size = 256
        assert cfg.img_size == 256
