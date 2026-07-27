import math
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from rtmdet import RotRTMDet, RotRTMDetConfig


class TestFromPreset:
    def test_from_preset_tiny(self):
        model = RotRTMDet.from_preset("tiny", pretrained=False)
        assert isinstance(model, RotRTMDet)
        assert model.cfg.preset_name == "tiny"

    def test_from_preset_small(self):
        model = RotRTMDet.from_preset("small", pretrained=False)
        assert isinstance(model, RotRTMDet)

    def test_from_preset_medium(self):
        model = RotRTMDet.from_preset("medium", pretrained=False)
        assert isinstance(model, RotRTMDet)

    def test_from_preset_large(self):
        model = RotRTMDet.from_preset("large", pretrained=False)
        assert isinstance(model, RotRTMDet)

    def test_img_size_override(self):
        model = RotRTMDet.from_preset("tiny", img_size=256, pretrained=False)
        assert model.cfg.img_size == 256

    def test_num_classes_override(self):
        model = RotRTMDet.from_preset("tiny", num_classes=10, pretrained=False)
        assert model.cfg.num_classes == 10

    def test_both_overrides(self):
        model = RotRTMDet.from_preset("tiny", img_size=128, num_classes=5, pretrained=False)
        assert model.cfg.img_size == 128
        assert model.cfg.num_classes == 5

    def test_dota_default_classes(self):
        model = RotRTMDet.from_preset("tiny", pretrained=False)
        assert model.cfg.num_classes == 15

    def test_nms_threshold_default(self):
        model = RotRTMDet.from_preset("tiny", pretrained=False)
        assert model.cfg.nms_iou_threshold == 0.1

    def test_device_placement(self):
        model = RotRTMDet.from_preset("tiny", pretrained=False)
        assert model.device is not None


class TestForward:
    def test_returns_three_lists(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        result = rot_tiny_model.forward(x)
        cls_scores, bbox_preds, angle_preds = result
        assert isinstance(cls_scores, list)
        assert isinstance(bbox_preds, list)
        assert isinstance(angle_preds, list)
        assert len(cls_scores) == 3
        assert len(bbox_preds) == 3
        assert len(angle_preds) == 3

    def test_return_logits_true(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _pad, cls = rot_tiny_model.forward(x, return_logits=True)
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(cls, torch.Tensor)

    def test_logits_bboxes_shape(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(2, 3, 640, 640)
        bboxes, _, _ = rot_tiny_model.forward(x, return_logits=True)
        assert bboxes.shape[0] == 2
        assert bboxes.shape[2] == 5  # [cx, cy, w, h, theta]

    def test_bboxes_have_5_channels(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _ = rot_tiny_model.forward(x, return_logits=True)
        assert bboxes.shape[-1] == 5

    def test_angle_in_range(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _ = rot_tiny_model.forward(x, return_logits=True)
        theta = bboxes[:, :, 4]
        assert (theta >= -math.pi / 2).all()
        assert (theta < math.pi / 2).all()

    def test_width_ge_height(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _ = rot_tiny_model.forward(x, return_logits=True)
        assert (bboxes[:, :, 2] >= bboxes[:, :, 3]).all()

    def test_batch_size_1(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        cls_scores, _, _ = rot_tiny_model.forward(x)
        assert cls_scores[0].shape[0] == 1

    def test_batch_size_4(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(4, 3, 640, 640)
        cls_scores, _, _ = rot_tiny_model.forward(x)
        assert cls_scores[0].shape[0] == 4


class TestCall:
    def test_call_with_tensor(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        result = rot_tiny_model(x)
        cls_scores, bbox_preds, angle_preds = result
        assert isinstance(cls_scores, list)

    def test_call_with_tensor_return_logits(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _ = rot_tiny_model(x, return_logits=True)
        assert isinstance(bboxes, torch.Tensor)
        assert bboxes.shape[-1] == 5

    def test_call_with_string_path(self, rot_tiny_model, tmp_image_path):
        rot_tiny_model.eval()
        bboxes, scores, classes = rot_tiny_model(str(tmp_image_path))
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(classes, torch.Tensor)

    def test_call_string_returns_obb_format(self, rot_tiny_model, tmp_image_path):
        rot_tiny_model.eval()
        bboxes, scores, classes = rot_tiny_model(str(tmp_image_path))
        assert bboxes.dim() == 2
        assert bboxes.shape[1] == 5
        assert scores.dim() == 1
        assert classes.dim() == 1


class TestPredict:
    def test_predict_returns_3_tuple(self, rot_tiny_model, tmp_image_path):
        bboxes, scores, classes = rot_tiny_model.predict(str(tmp_image_path))
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(classes, torch.Tensor)

    def test_scores_in_range(self, rot_tiny_model, tmp_image_path):
        _bboxes, scores, _classes = rot_tiny_model.predict(str(tmp_image_path))
        if scores.numel() > 0:
            assert scores.min() >= 0
            assert scores.max() <= 1

    def test_classes_non_negative(self, rot_tiny_model, tmp_image_path):
        _bboxes, _scores, classes = rot_tiny_model.predict(str(tmp_image_path))
        if classes.numel() > 0:
            assert classes.min() >= 0

    def test_angle_preserved(self, rot_tiny_model, tmp_image_path):
        bboxes, _scores, _classes = rot_tiny_model.predict(str(tmp_image_path))
        if bboxes.numel() > 0:
            theta = bboxes[:, 4]
            assert (theta >= -math.pi / 2).all()
            assert (theta < math.pi / 2).all()

    def test_non_square_image(self, rot_tiny_model, tmp_path):
        img = Image.new("RGB", (800, 600), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 350], fill=(255, 0, 0))
        path = tmp_path / "nonsquare.png"
        img.save(str(path))
        bboxes, _, _ = rot_tiny_model.predict(str(path))
        if bboxes.numel() > 0:
            assert bboxes[:, 0].min() >= 0  # cx >= 0
            assert bboxes[:, 0].max() <= 800  # cx <= orig_w
            assert bboxes[:, 1].min() >= 0  # cy >= 0
            assert bboxes[:, 1].max() <= 600  # cy <= orig_h


class TestSaveLoad:
    def test_to_file_creates_file(self, rot_tiny_model, tmp_checkpoint_path):
        rot_tiny_model.to_file(str(tmp_checkpoint_path))
        assert Path(tmp_checkpoint_path).exists()

    def test_from_file_loads_model(self, rot_tiny_model, tmp_checkpoint_path):
        rot_tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RotRTMDet.from_file(str(tmp_checkpoint_path))
        assert isinstance(model2, RotRTMDet)

    def test_roundtrip_weights(self, rot_tiny_model, tmp_checkpoint_path):
        rot_tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RotRTMDet.from_file(str(tmp_checkpoint_path))
        sd1 = {k: v.cpu() for k, v in rot_tiny_model.state_dict().items()}
        sd2 = {k: v.cpu() for k, v in model2.state_dict().items()}
        for k in sd1:  # noqa: PLC0206
            assert torch.allclose(sd1[k], sd2[k])

    def test_roundtrip_config(self, rot_tiny_model, tmp_checkpoint_path):
        rot_tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RotRTMDet.from_file(str(tmp_checkpoint_path))
        assert model2.cfg.num_classes == rot_tiny_model.cfg.num_classes
        assert model2.cfg.img_size == rot_tiny_model.cfg.img_size

    def test_to_file_requires_preset_name(self):
        cfg = RotRTMDetConfig(
            deepen_factor=0.5,
            widen_factor=0.5,
            exp_on_reg=True,
        )
        model = RotRTMDet(cfg)
        with pytest.raises(AssertionError):
            model.to_file("dummy.pt")

    def test_roundtrip_custom_config(self, tmp_checkpoint_path):
        cfg = RotRTMDetConfig.from_preset("tiny")
        cfg.num_classes = 10
        cfg.img_size = 256
        model = RotRTMDet(cfg).to("cpu")
        model.to_file(str(tmp_checkpoint_path))
        model2 = RotRTMDet.from_file(str(tmp_checkpoint_path))
        assert model2.cfg.num_classes == 10
        assert model2.cfg.img_size == 256


class TestEdgeCases:
    def test_forward_non_square_tensor(self, rot_tiny_model):
        rot_tiny_model.eval()
        x = torch.randn(1, 3, 480, 640)
        cls_scores, bbox_preds, angle_preds = rot_tiny_model.forward(x)
        assert len(cls_scores) == 3
        assert len(bbox_preds) == 3
        assert len(angle_preds) == 3

    def test_predict_non_square_image(self, rot_tiny_model, tmp_path):
        img = Image.new("RGB", (800, 600), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 350], fill=(255, 0, 0))
        path = tmp_path / "nonsquare.png"
        img.save(str(path))
        bboxes, _, _ = rot_tiny_model.predict(str(path))
        if bboxes.numel() > 0:
            assert bboxes[:, 0].min() >= 0
            assert bboxes[:, 0].max() <= 800
            assert bboxes[:, 1].min() >= 0
            assert bboxes[:, 1].max() <= 600

    def test_high_threshold_filters_all(self, rot_tiny_model, tmp_image_path):
        rot_tiny_model.cfg.score_threshold = 1.0
        bboxes, _, _ = rot_tiny_model.predict(str(tmp_image_path))
        assert bboxes.numel() == 0

    def test_max_num_detections_caps(self, rot_tiny_model, tmp_image_path):
        rot_tiny_model.cfg.score_threshold = 0.001
        rot_tiny_model.cfg.max_num_detections = 5
        bboxes, _, _ = rot_tiny_model.predict(str(tmp_image_path))
        assert len(bboxes) <= 5


class TestDrawDetections:
    def _fake_rotated_detections(self):
        bboxes = torch.tensor([
            [150.0, 150.0, 60.0, 40.0, 0.5],
            [350.0, 350.0, 50.0, 30.0, -0.3],
        ])
        scores = torch.tensor([0.8, 0.6])
        classes = torch.tensor([0, 1])
        return bboxes, scores, classes

    def test_draw_from_path(self, rot_tiny_model, tmp_image_path):
        bboxes, scores, classes = self._fake_rotated_detections()
        img = rot_tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_from_tensor(self, rot_tiny_model):
        tensor = torch.randn(640, 640, 3)
        bboxes, scores, classes = self._fake_rotated_detections()
        img = rot_tiny_model.draw_detections(tensor, bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_empty_bboxes(self, rot_tiny_model, tmp_image_path):
        bboxes = torch.empty((0, 5))
        scores = torch.empty(0)
        classes = torch.empty(0, dtype=torch.long)
        img = rot_tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_image_size(self, rot_tiny_model, tmp_image_path):
        bboxes, scores, classes = self._fake_rotated_detections()
        img = rot_tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert img.size == (640, 640)


class TestDevice:
    def test_device_property_cpu(self, rot_tiny_model):
        assert rot_tiny_model.device == torch.device("cpu")