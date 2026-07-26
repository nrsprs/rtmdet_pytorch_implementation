from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from rtmdet import RTMDet, RTMDetConfig


class TestFromPreset:
    def test_from_preset_tiny(self):
        model = RTMDet.from_preset("tiny", pretrained=False)
        assert isinstance(model, RTMDet)
        assert model.cfg.preset_name == "tiny"

    def test_from_preset_small(self):
        model = RTMDet.from_preset("small", pretrained=False)
        assert isinstance(model, RTMDet)

    def test_from_preset_medium(self):
        model = RTMDet.from_preset("medium", pretrained=False)
        assert isinstance(model, RTMDet)

    def test_from_preset_large(self):
        model = RTMDet.from_preset("large", pretrained=False)
        assert isinstance(model, RTMDet)

    def test_img_size_override(self):
        model = RTMDet.from_preset("tiny", img_size=256, pretrained=False)
        assert model.cfg.img_size == 256

    def test_num_classes_override(self):
        model = RTMDet.from_preset("tiny", num_classes=10, pretrained=False)
        assert model.cfg.num_classes == 10

    def test_both_overrides(self):
        model = RTMDet.from_preset("tiny", img_size=128, num_classes=5, pretrained=False)
        assert model.cfg.img_size == 128
        assert model.cfg.num_classes == 5

    def test_device_placement(self):
        model = RTMDet.from_preset("tiny", pretrained=False)
        # from_preset uses _default_device which may be cuda/mps/cpu
        assert model.device is not None


class TestForward:
    def test_return_logits_false(self, tiny_model, tmp_image_path):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        result = tiny_model.forward(x)
        cls_scores, bbox_preds = result
        assert isinstance(cls_scores, list)
        assert isinstance(bbox_preds, list)
        assert len(cls_scores) == 3
        assert len(bbox_preds) == 3

    def test_return_logits_true(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        result = tiny_model.forward(x, return_logits=True)
        bboxes, _pad_w, _pad_h, cls = result
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(cls, torch.Tensor)

    def test_logits_bboxes_shape(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(2, 3, 640, 640)
        bboxes, _, _, _ = tiny_model.forward(x, return_logits=True)
        assert bboxes.shape[0] == 2
        assert bboxes.shape[2] == 4

    def test_logits_cls_shape(self, tiny_config):
        model = RTMDet(tiny_config).to("cpu")
        model.eval()
        x = torch.randn(2, 3, 640, 640)
        _, _, _, cls = model.forward(x, return_logits=True)  # type: ignore
        assert cls.shape[0] == 2
        assert cls.shape[2] == model.cfg.num_classes

    def test_exp_on_reg_true(self, medium_model):
        medium_model.eval()
        x = torch.randn(1, 3, medium_model.cfg.img_size, medium_model.cfg.img_size)
        bboxes, _, _, _cls = medium_model.forward(x, return_logits=True)
        assert bboxes.shape[2] == 4

    def test_exp_on_reg_false(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _, _cls = tiny_model.forward(x, return_logits=True)
        assert bboxes.shape[2] == 4

    def test_batch_size_1(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        cls_scores, _bbox_preds = tiny_model.forward(x)
        assert cls_scores[0].shape[0] == 1

    def test_batch_size_4(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(4, 3, 640, 640)
        cls_scores, _bbox_preds = tiny_model.forward(x)
        assert cls_scores[0].shape[0] == 4


class TestCall:
    def test_call_with_tensor(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        result = tiny_model(x)
        cls_scores, _bbox_preds = result
        assert isinstance(cls_scores, list)

    def test_call_with_tensor_return_logits(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 640, 640)
        bboxes, _, _, _cls = tiny_model(x, return_logits=True)
        assert isinstance(bboxes, torch.Tensor)

    def test_call_with_string_path(self, tiny_model, tmp_image_path):
        tiny_model.eval()
        bboxes, scores, classes = tiny_model(str(tmp_image_path))
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(classes, torch.Tensor)

    def test_call_string_returns_filtered(self, tiny_model, tmp_image_path):
        tiny_model.eval()
        bboxes, scores, classes = tiny_model(str(tmp_image_path))
        assert bboxes.dim() == 2
        assert bboxes.shape[1] == 4
        assert scores.dim() == 1
        assert classes.dim() == 1


class TestPredict:
    def test_predict_returns_3_tuple(self, tiny_model, tmp_image_path):
        bboxes, scores, classes = tiny_model.predict(str(tmp_image_path))
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(scores, torch.Tensor)
        assert isinstance(classes, torch.Tensor)

    def test_bboxes_format(self, tiny_model, tmp_image_path):
        bboxes, _scores, _classes = tiny_model.predict(str(tmp_image_path))
        if bboxes.numel() > 0:
            assert (bboxes[:, 0] < bboxes[:, 2]).all()
            assert (bboxes[:, 1] < bboxes[:, 3]).all()

    def test_scores_in_range(self, tiny_model, tmp_image_path):
        _bboxes, scores, _classes = tiny_model.predict(str(tmp_image_path))
        if scores.numel() > 0:
            assert scores.min() >= 0
            assert scores.max() <= 1

    def test_classes_non_negative(self, tiny_model, tmp_image_path):
        _bboxes, _scores, classes = tiny_model.predict(str(tmp_image_path))
        if classes.numel() > 0:
            assert classes.min() >= 0


class TestSaveLoad:
    def test_to_file_creates_file(self, tiny_model, tmp_checkpoint_path):
        tiny_model.to_file(str(tmp_checkpoint_path))
        assert Path(tmp_checkpoint_path).exists()

    def test_from_file_loads_model(self, tiny_model, tmp_checkpoint_path):
        tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RTMDet.from_file(str(tmp_checkpoint_path))
        assert isinstance(model2, RTMDet)

    def test_roundtrip_weights(self, tiny_model, tmp_checkpoint_path):
        tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RTMDet.from_file(str(tmp_checkpoint_path))
        sd1 = {k: v.cpu() for k, v in tiny_model.state_dict().items()}
        sd2 = {k: v.cpu() for k, v in model2.state_dict().items()}
        for k in sd1:  # noqa: PLC0206
            assert torch.allclose(sd1[k], sd2[k])

    def test_roundtrip_config(self, tiny_model, tmp_checkpoint_path):
        tiny_model.to_file(str(tmp_checkpoint_path))
        model2 = RTMDet.from_file(str(tmp_checkpoint_path))
        assert model2.cfg.num_classes == tiny_model.cfg.num_classes
        assert model2.cfg.img_size == tiny_model.cfg.img_size

    def test_to_file_requires_preset_name(self):
        cfg = RTMDetConfig(deepen_factor=0.5, widen_factor=0.5)
        model = RTMDet(cfg)
        with pytest.raises(AssertionError):
            model.to_file("dummy.pt")

    def test_roundtrip_custom_config(self, tmp_checkpoint_path):
        cfg = RTMDetConfig.from_preset("tiny")
        cfg.num_classes = 10
        cfg.img_size = 256
        model = RTMDet(cfg).to("cpu")
        model.to_file(str(tmp_checkpoint_path))
        model2 = RTMDet.from_file(str(tmp_checkpoint_path))
        assert model2.cfg.num_classes == 10
        assert model2.cfg.img_size == 256


class TestEdgeCases:
    def test_forward_non_square_tensor(self, tiny_model):
        tiny_model.eval()
        x = torch.randn(1, 3, 480, 640)
        cls_scores, _ = tiny_model.forward(x)
        assert len(cls_scores) == 3

    def test_predict_non_square_image(self, tiny_model, tmp_path):
        img = Image.new("RGB", (800, 600), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 350], fill=(255, 0, 0))
        path = tmp_path / "nonsquare.png"
        img.save(str(path))
        bboxes, _, _ = tiny_model.predict(str(path))
        if bboxes.numel() > 0:
            assert bboxes[:, 0].min() >= 0
            assert bboxes[:, 2].max() <= 800
            assert bboxes[:, 1].min() >= 0
            assert bboxes[:, 3].max() <= 600

    def test_high_threshold_filters_all(self, tiny_model, tmp_image_path):
        tiny_model.cfg.score_threshold = 1.0
        bboxes, _, _ = tiny_model.predict(str(tmp_image_path))
        assert bboxes.numel() == 0

    def test_max_num_detections_caps(self, tiny_model, tmp_image_path):
        tiny_model.cfg.score_threshold = 0.001
        tiny_model.cfg.max_num_detections = 5
        bboxes, _, _ = tiny_model.predict(str(tmp_image_path))
        assert len(bboxes) <= 5


class TestDrawDetections:
    def _fake_detections(self):
        bboxes = torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]])
        scores = torch.tensor([0.8, 0.6])
        classes = torch.tensor([0, 1])
        return bboxes, scores, classes

    def test_draw_from_path(self, tiny_model, tmp_image_path):
        bboxes, scores, classes = self._fake_detections()
        img = tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_from_tensor(self, tiny_model):
        tensor = torch.randn(640, 640, 3)  # channel-last for PIL
        bboxes, scores, classes = self._fake_detections()
        img = tiny_model.draw_detections(tensor, bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_empty_bboxes(self, tiny_model, tmp_image_path):
        bboxes = torch.empty((0, 4))
        scores = torch.empty(0)
        classes = torch.empty(0, dtype=torch.long)
        img = tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert isinstance(img, Image.Image)

    def test_draw_image_size(self, tiny_model, tmp_image_path):
        bboxes, scores, classes = self._fake_detections()
        img = tiny_model.draw_detections(str(tmp_image_path), bboxes, scores, classes)
        assert img.size == (640, 640)


class TestDevice:
    def test_device_property_cpu(self, tiny_model):
        assert tiny_model.device == torch.device("cpu")
