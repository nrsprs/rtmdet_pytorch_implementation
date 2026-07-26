from pathlib import Path

import torch

from rtmdet.checkpoint_utils import (
    _cached_weights_path,
    _default_cache_dir,
    _safe_load_state_dict,
    extract_sub_state_dict,
)


class TestExtractSubStateDict:
    def test_matching_prefix(self):
        sd = {"backbone.stem.0.conv.weight": torch.ones(1), "neck.out.0.weight": torch.ones(1)}
        result = extract_sub_state_dict(sd, "backbone.")
        assert "stem.0.conv.weight" in result
        assert len(result) == 1

    def test_no_match(self):
        sd = {"backbone.x": torch.ones(1)}
        result = extract_sub_state_dict(sd, "neck.")
        assert result == {}

    def test_strips_prefix(self):
        sd = {"bbox_head.cls.0.weight": torch.ones(1)}
        result = extract_sub_state_dict(sd, "bbox_head.")
        assert "cls.0.weight" in result
        assert "bbox_head.cls.0.weight" not in result

    def test_empty_dict(self):
        result = extract_sub_state_dict({}, "backbone.")
        assert result == {}

    def test_multiple_matches(self):
        sd = {
            "backbone.stage1.0.weight": torch.ones(1),
            "backbone.stage2.0.weight": torch.ones(1),
            "neck.x": torch.ones(1),
        }
        result = extract_sub_state_dict(sd, "backbone.")
        assert len(result) == 2


def _rand_like(v):
    """Create a random tensor matching v's shape, dtype, and device."""
    if v.dtype == torch.long:
        return torch.randint(0, 256, v.shape, dtype=v.dtype, device=v.device)
    return torch.randn_like(v)


class TestSafeLoadStateDict:
    def test_matching_shapes(self, tiny_model):
        new_sd = {k: _rand_like(v) for k, v in tiny_model.state_dict().items()}
        _safe_load_state_dict(tiny_model, new_sd)
        loaded_sd = tiny_model.state_dict()
        for k in new_sd:  # noqa: PLC0206
            assert torch.equal(new_sd[k], loaded_sd[k])

    def test_partial_match(self, tiny_model):
        original = {k: v.clone() for k, v in tiny_model.state_dict().items()}
        new_sd = {}
        for k, v in tiny_model.state_dict().items():
            if "backbone" in k:
                new_sd[k] = _rand_like(v)
            else:
                new_sd[k] = torch.randn(999)
        _safe_load_state_dict(tiny_model, new_sd)
        loaded = tiny_model.state_dict()
        for k in new_sd:  # noqa: PLC0206
            if "backbone" in k and k in loaded:
                assert torch.allclose(new_sd[k], loaded[k])
            elif k in loaded:
                assert torch.allclose(original[k], loaded[k])

    def test_no_match(self, tiny_model):
        original = {k: v.clone() for k, v in tiny_model.state_dict().items()}
        new_sd = {k: torch.randn(999) for k, v in tiny_model.state_dict().items()}
        _safe_load_state_dict(tiny_model, new_sd)
        for k, v in tiny_model.state_dict().items():
            assert torch.allclose(original[k], v)

    def test_extra_keys_ignored(self, tiny_model):
        new_sd = dict(tiny_model.state_dict())
        new_sd["fake.key"] = torch.ones(10)
        _safe_load_state_dict(tiny_model, new_sd)

    def test_missing_keys_ok(self, tiny_model):
        new_sd = dict(tiny_model.state_dict())
        del list(new_sd.keys())[0]  # noqa: RUF015
        _safe_load_state_dict(tiny_model, new_sd)


class TestCacheUtils:
    def test_default_cache_dir(self):
        cache_dir = _default_cache_dir()
        assert isinstance(cache_dir, Path)
        assert cache_dir.exists()

    def test_cached_weights_path(self):
        url = "https://example.com/mmdet/v3.0/rtmdet/model.pth"
        path = _cached_weights_path(url)
        assert path.name == "model.pth"
