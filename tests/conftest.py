import pytest
from PIL import Image, ImageDraw

from rtmdet import RTMDet, RTMDetConfig


@pytest.fixture(scope="session")
def tiny_config():
    return RTMDetConfig.from_preset("tiny")


@pytest.fixture(scope="session")
def small_config():
    return RTMDetConfig.from_preset("small")


@pytest.fixture(scope="session")
def medium_config():
    return RTMDetConfig.from_preset("medium")


@pytest.fixture(scope="session")
def large_config():
    return RTMDetConfig.from_preset("large")


@pytest.fixture(scope="session")
def all_configs(tiny_config, small_config, medium_config, large_config):
    return [tiny_config, small_config, medium_config, large_config]


@pytest.fixture(scope="session")
def preset_names():
    return ["tiny", "small", "medium", "large"]


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    return RTMDet(tiny_config).to("cpu")


@pytest.fixture(scope="session")
def medium_model(medium_config):
    return RTMDet(medium_config).to("cpu")


@pytest.fixture(scope="session")
def custom_config():
    return RTMDetConfig(
        deepen_factor=0.5,
        widen_factor=0.5,
        num_classes=10,
        img_size=256,
        exp_on_reg=True,
        score_threshold=0.5,
    )


@pytest.fixture(scope="session")
def custom_model(custom_config):
    return RTMDet(custom_config).to("cpu")


@pytest.fixture
def tmp_image_path(tmp_path):
    img = Image.new("RGB", (640, 640), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill=(255, 0, 0))
    draw.rectangle([400, 400, 550, 550], fill=(0, 255, 0))
    path = tmp_path / "test_image.png"
    img.save(str(path))
    return path


@pytest.fixture
def tmp_checkpoint_path(tmp_path):
    return tmp_path / "checkpoint.pt"