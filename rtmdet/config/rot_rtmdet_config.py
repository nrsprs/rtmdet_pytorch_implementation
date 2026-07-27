from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import Field, PositiveInt

from rtmdet.config.rtmdet_config import RTMDetConfig
from rtmdet.typings import RotatedPresetName


class RotRTMDetConfig(RTMDetConfig):
    num_classes: PositiveInt = Field(15, description="Number of classes (DOTA v1.0 default)")
    nms_iou_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Rotated IoU threshold for NMS")

    _PRESET_DIR: ClassVar[Path] = Path(__file__).resolve().parent / "defaults"

    @classmethod
    def from_preset(cls, name: RotatedPresetName) -> "RotRTMDetConfig":
        preset_path = (cls._PRESET_DIR / f"rotated_rtmdet_{name}.yaml").resolve()
        cfg = cls(**yaml.safe_load(preset_path.read_text()))
        cfg.preset_name = name
        return cfg