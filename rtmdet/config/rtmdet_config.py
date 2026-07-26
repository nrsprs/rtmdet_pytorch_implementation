from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, Field, PositiveInt

from rtmdet.typings import PresetName


class RTMDetConfig(BaseModel):
    """Configuration for the RTMDet model"""

    # ---- Identity ----
    preset_name: str = Field(
        "",
        description="Preset slug (tiny/small/medium/large) used to create this config. Empty for manual configs.",
    )
    # ---- Scaling  ----
    deepen_factor: float = Field(
        ...,
        gt=0.0,
        description="Scaling factor for the model depth (e.g., the number of layers or blocks)",
    )
    widen_factor: float = Field(
        ...,
        gt=0.0,
        description="Scaling factor for the model width (e.g., the number of channels or neurons)",
    )
    neck_out_channels: PositiveInt = Field(
        256,
        description="Number of channels of the output convolution layers in the PAFPN module",
    )
    # ---- Head ----
    exp_on_reg: bool = Field(True, description="Apply exp() to bbox predictions before stride multiplication. False for tiny/small, True for medium/large.")
    head_num_levels: PositiveInt = Field(
        3,
        ge=1,
        description="Number of pyramid levels the head operates on (e.g., 3 for P3-P5). Must equal the number of feature maps provided by the neck",
    )
    head_num_stacked_convs: PositiveInt = Field(
        2,
        description="Number of convolution blocks in each classification/regression tower",
    )
    num_classes: PositiveInt = Field(80, description="Number of classes")
    img_size: PositiveInt = Field(
        640, description="Input image size (assumes a square input)"
    )
    prior_strides: list[int] = Field(
        [8, 16, 32],
        description="Downsample strides of the feature pyramid levels (e.g., [8, 16, 32] for P3-P5)",
    )
    # ---- Post-processing ----
    score_threshold: float = Field(
        0.001,
        ge=0.0,
        le=1.0,
        description="Score threshold to filter predictions before NMS",
    )
    nms_iou_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS. All overlapping boxes with an IoU > are discarded",
    )
    max_num_detections: int = Field(
        300, description="Maximum number of detections kept after NMS per image"
    )

    _PRESET_DIR: ClassVar[Path] = Path(__file__).resolve().parent / "defaults"

    @classmethod
    def from_preset(cls, name: PresetName) -> "RTMDetConfig":
        preset_path = (cls._PRESET_DIR / f"rtmdet_{name}.yaml").resolve()
        cfg = cls(**yaml.safe_load(preset_path.read_text()))
        cfg.preset_name = name
        return cfg
