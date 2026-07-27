
from rtmdet.typings import PresetName, RotatedPresetName

_PRETRAINED_URLS: dict[PresetName, str] = {
    "tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "small": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "medium": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "large": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
}

_ROTATED_PRETRAINED_URLS: dict[RotatedPresetName, str] = {
    "tiny": "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota/rotated_rtmdet_tiny-3x-dota-9d821076.pth",
    "small": "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota-11f6ccf5.pth",
    "medium": "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_m-3x-dota/rotated_rtmdet_m-3x-dota-beeadda6.pth",
    "large": "https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-3x-dota/rotated_rtmdet_l-3x-dota-23992372.pth",
}
