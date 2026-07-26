from typing import Literal, TypeAlias

import torch

StateDict: TypeAlias = dict[str, torch.Tensor]
PresetName: TypeAlias = Literal["tiny", "small", "medium", "large"]
