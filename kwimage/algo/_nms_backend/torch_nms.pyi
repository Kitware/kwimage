from torch import Tensor
from typing import Union
import torch


def torch_nms(ltrb: Tensor,
              scores: Tensor,
              classes: Union[Tensor, None] = None,
              thresh: float = 0.5,
              bias: int = ...,
              fast: bool = ...) -> torch.ByteTensor:
    ...


def test_class_torch() -> None:
    ...
