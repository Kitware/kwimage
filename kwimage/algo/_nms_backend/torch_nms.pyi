from torch import Tensor
import torch


def torch_nms(ltrb: Tensor,
              scores: Tensor,
              classes: Tensor | None = None,
              thresh: float = 0.5,
              bias: int = ...,
              fast: bool = ...) -> torch.ByteTensor:
    ...


def test_class_torch() -> None:
    ...
