from typing import Sequence
from typing import Union
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

ARRAY_TYPES: Incomplete


class Spatial(ub.NiceRepr):
    ...


class ObjectList(Spatial):
    data: Incomplete
    meta: Incomplete

    def __init__(self, data, meta: Incomplete | None = ...) -> None:
        ...

    def __len__(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def dtype(self):
        ...

    def __nice__(self):
        ...

    def __getitem__(self, index):
        ...

    def __iter__(self):
        ...

    def translate(self,
                  offset,
                  output_dims: Incomplete | None = ...,
                  inplace: bool = ...):
        ...

    def scale(self,
              factor,
              output_dims: Incomplete | None = ...,
              inplace: bool = ...):
        ...

    def warp(self,
             transform,
             input_dims: Incomplete | None = ...,
             output_dims: Incomplete | None = ...,
             inplace: bool = ...):
        ...

    def apply(self, func):
        ...

    def to_coco(self, style: str = ...) -> Generator[Any, None, None]:
        ...

    def compress(self, flags, axis: int = ...):
        ...

    def take(self, indices, axis: int = ...):
        ...

    def draw(self, **kwargs):
        ...

    def draw_on(self, image, **kwargs):
        ...

    def tensor(self, device=...):
        ...

    def numpy(self):
        ...

    @classmethod
    def concatenate(cls,
                    items: Sequence[ObjectList],
                    axis: Union[int, None] = 0) -> ObjectList:
        ...

    def is_tensor(cls) -> None:
        ...

    def is_numpy(cls) -> None:
        ...
