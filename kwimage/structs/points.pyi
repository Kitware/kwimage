from typing import Union
from typing import Callable
import kwimage
from skimage.transform._geometric import GeometricTransform
from numpy.typing import ArrayLike
from typing import Any
from typing import Tuple
from typing import Dict
import kwcoco
from _typeshed import Incomplete
from kwimage.structs import _generic
from typing import Any


class _PointsWarpMixin:

    def to_imgaug(self, input_dims):
        ...

    @classmethod
    def from_imgaug(cls, kpoi):
        ...

    @property
    def dtype(self):
        ...

    def warp(self,
             transform: Union[ArrayLike, Callable, kwimage.Affine,
                              GeometricTransform, Any],
             input_dims: Tuple = None,
             output_dims: Tuple = None,
             inplace: bool = False):
        ...

    def scale(self,
              factor: Union[float, Tuple[float, float]],
              output_dims: Tuple = None,
              inplace: bool = ...):
        ...

    def translate(self,
                  offset,
                  output_dims: Tuple = None,
                  inplace: bool = ...):
        ...


class Points(_generic.Spatial, _PointsWarpMixin):
    __datakeys__: Incomplete
    __metakeys__: Incomplete
    data: Incomplete
    meta: Incomplete

    def __init__(self,
                 data: Incomplete | None = ...,
                 meta: Incomplete | None = ...,
                 datakeys: Incomplete | None = ...,
                 metakeys: Incomplete | None = ...,
                 **kwargs) -> None:
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def xy(self):
        ...

    @classmethod
    def random(Points,
               num: int = ...,
               classes: Incomplete | None = ...,
               rng: Incomplete | None = ...):
        ...

    def is_numpy(self):
        ...

    def is_tensor(self):
        ...

    def tensor(self, device=...):
        ...

    def round(self, inplace: bool = False):
        ...

    def numpy(self):
        ...

    def draw_on(self,
                image: Incomplete | None = ...,
                color: str = ...,
                radius: Incomplete | None = ...,
                copy: bool = ...):
        ...

    def draw(self,
             color: str = ...,
             ax: Incomplete | None = ...,
             alpha: Incomplete | None = ...,
             radius: int = ...,
             **kwargs):
        ...

    def compress(self, flags, axis: int = ..., inplace: bool = ...):
        ...

    def take(self, indices, axis: int = ..., inplace: bool = ...):
        ...

    @classmethod
    def concatenate(cls, points, axis: int = ...):
        ...

    def to_coco(self, style: str = 'orig') -> Dict:
        ...

    @classmethod
    def coerce(cls, data):
        ...

    @classmethod
    def from_coco(cls,
                  coco_kpts: Union[list, dict],
                  class_idxs: list = None,
                  classes: Union[list, kwcoco.CategoryTree] = None,
                  warn: bool = False):
        ...


class PointsList(_generic.ObjectList):
    ...
