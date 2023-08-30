from typing import Any
from typing import Tuple
from typing import Callable
import kwimage
from numpy.typing import ArrayLike
from numpy import ndarray
from typing import List
from typing import Dict
import kwcoco
from _typeshed import Incomplete
from kwimage.structs import _generic

from kwimage._typing import SKImageGeometricTransform

__docstubs__: str


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
             transform: ArrayLike | Callable | kwimage.Affine
             | SKImageGeometricTransform | Any,
             input_dims: Tuple | None = None,
             output_dims: Tuple | None = None,
             inplace: bool = False):
        ...

    def scale(self,
              factor: float | Tuple[float, float],
              output_dims: Tuple | None = None,
              inplace: bool = ...):
        ...

    def translate(self,
                  offset,
                  output_dims: Tuple | None = None,
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
                image: ndarray | None = None,
                color: str | Any | List[Any] = 'white',
                radius: None | int = None,
                copy: bool = False):
        ...

    def draw(self,
             color: str = ...,
             ax: Incomplete | None = ...,
             alpha: Incomplete | None = ...,
             radius: int = ...,
             setlim: bool = ...,
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
                  coco_kpts: list | dict,
                  class_idxs: list | None = None,
                  classes: list | kwcoco.CategoryTree | None = None,
                  warn: bool = False):
        ...


class PointsList(_generic.ObjectList):
    ...
