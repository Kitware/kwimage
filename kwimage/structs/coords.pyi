from numpy.typing import ArrayLike
from typing import Sequence
from typing import Tuple
from typing import Callable
from typing import Any
from numpy import ndarray
from typing import List
import matplotlib as mpl
import ubelt as ub
from _typeshed import Incomplete
from kwimage.structs import _generic

from kwimage._typing import SKImageGeometricTransform

__docstubs__: str


class Coords(_generic.Spatial, ub.NiceRepr):
    data: Incomplete
    meta: Incomplete

    def __init__(self,
                 data: Incomplete | None = ...,
                 meta: Incomplete | None = ...) -> None:
        ...

    def __array__(self):
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    @property
    def dtype(self):
        ...

    @property
    def dim(self):
        ...

    @property
    def shape(self):
        ...

    def copy(self):
        ...

    @classmethod
    def random(Coords,
               num: int = ...,
               dim: int = ...,
               rng: Incomplete | None = ...,
               meta: Incomplete | None = ...):
        ...

    def is_numpy(self) -> bool:
        ...

    def is_tensor(self) -> bool:
        ...

    def compress(self,
                 flags: ArrayLike,
                 axis: int = 0,
                 inplace: bool = False) -> Coords:
        ...

    def take(self,
             indices: ArrayLike,
             axis: int = 0,
             inplace: bool = False) -> Coords:
        ...

    def astype(self, dtype, inplace: bool = False) -> Coords:
        ...

    def round(self, decimals: int = 0, inplace: bool = False) -> Coords:
        ...

    def view(self, *shape) -> Coords:
        ...

    @classmethod
    def concatenate(cls, coords: Sequence[Coords], axis: int = 0) -> Coords:
        ...

    @property
    def device(self):
        ...

    def tensor(self, device=...) -> Coords:
        ...

    def numpy(self) -> Coords:
        ...

    def reorder_axes(self,
                     new_order: Tuple[int],
                     inplace: bool = False) -> Coords:
        ...

    def warp(self,
             transform: SKImageGeometricTransform | ArrayLike | Any | Callable,
             input_dims: Tuple | None = None,
             output_dims: Tuple | None = None,
             inplace: bool = False) -> Coords:
        ...

    def to_imgaug(self, input_dims) -> Any:
        ...

    @classmethod
    def from_imgaug(cls, kpoi):
        ...

    def scale(self,
              factor: float | Tuple[float, float],
              about: Tuple | None = None,
              output_dims: Tuple | None = None,
              inplace: bool = False) -> Coords:
        ...

    def translate(self,
                  offset: float | Tuple[float, float],
                  output_dims: Tuple | None = None,
                  inplace: bool = False) -> Coords:
        ...

    def rotate(self,
               theta: float,
               about: Tuple | None = None,
               output_dims: Tuple | None = None,
               inplace: bool = False) -> Coords:
        ...

    def fill(self,
             image,
             value,
             coord_axes: Tuple | None = None,
             interp: str = ...) -> ndarray:
        ...

    def soft_fill(self,
                  image,
                  coord_axes: Tuple | None = None,
                  radius: int = ...) -> ndarray:
        ...

    def draw_on(self,
                image: Incomplete | None = ...,
                fill_value: int = ...,
                coord_axes: Tuple = ...,
                interp: str = ...) -> ndarray:
        ...

    def draw(self,
             color: str = ...,
             ax: Incomplete | None = ...,
             alpha: Incomplete | None = ...,
             coord_axes: Tuple = ...,
             radius: int = ...,
             setlim: bool = False) -> List[mpl.collections.PatchCollection]:
        ...
