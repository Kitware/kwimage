from numpy import ndarray
from typing import List
import shapely
from typing import Any
from typing import Tuple
import kwimage
from typing import Callable
from numpy.typing import ArrayLike
import numpy as np
from numbers import Number
from typing import Optional
import matplotlib
from torch import Tensor
from numpy.random import RandomState
from typing import Sequence
import torch
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator

from kwimage._typing import SKImageGeometricTransform

__docstubs__: str


class NeedsWarpCorners(AssertionError):
    ...


class BoxFormat:
    cannonical: Incomplete
    XYWH: Incomplete
    CXYWH: Incomplete
    LTRB: Incomplete
    TLBR = LTRB
    XXYY: Incomplete
    aliases: Incomplete
    blocklist: Incomplete


def box_ious(ltrb1: ndarray,
             ltrb2: ndarray,
             bias: int = 0,
             impl: str | None = None):
    ...


class _BoxConversionMixins:
    convert_funcs: Incomplete

    def toformat(self, format: str, copy: bool = True) -> Boxes:
        ...

    def to_xxyy(self, copy: bool = ...):
        ...

    to_extent = to_xxyy

    def to_xywh(self, copy: bool = ...):
        ...

    def to_cxywh(self, copy: bool = ...):
        ...

    def to_ltrb(self, copy: bool = ...):
        ...

    def to_tlbr(self, **kwargs):
        ...

    def to_imgaug(self, shape: tuple):
        ...

    def to_shapely(self) -> List[shapely.geometry.Polygon]:
        ...

    def to_shapley(self) -> List[shapely.geometry.Polygon]:
        ...

    @classmethod
    def from_shapely(cls, geom: shapely.geometry.Polygon) -> Boxes:
        ...

    @classmethod
    def coerce(Boxes, data, format: str | None = None, **kwargs) -> Boxes:
        ...

    @classmethod
    def from_imgaug(Boxes, bboi: Any) -> Boxes:
        ...

    @classmethod
    def from_slice(Boxes,
                   slices,
                   shape: Tuple[int, int] | None = None,
                   clip: bool = True,
                   endpoint: bool = True,
                   wrap: bool = False):
        ...

    def to_slices(self, endpoint: bool = True) -> List[Tuple[slice, slice]]:
        ...

    def to_coco(self, style: str = ...) -> Generator[List[float], None, None]:
        ...

    def to_polygons(self) -> kwimage.PolygonList:
        ...


class _BoxPropertyMixins:

    @property
    def xy_center(self):
        ...

    @property
    def components(self):
        ...

    @property
    def dtype(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def tl_x(self):
        ...

    @property
    def tl_y(self):
        ...

    @property
    def br_x(self):
        ...

    @property
    def br_y(self):
        ...

    @property
    def width(self):
        ...

    @property
    def height(self):
        ...

    @property
    def aspect_ratio(self):
        ...

    @property
    def area(self):
        ...

    @property
    def center(self) -> Tuple[ndarray, ndarray]:
        ...

    @property
    def center_x(self) -> Tuple[ndarray, ndarray]:
        ...

    @property
    def center_y(self) -> Tuple[ndarray, ndarray]:
        ...


class _BoxTransformMixins:

    def warp(self,
             transform: ArrayLike | Callable | kwimage.Affine
             | SKImageGeometricTransform | Any,
             input_dims: Tuple | None = None,
             output_dims: Tuple | None = None,
             inplace: bool = False) -> Boxes:
        ...

    def corners(self) -> np.ndarray:
        ...

    def scale(self,
              factor: float | Tuple[float, float],
              about: str | ArrayLike = 'origin',
              output_dims: Tuple | None = None,
              inplace: bool = False) -> Boxes:
        ...

    def translate(self,
                  amount,
                  output_dims: Tuple | None = None,
                  inplace: bool = ...) -> Boxes:
        ...

    def clip(self,
             x_min: int,
             y_min: int,
             x_max: int,
             y_max: int,
             inplace: bool = False) -> Boxes:
        ...

    def resize(self,
               width: Number | ndarray | None = None,
               height: Number | ndarray | None = None,
               inplace: bool = False,
               about: str = 'xy') -> Boxes:
        ...

    def pad(self,
            x_left: int | float,
            y_top: int | float,
            x_right: int | float,
            y_bot: int | float,
            inplace: bool = ...) -> Boxes:
        ...

    def transpose(self):
        ...


class _BoxDrawMixins:

    def draw(self,
             color: str | Any | List[Any] = 'blue',
             alpha: float | List[float] | None = None,
             labels: List[str] | None = None,
             centers: bool = False,
             fill: bool = ...,
             lw: float = 2,
             ax: Optional[matplotlib.axes.Axes] = None,
             setlim: bool = False,
             **kwargs):
        ...

    def draw_on(self,
                image: ndarray | None = None,
                color: str | Any | List[Any] = 'blue',
                alpha: float | None = None,
                labels: List[str] | None = None,
                copy: bool = False,
                thickness: int = 2,
                edgecolor: str | tuple | None = None,
                facecolor: str | tuple | None = None,
                fill: bool = False,
                border: bool = True,
                label_loc: str = 'top_left') -> ndarray:
        ...


class Boxes(_BoxConversionMixins, _BoxPropertyMixins, _BoxTransformMixins,
            _BoxDrawMixins, ub.NiceRepr):
    data: ndarray | Tensor | Boxes
    format: str

    def __init__(self,
                 data: ndarray | Tensor | Boxes,
                 format: str | None = None,
                 check: bool = True) -> None:
        ...

    def __getitem__(self, index):
        ...

    def __eq__(self, other):
        ...

    def __len__(self) -> int:
        ...

    def __nice__(self) -> str:
        ...

    @classmethod
    def random(Boxes,
               num: int = 1,
               scale: float | Tuple[float, float] = 1.0,
               format: str = ...,
               anchors: ndarray | None = None,
               anchor_std: float = ...,
               tensor: bool = False,
               rng: None | int | RandomState = None) -> Boxes:
        ...

    def copy(self) -> Boxes:
        ...

    @classmethod
    def concatenate(cls, boxes: Sequence[Boxes], axis: int = 0) -> Boxes:
        ...

    def compress(self,
                 flags: ArrayLike,
                 axis: int = 0,
                 inplace: bool = False) -> Boxes:
        ...

    def take(self, idxs, axis: int = 0, inplace: bool = False) -> Boxes:
        ...

    def is_tensor(self) -> bool:
        ...

    def is_numpy(self) -> bool:
        ...

    @property
    def device(self):
        ...

    def astype(self, dtype) -> Boxes:
        ...

    def round(self, inplace: bool = False) -> Boxes:
        ...

    def quantize(self, inplace: bool = False, dtype: type = ...) -> Boxes:
        ...

    def numpy(self) -> Boxes:
        ...

    def tensor(self, device: int | None | torch.device = ...) -> Boxes:
        ...

    def ious(self,
             other: Boxes,
             bias: int = 0,
             impl: str = 'auto',
             mode: str | None = None) -> ndarray:
        ...

    def iooas(self, other: Boxes, bias: int = 0) -> ndarray:
        ...

    def isect_area(self, other: Boxes, bias: int = 0) -> ndarray:
        ...

    def intersection(self, other: Boxes) -> Boxes:
        ...

    def union_hull(self, other: Boxes) -> Boxes:
        ...

    def bounding_box(self) -> Boxes:
        ...

    def contains(self, other: kwimage.Points) -> ArrayLike:
        ...

    def view(self, *shape: Tuple[int, ...]) -> Boxes:
        ...
