from numpy import ndarray
from typing import List
import shapely
from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable
import kwimage
from numpy.typing import ArrayLike
import numpy as np
from torch import Tensor
from numpy.random import RandomState
from typing import Sequence
import skimage
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class NeedsWarpCorners(AssertionError):
    ...


class BoxFormat:
    cannonical: Incomplete
    XYWH: Incomplete
    CXYWH: Incomplete
    LTRB: Incomplete
    TLBR: Incomplete
    XXYY: Incomplete
    aliases: Incomplete
    blocklist: Incomplete


def box_ious(ltrb1: ndarray,
             ltrb2: ndarray,
             bias: int = 0,
             impl: Incomplete | None = ...):
    ...


class _BoxConversionMixins:
    convert_funcs: Incomplete

    def toformat(self, format: str, copy: bool = True) -> Boxes:
        ...

    def to_xxyy(self, copy: bool = ...):
        ...

    to_extent: Incomplete

    def to_xywh(self, copy: bool = ...):
        ...

    def to_cxywh(self, copy: bool = ...):
        ...

    def to_ltrb(self, copy: bool = ...):
        ...

    to_tlbr: Incomplete

    def to_imgaug(self, shape: tuple):
        ...

    def to_shapely(self) -> List[shapely.geometry.Polygon]:
        ...

    to_shapley: Incomplete

    @classmethod
    def from_shapely(cls, geom) -> Boxes:
        ...

    @classmethod
    def coerce(Boxes, data):
        ...

    @classmethod
    def from_imgaug(Boxes, bboi: Any) -> Boxes:
        ...

    @classmethod
    def from_slice(Boxes,
                   slices,
                   shape: Tuple[int, int] = None,
                   clip: bool = True,
                   endpoint: bool = True):
        ...

    def to_slices(self, endpoint: bool = True):
        ...

    def to_coco(self, style: str = ...) -> Generator[Any, None, None]:
        ...

    def to_polygons(self):
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


class _BoxTransformMixins:

    def warp(self,
             transform: Union[ArrayLike, Callable, kwimage.Affine,
                              skimage.transform._geometric.GeometricTransform,
                              Any],
             input_dims: Tuple = None,
             output_dims: Tuple = None,
             inplace: bool = False):
        ...

    def corners(self) -> np.ndarray:
        ...

    def scale(self,
              factor: Union[float, Tuple[float, float]],
              about: Union[str, ArrayLike] = 'origin',
              output_dims: Tuple = None,
              inplace: bool = False):
        ...

    def translate(self,
                  amount,
                  output_dims: Tuple = None,
                  inplace: bool = ...):
        ...

    def clip(self,
             x_min: int,
             y_min: int,
             x_max: int,
             y_max: int,
             inplace: bool = False) -> Boxes:
        ...

    def pad(self, x_left, y_top, x_right, y_bot, inplace: bool = ...):
        ...

    def transpose(self):
        ...


class _BoxDrawMixins:

    def draw(self,
             color: str = ...,
             alpha: Incomplete | None = ...,
             labels: Incomplete | None = ...,
             centers: bool = ...,
             fill: bool = ...,
             lw: int = ...,
             ax: Incomplete | None = ...,
             setlim: bool = ...):
        ...

    def draw_on(self,
                image: ndarray = None,
                color: Union[str, Any, List[Any]] = 'blue',
                alpha: float = None,
                labels: List[str] = None,
                copy: bool = False,
                thickness: int = 2,
                label_loc: str = 'top_left'):
        ...


class Boxes(_BoxConversionMixins, _BoxPropertyMixins, _BoxTransformMixins,
            _BoxDrawMixins, ub.NiceRepr):
    data: Incomplete
    format: Incomplete

    def __init__(self,
                 data: Union[ndarray, Tensor, Boxes],
                 format: str = None,
                 check: bool = True) -> None:
        ...

    def __getitem__(self, index):
        ...

    def __eq__(self, other):
        ...

    def __len__(self):
        ...

    def __nice__(self):
        ...

    @classmethod
    def random(Boxes,
               num: int = 1,
               scale: Union[float, Tuple[float, float]] = 1.0,
               format: str = ...,
               anchors: ndarray = None,
               anchor_std: float = ...,
               tensor: bool = False,
               rng: Union[None, int, RandomState] = None):
        ...

    def copy(self):
        ...

    @classmethod
    def concatenate(cls, boxes: Sequence[Boxes], axis: int = 0) -> Boxes:
        ...

    def compress(self, flags: ArrayLike, axis: int = 0, inplace: bool = False):
        ...

    def take(self, idxs, axis: int = 0, inplace: bool = False):
        ...

    def is_tensor(self):
        ...

    def is_numpy(self):
        ...

    @property
    def device(self):
        ...

    def astype(self, dtype):
        ...

    def round(self, inplace: bool = False):
        ...

    def quantize(self, inplace: bool = False, dtype: type = ...):
        ...

    def numpy(self):
        ...

    def tensor(self, device=...):
        ...

    def ious(self,
             other: Boxes,
             bias: int = 0,
             impl: str = 'auto',
             mode: Incomplete | None = ...):
        ...

    def iooas(self, other: Boxes, bias: int = 0):
        ...

    def isect_area(self, other, bias: int = ...):
        ...

    def intersection(self, other) -> Boxes:
        ...

    def union_hull(self, other) -> Boxes:
        ...

    def bounding_box(self) -> Boxes:
        ...

    def contains(self, other: kwimage.Points) -> ArrayLike:
        ...

    def view(self, *shape):
        ...
