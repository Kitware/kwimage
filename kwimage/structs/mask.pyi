from typing import Union
from typing import List
from numpy import ndarray
from typing import Tuple
from numbers import Number
from numpy.random import RandomState
import kwimage
from typing import Any
import ubelt as ub
from . import _generic
from _typeshed import Incomplete
from typing import Any


class _Mask_Backends:

    def __init__(self) -> None:
        ...

    def get_backend(self, prefs):
        ...


class MaskFormat:
    cannonical: Incomplete
    BYTES_RLE: Incomplete
    ARRAY_RLE: Incomplete
    C_MASK: Incomplete
    F_MASK: Incomplete
    aliases: Incomplete


class _MaskConversionMixin:
    convert_funcs: Incomplete

    def toformat(self, format: str, copy: bool = False) -> Mask:
        ...

    def to_bytes_rle(self, copy: bool = False) -> Mask:
        ...

    def to_array_rle(self, copy: bool = False) -> Mask:
        ...

    def to_fortran_mask(self, copy: bool = False) -> Mask:
        ...

    def to_c_mask(self, copy: bool = False) -> Mask:
        ...

    def numpy(self) -> Mask:
        ...

    def tensor(self, device=...) -> Mask:
        ...


class _MaskConstructorMixin:

    @classmethod
    def from_polygons(Mask, polygons: Union[ndarray, List[ndarray]],
                      dims: Tuple) -> Mask:
        ...

    @classmethod
    def from_mask(Mask,
                  mask: ndarray,
                  offset: Tuple[int, int] = None,
                  shape: Tuple[int, int] = None,
                  method: str = ...):
        ...


class _MaskTransformMixin:

    def scale(self,
              factor: Union[float, Tuple[float, float]],
              output_dims: Tuple[int, int] = None,
              inplace: bool = ...) -> Mask:
        ...

    def warp(self,
             transform: ndarray,
             input_dims: Tuple[int, int] = None,
             output_dims: Tuple[int, int] = None,
             inplace: bool = ...) -> Mask:
        ...

    def translate(self,
                  offset: Union[Tuple, Number],
                  output_dims: Tuple[int, int] = None,
                  inplace: bool = False) -> Mask:
        ...


class _MaskDrawMixin:

    def draw_on(self,
                image: ndarray = None,
                color: Union[str, tuple] = 'blue',
                alpha: float = 0.5,
                show_border: bool = False,
                border_thick: int = ...,
                border_color: str = ...,
                copy: bool = ...) -> ndarray:
        ...

    def draw(self,
             color: Union[str, tuple] = 'blue',
             alpha: float = 0.5,
             ax: Incomplete | None = ...,
             show_border: bool = ...,
             border_thick: int = ...,
             border_color: str = ...) -> None:
        ...


class Mask(ub.NiceRepr, _MaskConversionMixin, _MaskConstructorMixin,
           _MaskTransformMixin, _MaskDrawMixin):
    data: Incomplete
    format: Incomplete

    def __init__(self,
                 data: Incomplete | None = ...,
                 format: Incomplete | None = ...) -> None:
        ...

    @property
    def dtype(self):
        ...

    def __nice__(self):
        ...

    @classmethod
    def random(Mask,
               rng: Union[int, RandomState, None] = None,
               shape: Tuple[int, int] = ...) -> Mask:
        ...

    @classmethod
    def demo(cls) -> Mask:
        ...

    def copy(self) -> Mask:
        ...

    def union(self, *others) -> Mask:
        ...

    def intersection(self, *others) -> Mask:
        ...

    @property
    def shape(self):
        ...

    @property
    def area(self) -> int:
        ...

    def get_patch(self):
        ...

    def get_xywh(self) -> ndarray:
        ...

    def bounding_box(self) -> kwimage.Boxes:
        ...

    def get_polygon(self) -> List[ndarray]:
        ...

    def to_mask(self, dims: Incomplete | None = ...) -> kwimage.Mask:
        ...

    def to_boxes(self) -> kwimage.Boxes:
        ...

    def to_multi_polygon(self, pixels_are: str = ...) -> kwimage.MultiPolygon:
        ...

    def get_convex_hull(self):
        ...

    def iou(self, other):
        ...

    @classmethod
    def coerce(Mask, data: Any, dims: Tuple = None) -> Mask:
        ...

    def to_coco(self, style: str = 'orig') -> dict:
        ...


class MaskList(_generic.ObjectList):

    def to_polygon_list(self) -> kwimage.PolygonList:
        ...

    def to_segmentation_list(self) -> kwimage.SegmentationList:
        ...

    def to_mask_list(self) -> kwimage.MaskList:
        ...
