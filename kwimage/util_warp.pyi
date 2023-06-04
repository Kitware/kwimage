from typing import Tuple
from torch import Tensor
from numpy.typing import ArrayLike
from typing import Sequence
from _typeshed import Incomplete


def warp_tensor(inputs: Tensor,
                mat: Tensor,
                output_dims: Tuple[int, ...],
                mode: str = 'bilinear',
                padding_mode: str = 'zeros',
                isinv: bool = False,
                ishomog: bool | None = None,
                align_corners: bool = False,
                new_mode: bool = ...) -> Tensor:
    ...


def subpixel_align(dst, src, index, interp_axes: Incomplete | None = ...):
    ...


def subpixel_set(dst: ArrayLike,
                 src: ArrayLike,
                 index: Tuple[slice],
                 interp_axes: tuple | None = None):
    ...


def subpixel_accum(dst: ArrayLike,
                   src: ArrayLike,
                   index: Tuple[slice],
                   interp_axes: tuple | None = None):
    ...


def subpixel_maximum(dst: ArrayLike,
                     src: ArrayLike,
                     index: Tuple[slice],
                     interp_axes: tuple | None = None):
    ...


def subpixel_minimum(dst: ArrayLike,
                     src: ArrayLike,
                     index: Tuple[slice],
                     interp_axes: tuple | None = None):
    ...


def subpixel_slice(inputs: ArrayLike, index: Tuple[slice]):
    ...


def subpixel_translate(inputs: ArrayLike,
                       shift: Sequence,
                       interp_axes: Sequence | None = None,
                       output_shape: tuple | None = None):
    ...


def warp_points(matrix: ArrayLike, pts: ArrayLike, homog_mode: str = 'divide'):
    ...


def remove_homog(pts, mode: str = ...):
    ...


def add_homog(pts):
    ...


def subpixel_getvalue(img: ArrayLike,
                      pts: ArrayLike,
                      coord_axes: Sequence | None = None,
                      interp: str = 'bilinear',
                      bordermode: str = 'edge'):
    ...


def subpixel_setvalue(img: ArrayLike,
                      pts: ArrayLike,
                      value: ArrayLike,
                      coord_axes: Sequence | None = None,
                      interp: str = 'bilinear',
                      bordermode: str = 'edge'):
    ...
