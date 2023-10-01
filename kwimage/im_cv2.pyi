from typing import Iterable
from numpy import ndarray
from typing import Tuple
from numbers import Number
from typing import Dict
from nptyping import UInt8
from typing import Any
import kwimage
import numpy
from _typeshed import Incomplete


def imscale(img,
            scale,
            interpolation: Incomplete | None = ...,
            return_scale: bool = ...) -> None:
    ...


def imcrop(img: ndarray,
           dsize: Tuple[None | int, None | int],
           about: Tuple[str | int, str | int] | None = None,
           origin: Tuple[int, int] | None = None,
           border_value: Number | Tuple | str | None = None,
           interpolation: str = 'nearest') -> ndarray:
    ...


DTYPE_KEY_TO_DTYPE: Incomplete
DTYPE_TO_DTYPE_KEY: Incomplete
CV2_ALLOWED_DTYPE_MAPPINGS: Incomplete


def imresize(
    img: ndarray,
    scale: float | Tuple[float, float] | None = None,
    dsize: Tuple[int | None, int | None] | None = None,
    max_dim: int | None = None,
    min_dim: int | None = None,
    interpolation: str | int | None = None,
    grow_interpolation: str | int | None = None,
    letterbox: bool = False,
    return_info: bool = False,
    antialias: bool = False,
    border_value: int | float | Iterable[int | float] = 0
) -> ndarray | Tuple[ndarray, Dict]:
    ...


def convert_colorspace(img: ndarray,
                       src_space: str,
                       dst_space: str,
                       copy: bool = ...,
                       implicit: bool = False,
                       dst: ndarray[Any, UInt8] | None = None) -> ndarray:
    ...


def gaussian_patch(
        shape: Tuple[int, int] = ...,
        sigma: float | Tuple[float, float] | None = None) -> ndarray:
    ...


def gaussian_blur(image: ndarray,
                  kernel: int | Tuple[int, int] | None = None,
                  sigma: float | Tuple[float, float] | None = None,
                  border_mode: str | int | None = None,
                  dst: ndarray | None = None) -> ndarray:
    ...


def warp_affine(image: ndarray,
                transform: ndarray | dict | kwimage.Affine,
                dsize: Tuple[int, int] | None | str = None,
                antialias: bool = ...,
                interpolation: str | int = 'linear',
                border_mode: str | int | None = None,
                border_value: int | float | Iterable[int | float] = 0,
                large_warp_dim: int | None | str = None,
                return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...


def morphology(data,
               mode: str,
               kernel: ndarray | int | Tuple[int, int] = 5,
               element: str = 'rect',
               iterations: int = 1,
               border_mode: str | int = 'constant',
               border_value: int | float | Iterable[int | float] = 0):
    ...


def connected_components(image: ndarray,
                         connectivity: int = 8,
                         ltype: numpy.dtype | str | int = ...,
                         with_stats: bool = ...,
                         algo: str = 'default') -> Tuple[ndarray, dict]:
    ...


def warp_projective(
        image: ndarray,
        transform: ndarray | dict | kwimage.Projective,
        dsize: Tuple[int, int] | None | str = None,
        antialias: bool = ...,
        interpolation: str | int = 'linear',
        border_mode: str | int | None = None,
        border_value: int | float | Iterable[int | float] = 0,
        large_warp_dim: int | None | str = None,
        return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...


def warp_image(image: ndarray,
               transform: ndarray | dict | kwimage.Matrix,
               dsize: Tuple[int, int] | None | str = None,
               antialias: bool = ...,
               interpolation: str | int = 'linear',
               border_mode: str | int | None = None,
               border_value: int | float | Iterable[int | float] = 0,
               large_warp_dim: int | None | str = None,
               return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...
