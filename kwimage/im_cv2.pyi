from typing import Union
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
from typing import Any


def imscale(img,
            scale,
            interpolation: Incomplete | None = ...,
            return_scale: bool = ...) -> None:
    ...


def imcrop(img: ndarray,
           dsize: Tuple[Union[None, int], Union[None, int]],
           about: Tuple[Union[str, int], Union[str, int]] = None,
           origin: Union[Tuple[int, int], None] = None,
           border_value: Union[Number, Tuple, str] = None,
           interpolation: str = 'nearest') -> ndarray:
    ...


def imresize(
    img: ndarray,
    scale: Union[float, Tuple[float, float]] = None,
    dsize: Tuple[Union[int, None], Union[int, None]] = None,
    max_dim: int = None,
    min_dim: int = None,
    interpolation: Union[str, int] = None,
    grow_interpolation: Union[str, int] = None,
    letterbox: bool = False,
    return_info: bool = False,
    antialias: bool = False,
    border_value: Union[int, float, Iterable[Union[int, float]]] = 0
) -> ndarray | Tuple[ndarray, Dict]:
    ...


def convert_colorspace(img: ndarray,
                       src_space: str,
                       dst_space: str,
                       copy: bool = ...,
                       implicit: bool = False,
                       dst: ndarray[Any, UInt8] = None) -> ndarray:
    ...


def gaussian_patch(
        shape: Tuple[int, int] = ...,
        sigma: Union[float, Tuple[float, float], None] = None) -> ndarray:
    ...


def gaussian_blur(image: ndarray,
                  kernel: Union[int, Tuple[int, int]] = None,
                  sigma: Union[float, Tuple[float, float]] = None,
                  border_mode: Union[str, int, None] = None,
                  dst: Union[ndarray, None] = None) -> ndarray:
    ...


def warp_affine(image: ndarray,
                transform: Union[ndarray, dict, kwimage.Affine],
                dsize: Union[Tuple[int, int], None, str] = None,
                antialias: bool = ...,
                interpolation: Union[str, int] = 'linear',
                border_mode: Union[str, int] = None,
                border_value: Union[int, float, Iterable[Union[int,
                                                               float]]] = 0,
                large_warp_dim: Union[int, None, str] = None,
                return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...


def morphology(data,
               mode: str,
               kernel: Union[ndarray, int, Tuple[int, int]] = 5,
               element: str = 'rect',
               iterations: int = 1,
               border_mode: Union[str, int] = 'constant',
               border_value: Union[int, float, Iterable[Union[int,
                                                              float]]] = 0):
    ...


def connected_components(image: ndarray,
                         connectivity: int = 8,
                         ltype: Union[numpy.dtype, str, int] = ...,
                         with_stats: bool = ...,
                         algo: str = 'default') -> Tuple[ndarray, dict]:
    ...


def warp_projective(
        image: ndarray,
        transform: Union[ndarray, dict, kwimage.Projective],
        dsize: Union[Tuple[int, int], None, str] = None,
        antialias: bool = ...,
        interpolation: Union[str, int] = 'linear',
        border_mode: Union[str, int] = None,
        border_value: Union[int, float, Iterable[Union[int, float]]] = 0,
        large_warp_dim: Union[int, None, str] = None,
        return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...


def warp_image(image: ndarray,
               transform: Union[ndarray, dict, kwimage.Matrix],
               dsize: Union[Tuple[int, int], None, str] = None,
               antialias: bool = ...,
               interpolation: Union[str, int] = 'linear',
               border_mode: Union[str, int] = None,
               border_value: Union[int, float, Iterable[Union[int,
                                                              float]]] = 0,
               large_warp_dim: Union[int, None, str] = None,
               return_info: bool = False) -> ndarray | Tuple[ndarray, Dict]:
    ...
