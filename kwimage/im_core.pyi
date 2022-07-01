from numpy import ndarray
from typing import Any
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict
from _typeshed import Incomplete
from typing import Any


def num_channels(img: ndarray) -> int:
    ...


def ensure_float01(img: ndarray,
                   dtype: type = ...,
                   copy: bool = True) -> ndarray:
    ...


def ensure_uint255(img: ndarray, copy: bool = True) -> ndarray:
    ...


def make_channels_comparable(img1: ndarray,
                             img2: ndarray,
                             atleast3d: bool = False):
    ...


def atleast_3channels(arr: ndarray, copy: bool = True) -> ndarray:
    ...


def padded_slice(data: Any,
                 in_slice: Union[slice, Tuple[slice, ...]],
                 pad: List[Union[int, Tuple]] = None,
                 padkw: Dict = None,
                 return_info: bool = False) -> Tuple[Any, Dict]:
    ...


def normalize(arr,
              mode: str = ...,
              alpha: Incomplete | None = ...,
              beta: Incomplete | None = ...,
              out: Incomplete | None = ...):
    ...


def find_robust_normalizers(
        data: ndarray,
        params: Union[str, dict] = 'auto') -> Dict[str, str | float]:
    ...


def normalize_intensity(imdata: ndarray,
                        return_info: bool = False,
                        nodata: Union[None, int] = None,
                        axis: Union[None, int] = None,
                        dtype: type = ...,
                        params: Union[str, dict] = 'auto',
                        mask: Union[ndarray, None] = None) -> ndarray:
    ...
