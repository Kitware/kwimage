from typing import Iterable
from numpy import ndarray
from typing import Union
from numbers import Number
from typing import List
from typing import Tuple


def stack_images(images: Iterable[ndarray],
                 axis: int = 0,
                 resize: Union[int, str, None] = None,
                 interpolation: Union[int, str] = None,
                 overlap: int = 0,
                 return_info: bool = False,
                 bg_value: Union[Number, ndarray] = None,
                 pad: int = None) -> Tuple[ndarray, List]:
    ...


def stack_images_grid(
        images: Iterable[ndarray],
        chunksize: int = None,
        axis: int = 0,
        overlap: int = 0,
        pad: int = None,
        return_info: bool = False,
        bg_value: Union[Number, ndarray] = None) -> Tuple[ndarray, List]:
    ...
