from typing import Tuple
from numpy import ndarray
from typing import Union
from numbers import Number


def grab_test_image(key: str = 'astro',
                    space: str = 'rgb',
                    dsize: Tuple[int, int] = None,
                    interpolation: str = ...) -> ndarray:
    ...


def grab_test_image_fpath(key: str = 'astro',
                          dsize: Union[None, Tuple[int, int]] = None,
                          overviews: Union[None, int] = None) -> str:
    ...


def checkerboard(num_squares: Union[int, str] = 'auto',
                 square_shape: Union[int, Tuple[int, int], str] = 'auto',
                 dsize: Tuple[int, int] = ...,
                 dtype: type = float,
                 on_value: Union[Number, int] = 1,
                 off_value: Union[Number, int] = 0):
    ...
