from typing import Tuple
from numpy import ndarray
from numbers import Number


def grab_test_image(key: str = 'astro',
                    space: str = 'rgb',
                    dsize: Tuple[int, int] | None = None,
                    interpolation: str = ...) -> ndarray:
    ...


def grab_test_image_fpath(key: str = 'astro',
                          dsize: None | Tuple[int, int] = None,
                          overviews: None | int = None) -> str:
    ...


def checkerboard(num_squares: int | str = 'auto',
                 square_shape: int | Tuple[int, int] | str = 'auto',
                 dsize: Tuple[int, int] = ...,
                 dtype: type = float,
                 on_value: Number | int = 1,
                 off_value: Number | int = 0):
    ...
