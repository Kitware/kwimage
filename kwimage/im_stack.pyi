from typing import Iterable
from numpy import ndarray
from numbers import Number
from typing import List
from typing import Tuple


def stack_images(images: Iterable[ndarray],
                 axis: int = 0,
                 resize: int | str | None = None,
                 interpolation: int | str | None = None,
                 overlap: int = 0,
                 return_info: bool = False,
                 bg_value: Number | ndarray | str | None = None,
                 pad: int | None = None,
                 allow_casting: bool = True) -> Tuple[ndarray, List]:
    ...


def stack_images_grid(images: Iterable[ndarray],
                      chunksize: int | None = None,
                      axis: int = 0,
                      overlap: int = 0,
                      pad: int | None = None,
                      return_info: bool = False,
                      bg_value: Number | ndarray | str | None = None,
                      resize: int | str | None = None,
                      allow_casting: bool = True) -> Tuple[ndarray, List]:
    ...
