from typing import Sequence
from numpy import ndarray
import numpy as np
from typing import Union
import numpy as np


def overlay_alpha_layers(layers: Sequence[ndarray],
                         keepalpha: bool = True,
                         dtype: np.dtype = ...) -> ndarray:
    ...


def overlay_alpha_images(img1: ndarray,
                         img2: ndarray,
                         keepalpha: bool = True,
                         dtype: np.dtype = ...,
                         impl: str = 'inplace') -> ndarray:
    ...


def ensure_alpha_channel(img: ndarray,
                         alpha: Union[float, ndarray] = 1.0,
                         dtype: type = ...,
                         copy: bool = False) -> ndarray:
    ...
