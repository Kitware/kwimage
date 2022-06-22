from typing import Union
from numpy import ndarray
from typing import Tuple
from typing import Sequence
import kwcoco
import kwimage
from typing import List
from typing import Any
from nptyping import Float32
import numpy as np
import numpy as np
from _typeshed import Incomplete


def draw_text_on_image(img: Union[ndarray, None, dict],
                       text: str,
                       org: Tuple[int, int] = None,
                       return_info: bool = False,
                       **kwargs) -> ndarray:
    ...


def draw_clf_on_image(im: ndarray,
                      classes: Union[Sequence[str], kwcoco.CategoryTree],
                      tcx: int = None,
                      probs: ndarray = None,
                      pcx: int = None,
                      border: int = ...):
    ...


def draw_boxes_on_image(img: ndarray,
                        boxes: Union[kwimage.Boxes, ndarray],
                        color: str = ...,
                        thickness: int = ...,
                        box_format: Incomplete | None = ...,
                        colorspace: str = 'rgb'):
    ...


def draw_line_segments_on_image(img,
                                pts1: ndarray,
                                pts2: ndarray,
                                color: Union[str, List] = 'blue',
                                colorspace: str = 'rgb',
                                thickness: int = 1,
                                **kwargs) -> ndarray:
    ...


def make_heatmask(probs: ndarray,
                  cmap: str = 'plasma',
                  with_alpha: float = 1.0,
                  space: str = 'rgb',
                  dsize: tuple = None):
    ...


def make_orimask(radians: ndarray,
                 mag: ndarray = None,
                 alpha: Union[float, ndarray] = 1.0) -> ndarray[Any, Float32]:
    ...


def make_vector_field(
        dx: ndarray,
        dy: ndarray,
        stride: Union[int, float] = 0.02,
        thresh: float = 0.0,
        scale: float = 1.0,
        alpha: float = 1.0,
        color: Union[str, tuple, kwimage.Color] = 'strawberry',
        thickness: int = 1,
        tipLength: float = 0.1,
        line_type: Union[int, str] = 'aa') -> ndarray[Any, Float32]:
    ...


def draw_vector_field(
        image: ndarray,
        dx: ndarray,
        dy: ndarray,
        stride: Union[int, float] = 0.02,
        thresh: float = 0.0,
        scale: float = 1.0,
        alpha: float = 1.0,
        color: Union[str, tuple, kwimage.Color] = 'strawberry',
        thickness: int = 1,
        tipLength: float = 0.1,
        line_type: Union[int, str] = 'aa') -> ndarray[Any, Float32]:
    ...


def draw_header_text(image: Union[ndarray, dict, None],
                     text: str,
                     fit: Union[bool, str] = False,
                     color: Union[str, Tuple] = 'strawberry',
                     halign: str = 'center',
                     stack: Union[bool, str] = 'auto') -> ndarray:
    ...


def fill_nans_with_checkers(canvas: np.ndarray,
                            square_shape: int = ...) -> np.ndarray:
    ...


def nodata_checkerboard(canvas, square_shape: int = ...):
    ...
