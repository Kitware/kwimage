from numpy import ndarray
from typing import Tuple
from typing import Sequence
import kwcoco
import kwimage
from typing import List
from nptyping import Float32
from typing import Any
import numpy as np
from numbers import Number
from _typeshed import Incomplete


def draw_text_on_image(img: ndarray | None | dict,
                       text: str,
                       org: Tuple[int, int] | None = None,
                       return_info: bool = False,
                       **kwargs) -> ndarray | Tuple[ndarray, dict]:
    ...


def draw_clf_on_image(im: ndarray,
                      classes: Sequence[str] | kwcoco.CategoryTree,
                      tcx: int | None = None,
                      probs: ndarray | None = None,
                      pcx: int | None = None,
                      border: int = ...):
    ...


def draw_boxes_on_image(img: ndarray,
                        boxes: kwimage.Boxes | ndarray,
                        color: str = ...,
                        thickness: int = ...,
                        box_format: Incomplete | None = ...,
                        colorspace: str = 'rgb'):
    ...


def draw_line_segments_on_image(img,
                                pts1: ndarray,
                                pts2: ndarray,
                                color: str | List = 'blue',
                                colorspace: str = 'rgb',
                                thickness: int = 1,
                                **kwargs) -> ndarray:
    ...


def make_heatmask(probs: ndarray,
                  cmap: str = 'plasma',
                  with_alpha: float = 1.0,
                  space: str = 'rgb',
                  dsize: tuple | None = None):
    ...


def make_orimask(radians: ndarray,
                 mag: ndarray | None = None,
                 alpha: float | ndarray = 1.0) -> ndarray[Any, Float32]:
    ...


def make_vector_field(dx: ndarray,
                      dy: ndarray,
                      stride: int | float = 0.02,
                      thresh: float = 0.0,
                      scale: float = 1.0,
                      alpha: float = 1.0,
                      color: str | tuple | kwimage.Color = 'strawberry',
                      thickness: int = 1,
                      tipLength: float = 0.1,
                      line_type: int | str = 'aa') -> ndarray[Any, Float32]:
    ...


def draw_vector_field(image: ndarray,
                      dx: ndarray,
                      dy: ndarray,
                      stride: int | float = 0.02,
                      thresh: float = 0.0,
                      scale: float = 1.0,
                      alpha: float = 1.0,
                      color: str | tuple | kwimage.Color = 'strawberry',
                      thickness: int = 1,
                      tipLength: float = 0.1,
                      line_type: int | str = 'aa') -> ndarray[Any, Float32]:
    ...


def draw_header_text(image: ndarray | dict | None = None,
                     text: str | None = None,
                     fit: bool | str = False,
                     color: str | Tuple = 'strawberry',
                     halign: str = 'center',
                     stack: bool | str = 'auto',
                     bg_color: str = ...,
                     **kwargs) -> ndarray:
    ...


def fill_nans_with_checkers(canvas: np.ndarray,
                            square_shape: int | Tuple[int, int] | str = 8,
                            on_value: Number | str = 'auto',
                            off_value: Number | str = 'auto') -> np.ndarray:
    ...


def nodata_checkerboard(canvas: ndarray,
                        square_shape: int = 8,
                        on_value: Number | str = 'auto',
                        off_value: Number | str = 'auto') -> ndarray:
    ...
