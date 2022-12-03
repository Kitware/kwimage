from numpy import ndarray
from typing import Union
from os import PathLike
from _typeshed import Incomplete

JPG_EXTENSIONS: Incomplete
GDAL_EXTENSIONS: Incomplete
ITK_EXTENSIONS: Incomplete
IMAGE_EXTENSIONS: Incomplete


def imread(fpath: str,
           space: str = 'auto',
           backend: str = 'auto',
           **kw) -> ndarray:
    ...


def imwrite(fpath: PathLike,
            image: ndarray,
            space: Union[str, None] = 'auto',
            backend: str = 'auto',
            **kwargs) -> str:
    ...


def load_image_shape(fpath: str, backend: str = 'auto'):
    ...
