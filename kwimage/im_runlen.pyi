from numpy import ndarray
from typing import Dict
from typing import Tuple


def encode_run_length(img: ndarray,
                      binary: bool = False,
                      order: str = 'C') -> Dict[str, object]:
    ...


def decode_run_length(counts: ndarray,
                      shape: Tuple[int, int],
                      binary: bool = False,
                      dtype: type = ...,
                      order: str = 'C') -> ndarray:
    ...


def rle_translate(rle: dict,
                  offset: Tuple[int, int],
                  output_shape: Tuple[int, int] = None):
    ...
