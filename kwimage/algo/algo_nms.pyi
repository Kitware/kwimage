from numpy import ndarray
from typing import Tuple
from nptyping import Float32
from typing import Any
from nptyping import Int64
from nptyping import Shape
from _typeshed import Incomplete


def daq_spatial_nms(ltrb: ndarray,
                    scores: ndarray,
                    diameter: int | Tuple[int, int],
                    thresh: float,
                    max_depth: int = 6,
                    stop_size: int = 2048,
                    recsize: int = 2048,
                    impl: str = 'auto',
                    device_id: Incomplete | None = ...):
    ...


class _NMS_Impls:

    def __init__(self) -> None:
        ...


def available_nms_impls():
    ...


def non_max_supression(ltrb: ndarray[Any, Float32],
                       scores: ndarray[Any, Float32],
                       thresh: float,
                       bias: float = 0.0,
                       classes: ndarray[Shape['*'], Int64] | None = None,
                       impl: str = 'auto',
                       device_id: int | None = None):
    ...
