from numpy import ndarray
from _typeshed import Incomplete


def radial_fourier_mask(img_hwc: ndarray,
                        radius: int = ...,
                        axis: Incomplete | None = ...,
                        clip: Incomplete | None = ...):
    ...


def fourier_mask(img_hwc: ndarray,
                 mask: ndarray,
                 axis: Incomplete | None = ...,
                 clip: Incomplete | None = ...):
    ...
