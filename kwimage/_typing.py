# Helpers for typing

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from skimage.transform import _geometric

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Callable

    from numpy import ndarray
    from numpy.typing import ArrayLike
    import torch
    from torch import Tensor
    from skimage.transform._geometric import _GeometricTransform

    from kwimage.transform import Transform

SKImageGeometricTransform: Any = getattr(
    _geometric, '_GeometricTransform', None
)
if SKImageGeometricTransform is None:
    # Older version compatability
    SKImageGeometricTransform = getattr(_geometric, 'GeometricTransform')

if TYPE_CHECKING:
    ArrayData = ndarray | Tensor
    TorchDeviceLike = str | int | torch.device | None

    class ImgAugKeypoint(Protocol):
        x: float
        y: float

    class ImgAugKeypointsOnImage(Protocol):
        keypoints: Sequence[ImgAugKeypoint]

        def to_xy_array(self) -> ndarray: ...

    class ImgAugBoundingBox(Protocol):
        x1: float
        y1: float
        x2: float
        y2: float

    class ImgAugBoundingBoxesOnImage(Protocol):
        bounding_boxes: Sequence[ImgAugBoundingBox]

    class ImgAugAugmenter(Protocol):
        def augment_keypoints(
            self, keypoints: ImgAugKeypointsOnImage
        ) -> ImgAugKeypointsOnImage: ...

    TransformCallable = Callable[[ArrayData], ArrayData]
    TransformLike = (
        _GeometricTransform
        | ArrayLike
        | TransformCallable
        | Transform
        | ImgAugAugmenter
        | None
    )

__all__ = [
    'SKImageGeometricTransform',
]
