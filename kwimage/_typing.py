# Helpers for typing

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, TypeAlias
from numpy.typing import ArrayLike
from skimage.transform import _geometric
from kwimage.transform import Transform
if TYPE_CHECKING:
    from typing import Any

SKImageGeometricTransform: Any = getattr(_geometric, '_GeometricTransform', None)
if SKImageGeometricTransform is None:
    # Older version compatability
    SKImageGeometricTransform = getattr(_geometric, 'GeometricTransform')

TransformLike: TypeAlias = SKImageGeometricTransform | ArrayLike | Callable | Transform

__all__ = [
    'SKImageGeometricTransform',
    'TransformLike',
]
