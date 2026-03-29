# Helpers for typing

from __future__ import annotations

from typing import TYPE_CHECKING

from skimage.transform import _geometric

if TYPE_CHECKING:
    from typing import Any, Callable

    from numpy.typing import ArrayLike

    from kwimage.transform import Transform

SKImageGeometricTransform: Any = getattr(
    _geometric, '_GeometricTransform', None
)
if SKImageGeometricTransform is None:
    # Older version compatability
    SKImageGeometricTransform = getattr(_geometric, 'GeometricTransform')

if TYPE_CHECKING:
    TransformLike = SKImageGeometricTransform | ArrayLike | Callable | Transform

__all__ = [
    'SKImageGeometricTransform',
    'TransformLike',
]
