# Helpers for typing

from __future__ import annotations
from typing import TYPE_CHECKING
from skimage.transform import _geometric
if TYPE_CHECKING:
    from typing import Any

SKImageGeometricTransform: Any = getattr(_geometric, '_GeometricTransform', None)
if SKImageGeometricTransform is None:
    # Older version compatability
    SKImageGeometricTransform = getattr(_geometric, 'GeometricTransform')


__all__ = [
    'SKImageGeometricTransform'
]
