# Helpers for typing

import skimage.transform import _geometric

try:
    SKImageGeometricTransform = _geometric._GeometricTransform
except AttributeError:
    # Older version compatability
    SKImageGeometricTransform = _geometric.GeometricTransform


__all__ = [
    'SKImageGeometricTransform'
]
