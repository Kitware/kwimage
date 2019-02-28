"""
mkinit ~/code/kwimage/kwimage/structs/__init__.py -w --relative
"""
from .boxes import (Boxes,)
from .detections import (Detections,)
from .heatmap import (Heatmap, smooth_prob,)
from .mask import (Mask, Masks,)

__all__ = ['Boxes', 'Detections', 'Heatmap', 'Mask', 'Masks', 'smooth_prob']
