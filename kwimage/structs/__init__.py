"""
mkinit ~/code/kwimage/kwimage/structs/__init__.py -w --relative
"""
from . import boxes
from . import detections
from . import heatmap
from . import mask

from .boxes import (Boxes,)
from .detections import (Detections,)
from .heatmap import (Heatmap, smooth_prob,)
from .mask import (Mask,)

__all__ = ['Boxes', 'Detections', 'Heatmap', 'Mask', 'boxes', 'detections',
           'heatmap', 'mask', 'smooth_prob']
