"""
mkinit ~/code/kwimage/kwimage/structs/__init__.py -w --relative
"""
from . import boxes
from . import detections
from . import heatmap

from .boxes import (Boxes,)
from .detections import (Detections,)
from .heatmap import (Heatmap, smooth_prob,)

__all__ = ['Boxes', 'Detections', 'Heatmap', 'boxes', 'detections', 'heatmap',
           'smooth_prob']
