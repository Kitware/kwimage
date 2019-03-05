"""
mkinit ~/code/kwimage/kwimage/structs/__init__.py -w --relative --nomod
"""
from .boxes import (Boxes,)
from .detections import (Detections,)
from .heatmap import (Heatmap, smooth_prob,)
from .mask import (Mask, MaskList,)
from .points import (Points, PointsList,)

__all__ = ['Boxes', 'Detections', 'Heatmap', 'Mask', 'MaskList', 'Points',
           'PointsList', 'smooth_prob']
