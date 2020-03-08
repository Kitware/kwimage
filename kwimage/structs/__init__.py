"""
mkinit ~/code/kwimage/kwimage/structs/__init__.py -w --relative --nomod


A common thread in many kwimage.structs / kwannot objects is that they attempt
to store multiple data elements using a single data structure when possible
e.g. the classes are Boxes, Points, Detections, Coords, and not Box, Detection,
Coord. The exceptions are Polygon, Heatmap, and Mask, where it made more sense
to have one object-per item because each individual item is a reasonably sized
chuck of data.

Another commonality is that objects have only two main attributes: .data and
.meta. These allow the underlying representation of the object to vary as
needed.

Currently Boxes and Mask do not have a .meta attribute. They instead have a
.format attribute which is a text-code indicating the underlying layout of the
data.

The `data` and `meta` instance attributes in the Points, Detections, and
Heatmaps classes are dictionaries.  These classes also have a `__datakeys__`
and `__metakeys__` class attribute, which are lists of strings. These lists
specify which keys are expected in each dictionary. For instance,
`Points.__datakeys__ = ['xy', 'class_idxs', 'visible']` and
`Points.__metakeys__ = ['classes']`. All objects in the data dictionary are
expected to be aligned, whereas the meta dictionary is for auxillay data. For
example in Points, the xy position data['xy'][i] is expected to have the class
index data['class_idxs'][i]. By convention, a class index indexes into the list
of category names stored in `meta['classes']`.


The Heatmap.data behaves slighly different than Points. Its `data` dictionary
stores different per-pixel attributes like class probability scores, or offset
vectors. The `meta` dictionary stores data like the originaly image dimensions
(heatmaps are usually downsampled wrt the image that they correspond to) and
the transformation matrices would warp the "data" space back onto the original
image space.

Note that the developer can add any extra data or meta keys that they like, but
they should keep in mind that all items in `data` should be aligned, whereas
`meta` can contain arbitrary information.

"""
from .boxes import (Boxes,)
from .coords import (Coords,)
from .detections import (Detections,)
from .heatmap import (Heatmap, smooth_prob,)
from .mask import (Mask, MaskList,)
from .points import (Points, PointsList,)
from .polygon import (MultiPolygon, Polygon, PolygonList,)
from .segmentation import (Segmentation, SegmentationList,)

__all__ = ['Boxes', 'Coords', 'Detections', 'Heatmap', 'Mask', 'MaskList',
           'MultiPolygon', 'Points', 'PointsList', 'Polygon', 'PolygonList',
           'Segmentation', 'SegmentationList', 'smooth_prob']
