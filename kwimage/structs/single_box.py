import numpy as np
import ubelt as ub


class Box(ub.NiceRepr):
    """
    Represents a single Box.

    This is a convinience class. For multiple boxes use kwimage.Boxes, which is
    more efficient.

    Currently implemented by storing a Boxes object with one item and indexing
    into it. This implementation could be done more efficiently.

    SeeAlso:
        :class:`kwimage.structs.boxes.Boxes`

    Example:
        >>> from kwimage.structs.single_box import *  # NOQA
        >>> box = Box.random()
        >>> print(f'box={box}')
        >>> #
        >>> box.scale(10).quantize().to_slice()
        >>> #
        >>> sl = (slice(0, 10), slice(0, 30))
        >>> box = Box.from_slice(sl)
        >>> print(f'box={box}')
    """

    def __init__(self, boxes, _check: bool = False):
        if _check:
            raise Exception(
                'For now, only construct an instance of this using a class '
                ' method, like coerce, from_slice, from_shapely, etc...')
        self.boxes = boxes

    @property
    def format(self):
        return self.boxes.format

    @property
    def data(self):
        return self.boxes.data[0]

    def __nice__(self):
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        nice = '{}, {}'.format(self.format, data_repr)
        return nice

    @classmethod
    def random(self, *args, **kwargs):
        import kwimage
        boxes = kwimage.Boxes.random(*args, **kwargs)
        self = Box(boxes, _check=False)
        return self

    @classmethod
    def from_slice(self, slice_):
        import kwimage
        boxes = kwimage.Boxes.from_slice(slice_)
        self = Box(boxes, _check=False)
        return self

    @classmethod
    def from_shapely(self, geom):
        import kwimage
        boxes = kwimage.Boxes.from_shapely(geom)
        self = Box(boxes, _check=False)
        return self

    @classmethod
    def from_dsize(self, dsize):
        width, height = dsize
        import kwimage
        boxes = kwimage.Boxes([[0, 0, width, height]], 'ltrb')
        self = Box(boxes, _check=False)
        return self

    @classmethod
    def coerce(cls, data, **kwargs):
        if isinstance(data, Box):
            return data
        else:
            import numbers
            # import kwimage
            from kwarray.arrayapi import torch
            if isinstance(data, list):
                if data and isinstance(data[0], numbers.Number):
                    data = np.array(data)[None, :]
            if isinstance(data, np.ndarray) or torch and torch.is_tensor(data):
                if len(data.shape) == 1:
                    data = data[None, :]
            # return cls(kwimage.Boxes.coerce(data, **kwargs))
            # inline new coerce code until new version lands
            from kwimage import Boxes
            from shapely.geometry import Polygon
            if isinstance(data, Boxes):
                self = data
            elif isinstance(data, Polygon):
                self = Boxes.from_shapely(data)
            else:
                _arr_data = None
                if isinstance(data, np.ndarray):
                    _arr_data = np.array(data)
                elif isinstance(data, list):
                    _arr_data = np.array(data)

                if _arr_data is not None:
                    format = kwargs.get('format', None)
                    if format is None:
                        raise Exception('ambiguous, specify Box format')
                    self = Boxes(_arr_data, format=format)
                else:
                    raise NotImplementedError
            return cls(self)

    @property
    def dsize(self):
        return (int(self.width), int(self.height))

    def translate(self, *args, **kwargs):
        new_boxes = self.boxes.translate(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def warp(self, *args, **kwargs):
        new_boxes = self.boxes.warp(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def scale(self, *args, **kwargs):
        new_boxes = self.boxes.scale(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def clip(self, *args, **kwargs):
        new_boxes = self.boxes.clip(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def quantize(self, *args, **kwargs):
        new_boxes = self.boxes.quantize(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def copy(self, *args, **kwargs):
        new_boxes = self.boxes.copy(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def round(self, *args, **kwargs):
        new_boxes = self.boxes.round(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def pad(self, *args, **kwargs):
        new_boxes = self.boxes.pad(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def resize(self, *args, **kwargs):
        new_boxes = self.boxes.resize(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def to_ltbr(self, *args, **kwargs):
        return self.__class__(self.boxes.to_ltbr(*args, **kwargs))

    def to_xywh(self, *args, **kwargs):
        return self.__class__(self.boxes.to_xywh(*args, **kwargs))

    def to_cxywh(self, *args, **kwargs):
        return self.__class__(self.boxes.to_cxywh(*args, **kwargs))

    def toformat(self, *args, **kwargs):
        return self.__class__(self.boxes.toformat(*args, **kwargs))

    def astype(self, *args, **kwargs):
        return self.__class__(self.boxes.astype(*args, **kwargs))

    def corners(self, *args, **kwargs):
        return self.boxes.corners(*args, **kwargs)[0]

    def to_boxes(self):
        return self.boxes

    @property
    def width(self):
        return self.boxes.width.ravel()[0]

    @property
    def aspect_ratio(self):
        return self.boxes.aspect_ratio.ravel()[0]

    @property
    def height(self):
        return self.boxes.height.ravel()[0]

    @property
    def tl_x(self):
        return self.boxes.tl_x[0]

    @property
    def tl_y(self):
        return self.boxes.tl_y[0]

    @property
    def br_x(self):
        return self.boxes.br_x[0]

    @property
    def br_y(self):
        return self.boxes.br_y[0]

    @property
    def dtype(self):
        return self.boxes.dtype

    @property
    def area(self):
        return self.boxes.area.ravel()[0]

    def to_slice(self, endpoint=True):
        return self.boxes.to_slices(endpoint=endpoint)[0]

    def to_shapely(self):
        return self.boxes.to_shapely()[0]

    def to_polygon(self):
        return self.boxes.to_polygons()[0]

    def to_coco(self):
        return self.boxes.to_coco()[0]
