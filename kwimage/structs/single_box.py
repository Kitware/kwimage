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
    def random(self, **kwargs):
        import kwimage
        if kwargs.get('num', 1) != 1:
            raise ValueError('Cannot specify num for Box. Use Boxes instead.')
        kwargs['num'] = 1
        boxes = kwimage.Boxes.random(**kwargs)
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

    def draw_on(self, image=None, color='blue', alpha=None, label=None,
                copy=False, thickness=2, label_loc='top_left'):
        """
        Draws a box directly on an image using OpenCV

        Example:
            >>> import kwimage
            >>> self = kwimage.Box.random(scale=256, rng=10, format='ltrb')
            >>> canvas = np.zeros((256, 256, 3), dtype=np.uint8)
            >>> image = self.draw_on(canvas)
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> kwplot.show_if_requested()
        """
        return self.boxes.draw_on(image=image, color=color, alpha=alpha,
                                  labels=[label], copy=copy,
                                  thickness=thickness, label_loc=label_loc)

    def draw(self, color='blue', alpha=None, label=None, centers=False,
             fill=False, lw=2, ax=None, setlim=False, **kwargs):
        """
        Draws a box directly on an image using OpenCV

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwimage
            >>> self = kwimage.Box.random(scale=512.0, rng=0, format='ltrb')
            >>> self.translate((-128, -128), inplace=True)
            >>> #image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> #kwplot.imshow(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(color='blue', setlim=1.2)
            >>> # xdoc: +REQUIRES(--show)
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()
        """
        if label is None:
            labels = None
        else:
            labels = [label]
        return self.boxes.draw(color=color, alpha=alpha, labels=labels,
                               centers=centers, fill=fill, lw=lw, ax=ax,
                               setlim=setlim, **kwargs)
