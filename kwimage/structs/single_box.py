import numpy as np
import ubelt as ub


# TODO: Perhaps dynamically fix the wraped method signatures to agree with
# Boxes when possible. OR just make this efficient (perhaps that is a fools
# errand because single box objects wont be efficient in Python). Probably just
# maintain the wrapper.


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
        data_repr = repr(self.data.tolist())
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
        """
        Example:
            >>> import kwimage
            >>> slice_ = kwimage.Box.random().scale(10).quantize().to_slice()
            >>> new = kwimage.Box.from_slice(slice_)
        """
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
    def from_data(self, data, format):
        import kwimage
        boxes = kwimage.Boxes([data], format)
        self = Box(boxes, _check=False)
        return self

    @classmethod
    def coerce(cls, data, format=None, **kwargs):
        if isinstance(data, Box):
            return data
        else:
            import numbers
            import sys
            torch = sys.modules.get('torch', None)
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

    def intersection(self, other):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Box.coerce([0, 0, 10, 10], 'xywh')
            >>> other = kwimage.Box.coerce([3, 3, 10, 10], 'xywh')
            >>> print(str(self.intersection(other)))
            <Box(ltrb, [3, 3, 10, 10])>
        """
        new_boxes = self.boxes.intersection(other.boxes)
        new = self.__class__(new_boxes)
        return new

    def union_hull(self, other):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Box.coerce([0, 0, 10, 10], 'xywh')
            >>> other = kwimage.Box.coerce([3, 3, 10, 10], 'xywh')
            >>> print(str(self.union_hull(other)))
            <Box(ltrb, [0, 0, 13, 13])>
        """
        new_boxes = self.boxes.union_hull(other.boxes)
        new = self.__class__(new_boxes)
        return new

    def to_ltrb(self, *args, **kwargs):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Box.random().to_ltrb()
            >>> assert self.format == 'ltrb'
        """
        return self.__class__(self.boxes.to_ltrb(*args, **kwargs))

    def to_xywh(self, *args, **kwargs):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Box.random().to_xywh()
            >>> assert self.format == 'xywh'
        """
        return self.__class__(self.boxes.to_xywh(*args, **kwargs))

    def to_cxywh(self, *args, **kwargs):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Box.random().to_cxywh()
            >>> assert self.format == 'cxywh'
        """
        return self.__class__(self.boxes.to_cxywh(*args, **kwargs))

    def toformat(self, *args, **kwargs):
        return self.__class__(self.boxes.toformat(*args, **kwargs))

    def astype(self, *args, **kwargs):
        return self.__class__(self.boxes.astype(*args, **kwargs))

    def corners(self, *args, **kwargs):
        """
        Example:
            >>> import kwimage
            >>> assert kwimage.Box.random().corners().shape == (4, 2)
        """
        return self.boxes.corners(*args, **kwargs)

    def to_boxes(self):
        """
        Returns the underlying "kwimage.Boxes" data structure.
        """
        return self.boxes

    @property
    def aspect_ratio(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().aspect_ratio)
        """
        return self.boxes.aspect_ratio.ravel()[0]

    @property
    def center(self):
        """
        Example:
            >>> import kwimage
            >>> assert len(kwimage.Box.random().center) == 2
        """
        xs, ys = self.boxes.center
        return xs.ravel()[0], ys.ravel()[0]

    @property
    def center_x(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().center_x)
        """
        return self.boxes.center_x.ravel()[0]

    @property
    def center_y(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().center_y)
        """
        return self.boxes.center_y.ravel()[0]

    @property
    def width(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().width)
        """
        return self.boxes.width.ravel()[0]

    @property
    def height(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().height)
        """
        return self.boxes.height.ravel()[0]

    @property
    def tl_x(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().tl_x)
        """
        return self.boxes.tl_x.ravel()[0]

    @property
    def tl_y(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().tl_y)
        """
        return self.boxes.tl_y.ravel()[0]

    @property
    def br_x(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().br_y)
        """
        return self.boxes.br_x.ravel()[0]

    @property
    def br_y(self):
        """
        Example:
            >>> import kwimage
            >>> assert not ub.iterable(kwimage.Box.random().br_y)
        """
        return self.boxes.br_y.ravel()[0]

    @property
    def dtype(self):
        return self.boxes.dtype

    @property
    def area(self):
        return self.boxes.area.ravel()[0]

    def to_slice(self, endpoint=True):
        """
        Example:
            >>> import kwimage
            >>> kwimage.Box.random(rng=0).scale(10).quantize().to_slice()
            (slice(5, 8, None), slice(5, 7, None))
        """
        return self.boxes.to_slices(endpoint=endpoint)[0]

    def to_shapely(self):
        """
        Example:
            >>> import kwimage
            >>> kwimage.Box.random().to_shapely()
        """
        return self.boxes.to_shapely()[0]

    def to_polygon(self):
        """
        Example:
            >>> import kwimage
            >>> kwimage.Box.random().to_polygon()
        """
        return self.boxes.to_polygons()[0]

    def to_coco(self):
        """
        Example:
            >>> import kwimage
            >>> kwimage.Box.random().to_coco()
        """
        return list(self.boxes.to_coco())[0]

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


def _transfer_docstrings():
    """
    Helper to populate Box docstrings from Boxes. In the future we should may
    want to autogenerate better docstrings statically, but for now lets at
    least add some introspection.
    """
    from kwimage.structs import Boxes
    import types
    for k in dir(Box):
        if not k.startswith('_') and hasattr(Boxes, k):
            v1 = getattr(Box, k)
            if isinstance(v1, types.MethodType):
                v1 = v1.__func__
            if isinstance(v1, property):
                v1 = v1.fget
            if hasattr(v1, '__doc__') and v1.__doc__ is None:
                v2 = getattr(Boxes, k)
                if isinstance(v2, property):
                    v2 = v2.fget
                if v2.__doc__ is not None:
                    v1.__doc__ = (
                        '\n        This function wraps one of the same name in '
                        'kwimage.Boxes, but does not have a docstring of its own. '
                        'In the meantime we will show the docstring from Boxes\n'
                    ) + v2.__doc__

if 0:
    _transfer_docstrings()
