"""
Vectorized Bounding Boxes

:class:`kwimage.Boxes` is a tool for efficiently transporting a set of bounding
boxes within python as well as methods for operating on bounding boxes. It is a
VERY thin wrapper around a pure numpy/torch array/tensor representation, and
thus it is very fast.

Raw bounding boxes come in lots of different formats. There are lots of ways to
parameterize two points! Because of this THE USER MUST ALWAYS BE EXPLICIT ABOUT
THE BOX FORMAT.

There are 3 main bounding box formats:

    * xywh: top left xy-coordinates and width height offsets

    * cxywh: center xy-coordinates and width height offsets

    * ltrb: top left and bottom right xy coordinates

Here is some example usage

Example:
    >>> import kwimage
    >>> data = np.array([[ 0,  0, 10, 10],
    >>>                  [ 5,  5, 50, 50],
    >>>                  [10,  0, 20, 10],
    >>>                  [20,  0, 30, 10]])
    >>> # Note that the format of raw data is ambiguous, so you must specify
    >>> boxes = kwimage.Boxes(data, 'ltrb')
    >>> print('boxes = {!r}'.format(boxes))
    boxes = <Boxes(ltrb,
        array([[ 0,  0, 10, 10],
               [ 5,  5, 50, 50],
               [10,  0, 20, 10],
               [20,  0, 30, 10]]))>

    >>> # Now you can operate on those boxes easily
    >>> print(boxes.translate((10, 10)))
    <Boxes(ltrb,
        array([[10., 10., 20., 20.],
               [15., 15., 60., 60.],
               [20., 10., 30., 20.],
               [30., 10., 40., 20.]]))>
    >>> print(boxes.to_cxywh())
    <Boxes(cxywh,
        array([[ 5. ,  5. , 10. , 10. ],
               [27.5, 27.5, 45. , 45. ],
               [15. ,  5. , 10. , 10. ],
               [25. ,  5. , 10. , 10. ]]))>
    >>> print(ub.repr2(boxes.ious(boxes), precision=2, with_dtype=False))
    np.array([[1.  , 0.01, 0.  , 0.  ],
              [0.01, 1.  , 0.02, 0.02],
              [0.  , 0.02, 1.  , 0.  ],
              [0.  , 0.02, 0.  , 1.  ]])
    >>> # OpenCV and Matplotlib have first class visualization support
    >>> # xdoc: +REQUIRES(--show)
    >>> # xdoc: +REQUIRES(module:kwplot)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> # opencv "draw_on" method
    >>> background = kwimage.checkerboard(dsize=(64, 64), dtype=np.uint8, on_value='kw_green', off_value='kw_blue')
    >>> canvas = background.copy()
    >>> boxes.draw_on(canvas, color='kw_red')
    >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1), doclf=1, title='[cv2] kwimage.Boxes.draw_on')
    >>> # matplotlib "draw_on" method
    >>> kwplot.imshow(background, fnum=1, pnum=(1, 2, 2), title='[mpl] kwimage.Boxes.draw')
    >>> boxes.draw(color='kw_red')
    >>> plt.gcf().suptitle('Matplotlib and OpenCV have first class visualization support')
    >>> kwplot.show_if_requested()


SeeAlso:
    :class:`kwimage.structs.single_box.Box`
"""
import numpy as np
import ubelt as ub
import warnings
import skimage
import kwarray
import numbers
from kwimage.structs import _generic  # NOQA
from kwimage import _internal

try:
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion


try:
    import torch
except Exception:
    torch = None
    _TORCH_HAS_EMPTY_SHAPE = None
    _TORCH_HAS_BOOL_COMP = None
else:
    _TORCH_HAS_EMPTY_SHAPE = LooseVersion(torch.__version__) >= LooseVersion('1.0.0')
    _TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')

__all__ = ['Boxes']

if not _internal.KWIMAGE_DISABLE_C_EXTENSIONS:
    try:
        from kwimage_ext.structs._boxes_backend.cython_boxes import bbox_ious_c as _bbox_ious_c
    except ImportError:
        _bbox_ious_c = None
    except Exception as ex:
        _bbox_ious_c = None
        warnings.warn(
            'Optional cython_boxes backend is not available: {!r}'.format(ex))
else:
    _bbox_ious_c = None


class NeedsWarpCorners(AssertionError):
    pass


class BoxFormat:
    """
    Defines valid Box formats and their aliases.

    Attrs:
        aliases (Mapping[str, str]):
            maps format aliases to their cannonical name.

    See module level docstring for format definitions
    """
    # xywh = 'xywh'
    # cxywh = 'cxywh'
    # xy1xy2 = 'ltrb'
    # xx1yy2 = 'extent'

    # Note: keep the strings as the "old"-style names for now
    # TODO: change the string values to match their associated NameConstant
    #     - [x] bump versions
    #     - [x] use the later in the or statements
    #     - [ ] ensure nothing depends on the old values.
    #     - [x] Change cannonical TLBR to LTRB

    # Definitions:
    #     x1 = top-left-x
    #     y1 = top-left-y
    #     x2 = bottom-right-x
    #     y2 = bottom-right-y
    #      w = box width
    #      h = box height
    #     cx = center-x
    #     cy = center-y
    #      r = row = y-coordinate
    #      c = column = x-coordinate

    cannonical = []

    def _register(k, cannonical=cannonical):
        cannonical.append(k)
        return k

    # Column-Major-Formats
    XYWH  = _register('xywh')   # (x1, y1, w, h)
    CXYWH = _register('cxywh')  # (cx, cy, w, h)
    LTRB  = _register('ltrb')   # (x1, y1, x2, y2)
    TLBR  = LTRB  # deprecated, but kept for backwards compatability
    XXYY  = _register('xxyy')   # (x1, x2, y1, y2)

    # Row-Major-Formats
    # Note: prefix row major format with an underscore.
    # Reason: Boxes prefers column-major formats
    _YYXX  = _register('_yyxx')   # (y1, y2, x1, x2)
    _RCHW  = _register('_rchw')   # (y1, y2, h, w)

    aliases = {
        # NOTE: Once a name enters here it is very difficult to remove
        # Once we hit version 1.0, this table cannot be removed from without
        # bumping a major version.
        'xywh': XYWH,
        'ltwh': XYWH,  # left-top-width-height

        'cxywh': CXYWH,

        'tlbr': LTRB,  # note tlbr is a confusing code, its actually LTRB. For legacy reasons we are retaining it for now.
        'ltrb': LTRB,  # left-top-right-bottom
        'xyxy': LTRB,
        'xy1xy2': LTRB,

        'xxyy': XXYY,
        'xx1yy2': XXYY,

        # Transposed formats
        '_yyxx': _YYXX,
        '_yy1xx2': _YYXX,

        # Explicit tuple format
        'cx,cy,w,h'           : CXYWH,
        'tl_x,tl_y,w,h'       : XYWH,
        'tl_x,tl_y,br_x,br_y' : LTRB,
        'x1,x2,y1,y2'         : XXYY,
    }
    for key in cannonical:
        aliases[key] = key

    # these are old deprecated format codes that were once used and were
    # removed due to inconsistent implementations. Thus we should not re-use
    # then in the future (maybe unless there is a major version bump).
    blocklist = [
        'tlhw',
        # 'tlbr', # eventually
    ]


def box_ious(ltrb1, ltrb2, bias=0, impl=None):
    """
    Args:
        ltrb1 (ndarray): (N, 4) ltrb format
        ltrb2 (ndarray): (K, 4) ltrb format
        bias (int): either 0 or 1, does tl=br have area of 0 or 1?
        impl (str | None): Can be "auto", "torch", "py", or "c".
            "c" requires :mod:`kwimage_ext`.

    Benchmark:
        See ~/code/kwarray/dev/bench_bbox.py

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> ltrb1 = Boxes.random(5, scale=10.0, rng=0, format='ltrb').data
        >>> ltrb2 = Boxes.random(7, scale=10.0, rng=1, format='ltrb').data
        >>> ious = box_ious(ltrb1, ltrb2)
        >>> print(ub.repr2(ious.tolist(), precision=2))
        [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.01],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02],
            [0.32, 0.02, 0.01, 0.07, 0.24, 0.12, 0.55],
            [0.00, 0.00, 0.00, 0.11, 0.00, 0.12, 0.04],
        ]

    Example:
        >>> ltrb1 = Boxes.random(5, scale=10.0, rng=0, format='ltrb').data
        >>> ltrb2 = Boxes.random(7, scale=10.0, rng=1, format='ltrb').data
        >>> if _bbox_ious_c is not None:
        >>>     ious_c = box_ious(ltrb1, ltrb2, bias=0, impl='c')
        >>>     ious_py = box_ious(ltrb1, ltrb2, bias=0, impl='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
        >>>     ious_c = box_ious(ltrb1, ltrb2, bias=1, impl='c')
        >>>     ious_py = box_ious(ltrb1, ltrb2, bias=1, impl='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
    """
    if impl is None or impl == 'auto':
        if torch is not None and torch.is_tensor(ltrb1):
            impl = 'torch'
        else:
            impl = 'py' if _bbox_ious_c is None else 'c'

    if impl == 'torch' or (torch is not None and torch.is_tensor(ltrb1)):
        # TODO: add tests for equality with other methods or show why it should
        # be different.
        # NOTE: this is done in boxes.ious
        return _box_ious_torch(ltrb1, ltrb2, bias)
    elif impl == 'c':
        if _bbox_ious_c is None:
            raise Exception('The Boxes C module is not available')
        return _bbox_ious_c(ltrb1.astype(np.float32),
                            ltrb2.astype(np.float32), bias)
    elif impl == 'py':
        return _box_ious_py(ltrb1, ltrb2, bias)
    else:
        raise KeyError(impl)


def _box_ious_torch(ltrb1, ltrb2, bias=0):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> ltrb1 = Boxes.random(5, scale=10.0, rng=0, format='ltrb').tensor().data
        >>> ltrb2 = Boxes.random(7, scale=10.0, rng=1, format='ltrb').tensor().data
        >>> bias = 0
        >>> ious = _box_ious_torch(ltrb1, ltrb2, bias)
        >>> ious_np = _box_ious_py(ltrb1.numpy(), ltrb2.numpy(), bias)
        >>> assert np.all(ious_np == ious.numpy())
    """
    # ltrb1 = ltrb1.view(-1, 4)
    # ltrb2 = ltrb2.view(-1, 4)

    w1 = ltrb1[..., 2] - ltrb1[..., 0] + bias
    h1 = ltrb1[..., 3] - ltrb1[..., 1] + bias
    w2 = ltrb2[..., 2] - ltrb2[..., 0] + bias
    h2 = ltrb2[..., 3] - ltrb2[..., 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = torch.min(ltrb1[..., 2][..., None], ltrb2[..., 2])
    x_mins = torch.max(ltrb1[..., 0][..., None], ltrb2[..., 0])

    iws = (x_maxs - x_mins + bias).clamp(0, float('inf'))

    y_maxs = torch.min(ltrb1[..., 3][..., None], ltrb2[..., 3])
    y_mins = torch.max(ltrb1[..., 1][..., None], ltrb2[..., 1])

    ihs = (y_maxs - y_mins + bias).clamp(0, float('inf'))

    areas_sum = (areas1[..., None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


def _box_ious_py(ltrb1, ltrb2, bias=0):
    """
    This is the fastest python implementation of bbox_ious I found
    """
    w1 = ltrb1[:, 2] - ltrb1[:, 0] + bias
    h1 = ltrb1[:, 3] - ltrb1[:, 1] + bias
    w2 = ltrb2[:, 2] - ltrb2[:, 0] + bias
    h2 = ltrb2[:, 3] - ltrb2[:, 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = np.minimum(ltrb1[:, 2][:, None], ltrb2[:, 2])
    x_mins = np.maximum(ltrb1[:, 0][:, None], ltrb2[:, 0])

    iws = np.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = np.minimum(ltrb1[:, 3][:, None], ltrb2[:, 3])
    y_mins = np.maximum(ltrb1[:, 1][:, None], ltrb2[:, 1])

    ihs = np.maximum(y_maxs - y_mins + bias, 0)

    areas_sum = (areas1[:, None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


def _isect_areas(ltrb1, ltrb2, bias=0, _impl=None):
    """
    Returns only the area of the intersection
    """
    if _impl is None:
        _impl = np
    x_maxs = _impl.minimum(ltrb1[:, 2][:, None], ltrb2[:, 2])
    x_mins = _impl.maximum(ltrb1[:, 0][:, None], ltrb2[:, 0])

    iws = _impl.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = _impl.minimum(ltrb1[:, 3][:, None], ltrb2[:, 3])
    y_mins = _impl.maximum(ltrb1[:, 1][:, None], ltrb2[:, 1])

    ihs = _impl.maximum(y_maxs - y_mins + bias, 0)

    inter_areas = iws * ihs
    return inter_areas


class _BoxConversionMixins(object):
    """
    Methods for converting between different bounding box formats
    """

    convert_funcs = {}

    def _register_convertor(key, convert_funcs=convert_funcs):
        def _reg(func):
            convert_funcs[key] = func
            return func
        return _reg

    def toformat(self, format, copy=True):
        """
        Changes the internal representation of the bounding box using
        one of the registered convertor functions.

        Args:
            format (str):
                the string code for the format you want to transform into.

            copy (bool):
                if False, the conversion is done inplace, but only if possible.
                Defaults to True.

        Returns:
            Boxes : transformed boxes

        TODO:
            - [ ] rename to to_format?

        CommandLine:
            xdoctest -m kwimage.structs.boxes _BoxConversionMixins.toformat

        Example:
            >>> boxes = Boxes.random(2, scale=10, rng=0)

            >>> boxes.toformat('ltrb')
            <Boxes(ltrb,
                array([[5, 5, 6, 7],
                       [4, 6, 4, 8]]))>

            >>> boxes.toformat('cxywh')
            <Boxes(cxywh,
                array([[5.5, 6. , 1. , 2. ],
                       [4. , 7. , 0. , 2. ]]))>

            >>> boxes.toformat('xywh')
            <Boxes(xywh,
                array([[5, 5, 1, 2],
                       [4, 6, 0, 2]]))>

            >>> boxes.toformat('_yyxx')
            >>> boxes.toformat('_rchw')
            >>> boxes.toformat('xywh')
            >>> boxes.toformat('ltrb')
            ...
        """
        key = BoxFormat.aliases.get(format, format)
        try:
            func = self.convert_funcs[key]
            return func(self, copy)
        except KeyError:
            raise KeyError('Cannot convert {} to {}'.format(self.format, format))

    @_register_convertor(BoxFormat.XXYY)
    def to_xxyy(self, copy=True):
        if self.format == BoxFormat.XXYY:
            return self.copy() if copy else self
        else:
            # Only difference between ltrb and extent=xxyy is the column order
            # xxyy: is x1, x2, y1, y2
            ltrb = self.to_ltrb().data
            xxyy = ltrb[..., [0, 2, 1, 3]]
        return Boxes(xxyy, BoxFormat.XXYY, check=False)

    to_extent = to_xxyy

    @_register_convertor(BoxFormat.XYWH)
    def to_xywh(self, copy=True):
        if self.format == BoxFormat.XYWH:
            return self.copy() if copy else self
        elif self.format == BoxFormat.CXYWH:
            cx, cy, w, h = self.components
            x1 = cx - w / 2
            y1 = cy - h / 2
        elif self.format == BoxFormat.LTRB:
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
        elif self.format == BoxFormat.XXYY:
            x1, x2, y1, y2 = self.components
            w = x2 - x1
            h = y2 - y1
        elif self.format == BoxFormat._YYXX:
            y1, y2, x1, x2 = self.components
            w = x2 - x1
            h = y2 - y1
        elif self.format == BoxFormat._RCHW:
            y1, x1, h, w = self.components
            return self.to_ltrb(copy=copy).to_xywh(copy=copy)
        else:
            raise KeyError('Unknown conversion from format={} to xywh'.format(self.format))
        xywh = _cat([x1, y1, w, h])
        return Boxes(xywh, BoxFormat.XYWH, check=False)

    @_register_convertor(BoxFormat.CXYWH)
    def to_cxywh(self, copy=True):
        if self.format == BoxFormat.CXYWH:
            return self.copy() if copy else self
        elif self.format == BoxFormat.XYWH:
            x1, y1, w, h = self.components
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
        elif self.format == BoxFormat.LTRB:
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        elif self.format == BoxFormat.XXYY:
            x1, x2, y1, y2 = self.components
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        elif self.format == BoxFormat._YYXX:
            return self.to_ltrb(copy=copy).to_cxywh(copy=copy)
        elif self.format == BoxFormat._RCHW:
            return self.to_ltrb(copy=copy).to_cxywh(copy=copy)
        else:
            raise KeyError('Unknown conversion from format={} to cxywh'.format(self.format))
        cxywh = _cat([cx, cy, w, h])
        return Boxes(cxywh, BoxFormat.CXYWH, check=False)

    @_register_convertor(BoxFormat.LTRB)
    def to_ltrb(self, copy=True):
        if self.format == BoxFormat.LTRB:
            return self.copy() if copy else self
        elif self.format == BoxFormat.XXYY:
            x1, x2, y1, y2 = self.components
        elif self.format == BoxFormat.CXYWH:
            cx, cy, w, h = self.components
            half_w = (w / 2)
            half_h = (h / 2)
            x1 = cx - half_w
            x2 = cx + half_w
            y1 = cy - half_h
            y2 = cy + half_h
        elif self.format == BoxFormat.XYWH:
            x1, y1, w, h = self.components
            x2 = x1 + w
            y2 = y1 + h
        elif self.format == BoxFormat._YYXX:
            y1, y2, x1, x2 = self.components
        elif self.format == BoxFormat._RCHW:
            y1, x1, h, w = self.components
            x2 = x1 + w
            y2 = y1 + h
        else:
            raise KeyError('Unknown conversion from format={} to ltrb'.format(self.format))
        ltrb = _cat([x1, y1, x2, y2])
        return Boxes(ltrb, BoxFormat.LTRB, check=False)

    def to_tlbr(self, **kwargs):
        ub.schedule_deprecation(
            'kwimage', 'Boxes.to_tlbr', 'method',
            migration='Use Boxes.to_ltrb instead.', deprecate='0.9.8',
            error='0.11.0', remove='0.12.0')
        return self.to_ltrb(**kwargs)

    @_register_convertor(BoxFormat._RCHW)
    def _to_rchw(self, copy=True):
        if self.format == BoxFormat._RCHW:
            return self.copy() if copy else self
        if self.format == BoxFormat.XYWH:
            _rchw = self.data[..., [1, 0, 3, 2]]
        else:
            _rchw = self.to_xywh(copy)._to_rchw(copy)
        return Boxes(_rchw, BoxFormat._RCHW, check=False)

    @_register_convertor(BoxFormat._YYXX)
    def _to_yyxx(self, copy=True):
        if self.format == BoxFormat._YYXX:
            return self.copy() if copy else self
        if self.format == BoxFormat.LTRB:
            _yyxx = self.data[..., [1, 3, 0, 2]]
        else:
            _yyxx = self.to_ltrb(copy)._to_yyxx(copy)
        return Boxes(_yyxx, BoxFormat._YYXX, check=False)

    def to_imgaug(self, shape):
        """
        Args:
            shape (tuple): shape of image that boxes belong to

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> self = Boxes([[25, 30, 15, 10]], 'ltrb')
            >>> bboi = self.to_imgaug((10, 10))
        """
        import imgaug
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))

        if shape is None:
            shape = (
                int(np.ceil(self.br_y.max())) * 2,
                int(np.ceil(self.br_x.max())) * 2,
            )

        if not isinstance(shape, tuple):
            import kwarray
            shape = tuple(kwarray.ArrayAPI.list(shape))

        ltrb = self.to_ltrb(copy=False).data
        bbs = [imgaug.BoundingBox(x1, y1, x2, y2) for x1, y1, x2, y2 in ltrb]
        bboi = imgaug.BoundingBoxesOnImage(bbs, shape=shape)
        return bboi

    def to_shapely(self):
        """
        Convert boxes to a list of shapely polygons

        Returns:
            List[shapely.geometry.Polygon]: list of shapely polygons
        """
        from shapely.geometry import Polygon
        x1, y1, x2, y2 = self.to_ltrb(copy=False).components
        a = _cat([x1, y1]).tolist()
        b = _cat([x1, y2]).tolist()
        c = _cat([x2, y2]).tolist()
        d = _cat([x2, y1]).tolist()
        regions = [Polygon(points) for points in zip(a, b, c, d, a)]
        # This just returns polygons anyway
        # regions = [
        #     shapely.geometry.box(minx, miny, maxx, maxy)
        #     for minx, miny, maxx, maxy in zip(x1, y1, x2, y2)
        # ]
        return regions

    def to_shapley(self):
        """
        DEPRECATED (misspelling). Use to_shapely

        Returns:
            List[shapely.geometry.Polygon]: list of shapely polygons
        """
        ub.schedule_deprecation(
            'kwimage', 'Boxes.to_shapley', 'method',
            migration='is spelled incorrectly. Use to_shapely instead',
            deprecate='0.9.0', error='0.10.0', remove='0.11.0')
        return self.to_shapely()

    @classmethod
    def from_shapely(cls, geom):
        """
        Given a shapely polygon, return a Boxes object of its Bounds.

        Args:
            geom (shapely.geometry.Polygon): shapely geometry

        Returns:
            Boxes: bounding box of the geometry
        """
        from shapely.geometry import Polygon
        if isinstance(geom, Polygon):
            xmin, ymin, xmax, ymax = geom.bounds
            self = Boxes(np.array([geom.bounds]), 'ltrb')
        else:
            raise NotImplementedError
        return self

    @classmethod
    def coerce(Boxes, data, **kwargs):
        """
        Args:
            data : can be :
                * a Boxes object
                * a shapely Polygon
                * list of 4 numbers (also requires the format kwarg)

            **kwargs:
                format (str | None) :
                    specify the format code

        Returns:
            Boxes: the wrapped or converted object
        """
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
        return self

    @classmethod
    def from_imgaug(Boxes, bboi):
        """
        Args:
            bboi (ia.BoundingBoxesOnImage):

        Returns:
            Boxes

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> orig = Boxes.random(5, format='ltrb')
            >>> bboi = orig.to_imgaug(shape=(500, 500))
            >>> self = Boxes.from_imgaug(bboi)
            >>> assert np.all(self.data == orig.data)
        """
        ltrb = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        ltrb = ltrb.reshape(-1, 4)
        return Boxes(ltrb, format=BoxFormat.LTRB, check=False)

    @classmethod
    def from_slice(Boxes, slices, shape=None, clip=True, endpoint=True,
                   wrap=False):
        """
        Creates a box from a 2D slice

        Note:
            The input slices and shape are y/x based because they are typically
            used with c-style arrays. However, the returned boxes use x/y
            ordering by default.

        Args:
            Tuple[slice, slice, ...]: a tuple where the first two
                items are a slice in the y-dimension and x-dimension.

            shape (Tuple[int, int]): the height / width of the canvas
                the slices would be applied to. This is only necessary if clip
                is True to specify if the bounds of the slice are undefined or
                need to be clipped.

            clip (bool):
                if True, assume that the box should be positive.
                Thus, we clip to 0 and the "shape" if specified.

            wrap (bool):
                if True, and shape is specified then negative coordinates are
                interpreted as "wrapping around", i.e. relative to the lower
                right side.

            endpoint (bool):
                if True, the endpoint of this slice is included as the
                bottom/right box coordinate. If False we subtract 1 such that
                box coordinates lie on pixels specified by the slice.

        Note:
            * When using this function the user needs to carefully consider
              if clip should be True or False in their use case.

            * The bottom-right side of the box includes the "stop" coordinate.

        Returns:
            Boxes: a Boxes object containing one box corresponding to the
            spatial dimensions of the slice.

        Example:
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> # Normal case
            >>> slices = (slice(10, 20), slice(11, 17))
            >>> boxes = Boxes.from_slice(slices)
            >>> assert np.all(boxes.data[0] == [11, 10, 17, 20])
            >>> # Clipping case
            >>> boxes = Boxes.from_slice(slices, shape=(1, 1))
            >>> assert np.all(boxes.data[0] == [1, 1, 1, 1])
            >>> # Empty slices
            >>> boxes = Boxes.from_slice(None, shape=(10, 10))
            >>> assert np.all(boxes.data[0] == [0, 0, 10, 10])
            >>> boxes = Boxes.from_slice(tuple(), shape=(10, 10))
            >>> assert np.all(boxes.data[0] == [0, 0, 10, 10])
            >>> # Only one slice
            >>> boxes = Boxes.from_slice(slice(2, 5), shape=(10, 10))
            >>> assert np.all(boxes.data[0] == [0, 2, 10, 5])

        Example:
            >>> import kwimage
            >>> grid = list(ub.named_product({
            >>>     'slices': [(slice(-10, 10), slice(-11, 17))],
            >>>     'clip': [True, False],
            >>>     'wrap': [False,],
            >>>     'shape': [(5, 5), None],
            >>> }))
            >>> grid += list(ub.named_product({
            >>>     'slices': [(slice(-10, -2), slice(-11, -2))],
            >>>     'wrap': [True,],
            >>>     'clip': [True, False],
            >>>     'shape': [(5, 5)],
            >>> }))
            >>> results = {}
            >>> for kwargs in grid:
            >>>     key = ub.repr2(kwargs, compact=1)
            >>>     box = kwimage.Boxes.from_slice(**kwargs)
            >>>     results[key] = box
            >>> print('results = {}'.format(ub.repr2(results, nl=1, sort=0, align=':')))
        """
        # Rectify input slices to agree with a 2D canvas
        if slices is None:
            slices = (slice(None), slice(None))
        if isinstance(slices, slice):
            slices = (slices,)
        if len(slices) < 2:
            _tail = tuple([slice(None)] * (2 - len(slices)))
            slices = tuple(slices) + _tail

        y_sl, x_sl = slices[0:2]
        tl_x = x_sl.start
        tl_y = y_sl.start
        rb_x = x_sl.stop
        rb_y = y_sl.stop

        if tl_x is None:
            tl_x = 0

        if tl_y is None:
            tl_y = 0

        if shape is not None:
            height, width = shape[0:2]
        else:
            height, width = None, None

        if rb_x is None:
            if width is None:
                raise Exception('shape required for unbounded slices')
            rb_x = width
        # elif rb_x < 0:
        #     if clip:
        #         raise ValueError('Cannot have negative right side when clip=True')
        #         if width is None:
        #             raise Exception('shape required for unbounded slices')
        #         rb_x = width + rb_x

        if rb_y is None:
            if height is None:
                raise Exception('shape required for unbounded slices')
            rb_y = height
        # elif rb_y < 0:
        #     if not clip:
        #         if height is None:
        #             raise Exception('shape required for unbounded slices')
        #         rb_y = height + rb_y

        if wrap:
            if rb_x < 0:
                if width is None:
                    raise Exception('shape required to wrap unbounded slices')
                rb_x = width + rb_x
            if tl_x < 0:
                if width is None:
                    raise Exception('shape required to wrap unbounded slices')
                tl_x = width + tl_x
            if rb_y < 0:
                if height is None:
                    raise Exception('shape required to wrap unbounded slices')
                rb_y = height + rb_y
            if tl_y < 0:
                if height is None:
                    raise Exception('shape required to wrap unbounded slices')
                tl_y = height + tl_y

        if clip:
            tl_y = max(tl_y, 0)
            tl_x = max(tl_x, 0)

        if not endpoint:
            rb_x = rb_x - 1
            rb_y = rb_y - 1

        if tl_x > rb_x:
            raise ValueError(f'Invalid x slice tl_x={tl_x}, rb_x={rb_x}')
        if tl_y > rb_y:
            raise ValueError(f'Invalid y slice tl_y={tl_y}, rb_y={rb_y}')

        ltrb = np.array([[tl_x, tl_y, rb_x, rb_y]])
        box = Boxes(ltrb, 'ltrb')

        if clip:
            if shape is not None:
                box.clip(0, 0, width, height, inplace=True)
        return box

    def to_slices(self, endpoint=True):
        """
        Convert the boxes into slices

        Args:
            endpoint (bool):
                Indicates if the box specifies the slice endpoint.
                The box specifies the slice endpoint if its bottom/right corner
                should *not* be in the pixels extracted from the slice.

                if True, we assume the bot/right corner of the box specifies
                the stop point of the slice (i.e is not included).  if False,
                we add 1 from the bot/right to get the slice stop point such
                that the bot/right pixel will be included in the slice.

        TextArt:
            +--------------+
            l              r <- box coords
            s              t <- endpoint=True
            s               t <- endpoint=False

        Returns:
            List[Tuple[slice, slice]]:
                a list of slices corresponding to each box.
        """
        slices_list = []
        for tl_x, tl_y, br_x, br_y in self.to_ltrb().data:
            if not endpoint:
                br_x = br_x + 1
                br_y = br_y + 1
            sl = (slice(tl_y, br_y), slice(tl_x, br_x))
            slices_list.append(sl)
        return slices_list

    def to_coco(self, style='orig'):
        """
        Convert to COCO format.

        Yields:
            List[float]: tl_x, tl_y, w, h

        Example:
            >>> orig = Boxes.random(5)
            >>> coco_boxes = list(orig.to_coco())
            >>> print('coco_boxes = {!r}'.format(coco_boxes))
        """
        for row in self.to_xywh().data.tolist():
            yield [round(x, 4) for x in row]

    def to_polygons(self):
        """
        Convert each box to a polygon object

        Returns:
            kwimage.PolygonList

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(5)
            >>> polys = self.to_polygons()
            >>> print('polys = {!r}'.format(polys))
        """
        import kwimage
        poly_list = []
        for ltrb in self.to_ltrb().data:
            x1, y1, x2, y2 = ltrb
            # Exteriors are counterlockwise
            exterior = np.array([
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1],
                [x1, y1],
            ])
            poly = kwimage.Polygon(exterior=exterior)
            poly_list.append(poly)
        polys = kwimage.PolygonList(poly_list)
        return polys


class _BoxPropertyMixins(object):

    @property
    def xy_center(self):
        """
        Returns the xy coordinates of the box centers

        Note:
            the difference between this and `self.center` is that this returns
            a single ndarray[dim=2] whereas `self.center` returns two ndarrays.
        """
        import warnings
        warnings.warn('Redundant, use self.center instead', DeprecationWarning)
        xy = self.to_cxywh(copy=False).data[..., 0:2]
        return xy

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    def _component(self, idx):
        return self.data[..., idx:idx + 1]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def tl_x(self):
        """
        Top left x coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'ltrb').tl_x
            array([25])
        """
        return self.to_ltrb(copy=False)._component(0)

    @property
    def tl_y(self):
        """
        Top left y coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'ltrb').tl_y
            array([30])
        """
        return self.to_ltrb(copy=False)._component(1)

    @property
    def br_x(self):
        """
        Bottom right x coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'ltrb').br_x
            array([35])
        """
        return self.to_ltrb(copy=False)._component(2)

    @property
    def br_y(self):
        """
        Bottom right y coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'ltrb').br_y
            array([40])
        """
        return self.to_ltrb(copy=False)._component(3)

    @property
    def width(self):
        """
        Bounding box width

        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').width
            array([15])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').width
            array([[0]])
        """
        w = self.to_xywh(copy=False)._component(2)
        return w

    @property
    def height(self):
        """
        Bounding box height

        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').height
            array([10])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').height
            array([[0]])
        """
        h = self.to_xywh(copy=False)._component(3)
        return h

    @property
    def aspect_ratio(self):
        """
        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').aspect_ratio
            array([1.5])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').aspect_ratio
            array([[nan]])
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.width / self.height

    @property
    def area(self):
        """
        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').area
            array([150])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').area
            array([[0]])
        """
        w, h = self.to_xywh(copy=False).components[2:4]
        return w * h

    @property
    def center(self):
        """
        The center xy-coordinates

        Returns:
            Tuple[ndarray, ndarray]: the center x and y coordinates

        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').area
            array([150])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').area
            array([[0]])
        """
        cx, cy = self.to_cxywh(copy=False).components[0:2]
        return cx, cy


class _BoxTransformMixins(object):
    """
    methods for transforming bounding boxes
    """

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Args:
            augmenter (imgaug.augmenters.Augmenter):
                an imgaug augmenter

            input_dims (Tuple): h/w of the input image

            inplace (bool):
                if True, modifies data inplace

        Returns:
            Boxes

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> import imgaug
            >>> self = Boxes.random(10)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> input_dims = (10, 10)
            >>> new = self._warp_imgaug(augmenter, input_dims)
        """
        new = self if inplace else self.__class__(self.data, self.format)
        bboi = self.to_imgaug(shape=input_dims)
        bboi = augmenter.augment_bounding_boxes([bboi])[0]
        ltrb = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        ltrb = ltrb.reshape(-1, 4)
        new.data = ltrb
        new.format = BoxFormat.LTRB
        if self._impl.is_tensor:
            new = new.tensor()
        return new

    # @profile
    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Generalized coordinate transform. Note that transformations that are
        not axis-aligned will lose information (and also may not be
        implemented).

        Args:
            transform (ArrayLike | Callable | kwimage.Affine | skimage.transform._geometric.GeometricTransform | Any):
                scikit-image tranform, a 3x3 transformation matrix,
                an imgaug Augmenter, or generic callable which transforms
                an NxD ndarray.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused in non-raster spatial structures

            inplace (bool): if True, modifies data inplace

        Returns:
            Boxes: warped boxes (really the bounding boxes of the resulting
                warped polygons)

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> transform = skimage.transform.AffineTransform(scale=(2, 3), translation=(4, 5))
            >>> Boxes([25, 30, 15, 10], 'xywh').warp(transform)
            <Boxes(xywh, array([54., 95., 30., 30.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').warp(transform.params)
            <Boxes(xywh, array([54., 95., 30., 30.]))>


        Example:
            >>> import kwimage
            >>> # Can warp corners with a transformation matrix
            >>> self = kwimage.Boxes.random(3, rng=0).scale(100).round(0)
            >>> transform = np.array([[ 0.98412825,  0.0577905 ,  0.16778511],
            >>>                       [-0.05968319,  0.99819777,  0.00625538],
            >>>                       [-0.16712122, -0.01617005,  0.98580375]])
            >>> matrix = transform
            >>> new = self.warp(transform)
            >>> assert np.all(new.area != self.area)

        Example:
            >>> import kwimage
            >>> # can use a generic function to warp corners
            >>> self = kwimage.Boxes.random(3).scale(100).round(0)
            >>> def func(xy):
            ...     return xy * 2
            >>> new = self.warp(func)
            >>> assert np.allclose(new.data, self.scale(2).to_ltrb().data)
            >>> # If the box is distorted, the operation is not invertable
            >>> self = kwimage.Boxes.random(3).scale(100).round(0)
            >>> def func(xy):
            ...     return np.zeros_like(xy)
            >>> new = self.warp(func)
            >>> assert np.all(new.area == 0)
        """
        import kwimage

        if inplace:
            new = self
            new_data = self.data
        else:
            if torch is not None and torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(float, copy=True)
            new = Boxes(new_data, self.format)

        if transform is None:
            return new

        try:
            # First try to warp using simple calls to axis-aligned operations
            rotation = 0
            shear = 0
            scale = 0
            translation = 0
            matrix = None
            func = None

            if isinstance(transform, skimage.transform.AffineTransform):
                rotation = transform.rotation
                shear = transform.shear
                scale = transform.scale
                translation = transform.translation
            elif isinstance(transform, skimage.transform.SimilarityTransform):
                rotation = transform.rotation
                scale = transform.scale
                translation = transform.translation
            elif isinstance(transform, skimage.transform.EuclideanTransform):
                rotation = transform.rotation
                translation = transform.translation
            elif isinstance(transform, skimage.transform._geometric.GeometricTransform):
                matrix = transform.params
            elif isinstance(transform, _generic.ARRAY_TYPES):
                matrix = transform
            elif isinstance(transform, kwimage.Affine):
                matrix = transform.matrix
            elif isinstance(transform, kwimage.Projective):
                matrix = transform.matrix
                raise NeedsWarpCorners
            else:
                try:
                    import imgaug
                except ImportError:
                    pass
                    # import warnings
                    # warnings.warn('imgaug is not installed')
                else:
                    if isinstance(transform, imgaug.augmenters.Augmenter):
                        aug = new._warp_imgaug(transform, input_dims=input_dims, inplace=True)
                        return aug

                if callable(transform):
                    func = transform
                    raise NeedsWarpCorners
                else:
                    raise TypeError(type(transform))

            if matrix is not None:
                # See if we can extract simple params from the matrix
                require_zeros = [[0, 1], [1, 0], [2, 0], [2, 1]]
                require_ones = [[2, 2]]
                if np.any(matrix[tuple(zip(*require_zeros))] != 0):
                    raise NeedsWarpCorners
                if np.any(matrix[tuple(zip(*require_ones))] != 1):
                    raise NeedsWarpCorners
                scale = matrix[(0, 1), (0, 1)]
                translation = matrix[(0, 1), (2, 2)]

            if rotation != 0 or shear != 0:
                raise NeedsWarpCorners
            else:
                # We don't need do do anything fancy
                new.scale(scale, inplace=True)
                new.translate(translation, inplace=True)

        except NeedsWarpCorners:
            corners = []
            x1, y1, x2, y2 = [a.ravel() for a in self.to_ltrb().components]
            stacked = np.array([
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1],
            ])
            corners = stacked.transpose(2, 0, 1).reshape(-1, 2)
            corners = np.ascontiguousarray(corners)

            # apply the operation to warp the corner points
            if matrix is not None:
                corners_new = kwimage.warp_points(matrix, corners)
            elif func is not None:
                corners_new = func(corners)
            else:
                raise NotImplementedError(
                    'Corner warping is not implemented yet for '
                    'transform={!r}'.format(transform))

            x_pts_new = corners_new[..., 0].reshape(-1, 4)
            y_pts_new = corners_new[..., 1].reshape(-1, 4)

            x1_new = x_pts_new.min(axis=1)
            x2_new = x_pts_new.max(axis=1)
            y1_new = y_pts_new.min(axis=1)
            y2_new = y_pts_new.max(axis=1)

            data_new = np.hstack([
                x1_new[:, None], y1_new[:, None],
                x2_new[:, None], y2_new[:, None],
            ])
            new.data = data_new
            new.format = 'ltrb'

        return new

    def corners(self):
        """
        Return the corners of the boxes

        This function is unintuitive and may be deprecated

        Returns:
            np.ndarray : stacked corners in an array with shape [4*N, 2]
        """
        corners = []
        x1, y1, x2, y2 = [a.ravel() for a in self.to_ltrb().components]
        stacked = np.array([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1],
        ])
        corners = stacked.transpose(2, 0, 1).reshape(-1, 2)
        corners = np.ascontiguousarray(corners)
        return corners

    def scale(self, factor, about='origin', output_dims=None, inplace=False):
        """
        Scale a bounding boxes by a factor.

        works natively with ltrb, cxywh, xywh, xy, or wh formats

        Args:
            factor (float | Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.

            about (str | ArrayLike):
                Origin of the scaling operation, Can be a single point, an
                array of points for each box, or a special string:
                    'origin': all boxes are scaled about (0, 0)
                    'centroid' or 'center': all boxes are scaled about their own center.
                Defaults to 'origin'

            output_dims (Tuple): unused in non-raster spatial structures

            inplace (bool):
                if True works inplace if possible. Defaults to False

        Returns:
            Boxes: scaled boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes(np.array([1, 1, 10, 10]), 'xywh').scale(2).data
            array([ 2.,  2., 20., 20.])
            >>> Boxes(np.array([[1, 1, 10, 10]]), 'xywh').scale((2, .5)).data
            array([[ 2. ,  0.5, 20. ,  5. ]])

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> x = Boxes([25., 30., 15., 10.], 'ltrb')
            >>> x.scale(2)
            >>> print(x)
            <Boxes(ltrb, array([25., 30., 15., 10.]))>
            >>> x.scale(2.0, inplace=True)
            >>> print(x)
            <Boxes(ltrb, array([50., 60., 30., 20.]))>

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> import kwimage
            >>> rng = kwarray.ensure_rng(0)
            >>> boxes = kwimage.Boxes.random(num=3, scale=10, rng=rng).astype(np.float64)
            >>> scale_xy = (10 * rng.rand(len(boxes), 2))
            >>> print(ub.repr2(boxes.scale(scale_xy).data, precision=2))
            np.array([[28.4 , 46.28,  5.68, 18.51],
                      [ 2.84,  5.23,  0.  ,  1.74],
                      [ 1.42, 24.98,  0.4 , 16.65]], dtype=np.float64)
            >>> y0 = boxes.toformat('xywh').scale(scale_xy).toformat('xywh')
            >>> y1 = boxes.toformat('ltrb').scale(scale_xy).toformat('xywh')
            >>> y2 = boxes.toformat('xxyy').scale(scale_xy).toformat('xywh')
            >>> assert ub.allsame([y0.data, y1.data, y2.data], eq=np.allclose)

        Example:
            >>> import kwimage
            >>> # Test with about=center
            >>> self = kwimage.Boxes([[25., 30., 15., 10.], [2, 0, 15., 10.]], 'cxywh')
            >>> scale_xy = (2, 4)
            >>> print(ub.repr2(self.scale(scale_xy, about='center').data, precision=2))
            >>> y0 = self.toformat('xywh').scale(scale_xy, about='center').toformat('cxywh')
            >>> y1 = self.toformat('ltrb').scale(scale_xy, about='center').toformat('cxywh')
            >>> y2 = self.toformat('xxyy').scale(scale_xy, about='center').toformat('cxywh')
            >>> assert ub.allsame([y0.data, y1.data, y2.data], eq=np.allclose)
        """
        if not ub.iterable(factor):
            sx = sy = factor
        elif isinstance(factor, (list, tuple)):
            sx, sy = factor
        else:
            sx = factor[..., 0]
            sy = factor[..., 1]

        if inplace:
            new = self
            new_data = self.data
        else:
            if torch is not None and torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(float, copy=True)
            new = Boxes(new_data, self.format)

        if _numel(new_data) > 0:

            if isinstance(about, str):
                if about == 'origin':
                    about = None
                elif about in {'center', 'centroid'}:
                    about = self.xy_center
                else:
                    raise KeyError(about)

            if about is None:
                # scale about the origin
                if self.format in [BoxFormat.XYWH, BoxFormat.CXYWH, BoxFormat.LTRB]:
                    new_data[..., 0] *= sx
                    new_data[..., 1] *= sy
                    new_data[..., 2] *= sx
                    new_data[..., 3] *= sy
                elif self.format in [BoxFormat.XXYY]:
                    new_data[..., 0] *= sx
                    new_data[..., 1] *= sx
                    new_data[..., 2] *= sy
                    new_data[..., 3] *= sy
                else:
                    raise NotImplementedError('Cannot scale: {}'.format(self.format))
            else:
                # scale about some point: translate, scale, untranslate
                about_x = about[..., 0]
                about_y = about[..., 1]
                if self.format in [BoxFormat.XYWH, BoxFormat.CXYWH]:
                    new_data[..., 0] = (new_data[..., 0] - about_x) * sx + about_x
                    new_data[..., 1] = (new_data[..., 1] - about_y) * sy + about_y
                    new_data[..., 2] *= sx
                    new_data[..., 3] *= sy
                elif self.format in [BoxFormat.LTRB]:
                    new_data[..., 0] = (new_data[..., 0] - about_x) * sx + about_x
                    new_data[..., 1] = (new_data[..., 1] - about_y) * sy + about_y
                    new_data[..., 2] = (new_data[..., 2] - about_x) * sx + about_x
                    new_data[..., 3] = (new_data[..., 3] - about_y) * sy + about_y
                elif self.format in [BoxFormat.XXYY]:
                    new_data[..., 0] = (new_data[..., 0] - about_x) * sx + about_x
                    new_data[..., 1] = (new_data[..., 1] - about_x) * sx + about_x
                    new_data[..., 2] = (new_data[..., 2] - about_y) * sy + about_y
                    new_data[..., 3] = (new_data[..., 3] - about_y) * sy + about_y
                else:
                    raise NotImplementedError('Cannot scale about: {}'.format(self.format))
        return new

    def translate(self, amount, output_dims=None, inplace=False):
        """
        Shift the boxes up/down left/right

        Args:
            factor (float | Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Returns:
            Boxes: translated boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes([25, 30, 15, 10], 'xywh').translate(10)
            <Boxes(xywh, array([35., 40., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').translate((10, 0))
            <Boxes(xywh, array([35., 30., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'ltrb').translate((10, 5))

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> x = Boxes([25, 30, 15, 10], 'ltrb')
            >>> x.translate((10, 5))
            >>> print(x)
            <Boxes(ltrb, array([25, 30, 15, 10]))>
            >>> x.translate((10, 5), inplace=True)
            >>> print(x)
            <Boxes(ltrb, array([35, 35, 25, 15]))>

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> import kwimage
            >>> rng = kwarray.ensure_rng(0)
            >>> boxes = kwimage.Boxes.random(num=3, scale=10, rng=rng)
            >>> dxdy = (10 * rng.randn(len(boxes), 2)).astype(int)
            >>> boxes.translate(dxdy)
            <Boxes(xywh,
                array([[12.,  6.,  1.,  2.],
                       [ 8.,  9.,  0.,  2.],
                       [21.,  1.,  2.,  2.]]))>
            >>> y0 = boxes.toformat('xywh').translate(dxdy).toformat('xywh')
            >>> y1 = boxes.toformat('ltrb').translate(dxdy).toformat('xywh')
            >>> y2 = boxes.toformat('xxyy').translate(dxdy).toformat('xywh')
            >>> assert ub.allsame([y0, y1, y2])
        """
        if not ub.iterable(amount):
            tx = ty = amount
        elif isinstance(amount, (list, tuple)):
            tx, ty = amount
        else:
            tx = amount[..., 0]
            ty = amount[..., 1]

        if inplace:
            new = self
            new_data = self.data
        else:
            if torch is not None and torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(float, copy=True)
            new = Boxes(new_data, self.format)

        if _numel(new_data) > 0:
            if self.format in [BoxFormat.XYWH, BoxFormat.CXYWH]:
                new_data[..., 0] += tx
                new_data[..., 1] += ty
            elif self.format in [BoxFormat.LTRB]:
                new_data[..., 0] += tx
                new_data[..., 1] += ty
                new_data[..., 2] += tx
                new_data[..., 3] += ty
            elif self.format in [BoxFormat.XXYY]:
                new_data[..., 0] += tx
                new_data[..., 1] += tx
                new_data[..., 2] += ty
                new_data[..., 3] += ty
            else:
                raise NotImplementedError('Cannot translate: {}'.format(self.format))
        return new

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip boxes to image boundaries.

        If box is in ltrb format, inplace operation is an option.

        Args:
            x_min (int): minimum x-coordinate
            y_min (int): minimum x-coordinate
            x_max (int): maximum x-coordinate
            y_max (int): maximum y-coordinate
            inplace (bool):
                if True and possible, perform operation inplace. Defaults to
                False.

        Returns:
            Boxes: clipped boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> self = boxes = Boxes(np.array([[-10, -10, 120, 120], [1, -2, 30, 50]]), 'ltrb')
            >>> clipped = boxes.clip(0, 0, 110, 100, inplace=False)
            >>> assert np.any(boxes.data != clipped.data)
            >>> clipped2 = boxes.clip(0, 0, 110, 100, inplace=True)
            >>> assert clipped2.data is boxes.data
            >>> assert np.all(clipped2.data == clipped.data)
            >>> print(clipped)
            <Boxes(ltrb,
                array([[  0,   0, 110, 100],
                       [  1,   0,  30,  50]]))>
        """
        if inplace:
            if self.format != BoxFormat.LTRB:
                raise ValueError('Must be in ltrb format to operate inplace')
            new = self
        else:
            new = self.to_ltrb(copy=True)
        if len(new) == 0:
            return new

        if True:
            impl = self._impl
            x1, y1, x2, y2 = impl.T(new.data)
            np.clip(x1, x_min, x_max, out=x1)
            np.clip(y1, y_min, y_max, out=y1)
            np.clip(x2, x_min, x_max, out=x2)
            np.clip(y2, y_min, y_max, out=y2)
        else:
            if torch is not None and torch.is_tensor(new.data):
                x1, y1, x2, y2 = new.data.t()
                x1.clamp_(x_min, x_max)
                y1.clamp_(y_min, y_max)
                x2.clamp_(x_min, x_max)
                y2.clamp_(y_min, y_max)
            else:
                x1, y1, x2, y2 = new.data.T
                np.clip(x1, x_min, x_max, out=x1)
                np.clip(y1, y_min, y_max, out=y1)
                np.clip(x2, x_min, x_max, out=x2)
                np.clip(y2, y_min, y_max, out=y2)
        return new

    def resize(self, width=None, height=None, inplace=False):
        """
        Set the widths and/or heights of each box, while leaving the minimum
        x/y point constant.

        Args:
            width (Number | ndarray | None):
                if specified and a number, sets the width of each box to this
                value. If this is a broadcastable ndarray, the width of each
                box can be set individually.

            height (Number | ndarray | None):
                same as width, except this modifies height components.

            inplace (bool):
                if True and possible, perform operation inplace.
                Defaults to False.

        TODO:
            - [ ] It would be nice to specify in which direction the box
                  shrinks or is expanded, but that might not play nice with
                  quantized coordinates.

        Returns:
            Boxes : modified boxes

        SeeAlso:
            :func:`Boxes.warp` and :func:`Boxes.scale` for size rescaling based
            on a factor rather than a fixed width/height.

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes([[1, 1, 4, 4]] * 5, 'ltrb')
            >>> # Test setting only the width to a scalar
            >>> new1 = self.resize(width=10)
            >>> assert np.all(new1.width == 10)
            >>> assert np.all(new1.height == 3)
            >>> # Test setting only the height to a scalar
            >>> new2 = self.resize(height=10)
            >>> assert np.all(new2.width == 3)
            >>> assert np.all(new2.height == 10)
            >>> # Test setting width and height per-box values
            >>> new3 = self.resize(
            >>>     width=np.arange(0, 5),
            >>>     height=np.arange(4, 13, 2))
            >>> assert np.all(new3.width.ravel() == [0, 1, 2, 3, 4])
            >>> assert np.all(new3.height.ravel() == [4, 6, 8, 10, 12])

        Example:
            >>> # Test setting width and height per-box values
            >>> # for a multidimensional setting
            >>> import kwimage
            >>> self = kwimage.Boxes([[1, 1, 4, 4]] * 8, 'ltrb').view(2, 2, 2, -1)
            >>> new = self.resize(
            >>>     width=np.arange(0, 8).reshape(2, 2, 2),
            >>>     height=np.arange(10, 18).reshape(2, 2, 2))
            >>> assert np.all(new.width.ravel() == np.arange(0, 8))
            >>> assert np.all(new.height.ravel() == np.arange(10, 18))
        """
        if inplace:
            if self.format != BoxFormat.XYWH:
                raise ValueError('Must be in xywh format to operate inplace')
            new = self
        else:
            new = self.to_xywh(copy=True)
        if width is not None:
            new.data[..., 2] = width
        if height is not None:
            new.data[..., 3] = height
        new = new.toformat(self.format, copy=False)
        return new

    def _set_axis(self, new_width):
        pass

    def pad(self, x_left, y_top, x_right, y_bot, inplace=False):
        """
        Adds extra width/height to the left, top, right, and bottom
        of each bounding box.

        Args:
            x_left (int | float): xmin pad
            y_top (int | float): ymin pad
            x_right (int | float): xmax pad
            y_bot (int | float): ymax pad

        Returns:
            Boxes : padded boxes

        Note:
            The argument names to this function are assuming up is negative and
            down is positive. We may change them in the future to be agnostic
            to image vs blackboard coordinates.

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes([[0, 0, 10, 10]], 'xywh').to_ltrb().quantize()
            >>> padded = self.pad(1, 2, 3, 4)
            >>> print('padded = {}'.format(ub.repr2(padded, nl=1)))
            >>> assert np.all(padded.data == [[-1, -2, 13, 14]])
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(fnum=1, doclf=1)
            >>> self.draw('kw_blue')
            >>> padded.draw('kw_green', setlim=1.2)
            >>> plt.gca().set_title('kwimage.Boxes.pad')
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwimage
            >>> # test dtype promotion rules
            >>> self = kwimage.Boxes.random().scale(100).to_ltrb().quantize().astype(np.int64)
            >>> padded = self.pad(0, 0, 0, 0)
            >>> assert padded.data.dtype == np.int64
            >>> self = kwimage.Boxes.random().scale(100).to_ltrb().quantize().astype(np.int32)
            >>> padded = self.pad(0, 0, 0, 0)
            >>> assert padded.data.dtype == np.int32
            >>> padded = self.pad(0, 0, 0, np.float64(0.))
            >>> assert padded.data.dtype == np.float64
        """
        impl = self._impl

        if inplace:
            new = self
            new_data = self.data
        else:
            dtype = impl.result_type(self.data, x_left, y_top, x_right, y_bot)
            new_data = impl.astype(self.data, dtype, copy=True)
            new = Boxes(new_data, self.format)

        if _numel(new_data) > 0:
            if self.format in [BoxFormat.LTRB]:
                new_data[..., 0] -= x_left
                new_data[..., 1] -= y_top
                new_data[..., 2] += x_right
                new_data[..., 3] += y_bot
            else:
                raise NotImplementedError('Cannot pad: {}'.format(self.format))
        return new

    def transpose(self):
        """
        Reflects box coordinates about the line y=x.

        Example:
            >>> Boxes([[0, 1, 2, 4]], 'ltrb').transpose()
            <Boxes(ltrb, array([[1, 0, 4, 2]]))>
        """
        x, y, w, h = self.to_xywh().components
        new = self.__class__(_cat([y, x, h, w]), format=BoxFormat.XYWH)
        new = new.toformat(self.format)
        return new


class _BoxDrawMixins(object):
    """
    Non-core functions for box visualization

    Example:
        >>> # Drawing boxes (and annotation objects in general) with kwimage is
        >>> # easy.  For boxes we assume that your data is in an [Nx4] array,
        >>> # but its important that you further specify a format so we know
        >>> # which each column represents. The major formats are:
        >>> #     'xywh'  -  (x1, y1, w, h)
        >>> #     'cxywh' -  (cx, cy, w, h)
        >>> #     'ltrb'  -  (x1, y1, x2, y2)  # formerly tlbr
        >>> #     'xxyy'  -  (x1, x2, y1, y2)
        >>> # (see kwimage.structs.boxes.BoxFormat for more format details)
        >>> #
        >>> # For the purposes of this demo lets assume you have boxes in
        >>> # "xywh" format.
        >>> data = np.random.rand(10, 4) * 224
        >>> #
        >>> # Simply wrap your data with a Boxes object
        >>> import kwimage
        >>> boxes = kwimage.Boxes(data, format='xywh')
        >>> #
        >>> # Now to draw the boxes there are two ways
        >>> #
        >>> # Method (2): OpenCV
        >>> # To use opencv you need to draw onto an existing numpy image.
        >>> # Assuming we have such an image:
        >>> image = np.random.rand(224, 224, 3)
        >>> #
        >>> # It is good practice to copy it, as we will modify it in-place
        >>> canvas = image.copy()
        >>> # Then simply call draw-on and the underlying ndarray will be
        >>> # modified such that your boxes are drawn on it.
        >>> canvas = boxes.draw_on(canvas)
        >>> #
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> # xdoctest: +REQUIRES(module:matplotlib)
        >>> # Method (1): Matplotlib
        >>> # Assuming you already have a figure setup with correct limits
        >>> # you can draw the boxes on top of that figure via:
        >>> boxes.draw()

    """

    def draw(self, color='blue', alpha=None, labels=None, centers=False,
             fill=False, lw=2, ax=None, setlim=False, **kwargs):
        """
        Draws boxes using matplotlib. Wraps around kwplot.draw_boxes

        Args:
            color (str | Any | List[Any]):
                one color for all boxes or a list of colors for each box
                Can be any type accepted by kwimage.Color.coerce.
                Extended types: str | ColorLike | List[ColorLike]

            alpha (float | List[float] | None):
                A single transparency for all boxes, or a list of
                transparencies for each box.

            labels (List[str] | None): a text label for each box

            centers (bool): if True, draw box centers.

            lw (float): linewidth for the box edges

            ax (Optional[matplotlib.axes.Axes]):
                if specified, draws on this existing axes, otherwise defaults
                to the current axes.

            setlim (bool):
                if True will set the limit of the axes to show the drawn boxes.

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(num=10, scale=512.0, rng=0, format='ltrb')
            >>> self.translate((-128, -128), inplace=True)
            >>> self.data[0][:] = [3, 3, 253, 253]
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
        import kwplot
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        lw = kwargs.get('linewidth', lw)

        if setlim:
            xmins, ymins, xmaxs, ymaxs = self.to_ltrb().components
            xmin = xmins.min()
            ymin = ymins.min()
            xmax = xmaxs.max()
            ymax = ymaxs.max()
            _generic._setlim(xmin, ymin, xmax, ymax, setlim, ax=ax)

        boxes = self.to_xywh()
        if len(boxes.shape) == 1 and boxes.shape[0] == 4:
            # Hack to draw non-2d boxes
            boxes = boxes[None, :]

        return kwplot.draw_boxes(boxes, color=color, labels=labels,
                                 alpha=alpha, centers=centers, fill=fill,
                                 lw=lw, ax=ax)

    def draw_on(self, image=None, color='blue', alpha=None, labels=None,
                copy=False, thickness=2, label_loc='top_left'):
        """
        Draws boxes directly on the image using OpenCV

        Args:
            image (ndarray): must be in uint8 format

            color (str | Any | List[Any]):
                one color for all boxes or a list of colors for each box
                Can be any type accepted by kwimage.Color.coerce.
                Extended types: str | ColorLike | List[ColorLike]

            alpha (float): transparency of bboxes

            labels (List[str]): a text label for each box

            copy (bool):
                if False only copies if necessary. Defaults to False.

            thickness (int): rectangle thickness, negative values
                will draw a filled rectangle. Defaults to 2.

            label_loc (str): indicates where labels (if specified) should be
                drawn.

        Returns:
            ndarray: the image drawn onto.

        Example:
            >>> import kwimage
            >>> import numpy as np
            >>> self = kwimage.Boxes.random(num=10, scale=256, rng=0, format='ltrb')
            >>> self.data[0][:] = [3, 3, 253, 253]
            >>> color = 'blue'
            >>> image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image.copy(), color=color)
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image, fnum=2000, pnum=(1, 2, 1))
            >>> kwplot.imshow(image2, fnum=2000, pnum=(1, 2, 2))
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwimage
            >>> import numpy as np
            >>> self = kwimage.Boxes.random(num=10, rng=0).scale(128)
            >>> self.data[0][:] = [3, 3, 100, 100]
            >>> color = 'blue'
            >>> # Test drawong on all channel + dtype combinations
            >>> im3 = np.random.rand(128, 128, 3)
            >>> im_chans = {
            >>>     'im3': im3,
            >>>     'im1': kwimage.convert_colorspace(im3, 'rgb', 'gray'),
            >>>     'im4': kwimage.convert_colorspace(im3, 'rgb', 'rgba'),
            >>> }
            >>> inputs = {}
            >>> for k, im in im_chans.items():
            >>>     inputs[k + '_01'] = (kwimage.ensure_float01(im.copy()), {'alpha': None})
            >>>     inputs[k + '_255'] = (kwimage.ensure_uint255(im.copy()), {'alpha': None})
            >>>     inputs[k + '_01_a'] = (kwimage.ensure_float01(im.copy()), {'alpha': 0.5})
            >>>     inputs[k + '_255_a'] = (kwimage.ensure_uint255(im.copy()), {'alpha': 0.5})
            >>> outputs = {}
            >>> for k, v in inputs.items():
            >>>     im, kw = v
            >>>     outputs[k] = self.draw_on(im, color=color, **kw)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=2, doclf=True)
            >>> kwplot.autompl()
            >>> pnum_ = kwplot.PlotNums(nCols=2, nRows=len(inputs))
            >>> for k in inputs.keys():
            >>>     kwplot.imshow(inputs[k][0], fnum=2, pnum=pnum_(), title=k)
            >>>     kwplot.imshow(outputs[k], fnum=2, pnum=pnum_(), title=k)
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(num=10, scale=256, rng=0, format='ltrb')
            >>> image = self.draw_on()
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> kwplot.show_if_requested()
        """
        import cv2
        import kwimage
        def _coords(x, y):
            # ensure coords don't go out of bounds or cv2 throws weird error
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            return tuple(map(int, map(round, (x, y))))

        if image is None:
            # If image is not given, use the boxes to allocate enough
            # room to draw
            bounds = self.bounding_box().scale(1.1).quantize()
            w = bounds.width.item()
            h = bounds.height.item()
            w = h = max(w, h)
            image = np.zeros((h, w, 3), dtype=np.float32)

        dtype_fixer = _generic._consistent_dtype_fixer(image)
        h, w = image.shape[0:2]

        # Get the color that is compatible with the input image encoding
        # rect_color = kwimage.Color(color)._forimage(image)

        # Parameters for drawing the box rectangles
        rectkw = {
            'thickness': int(thickness),
            # 'color': rect_color,
        }

        # Parameters for drawing the label text
        fontkw = {
            # 'color': rect_color,
            'thickness': int(2),
            'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
            'fontScale': 0.75,
            'lineType': cv2.LINE_AA,
        }

        ltrb_list = self.to_ltrb().data
        num = len(ltrb_list)

        image = kwimage.atleast_3channels(image, copy=copy)
        image = np.ascontiguousarray(image)

        if isinstance(color, list) and not isinstance(color, numbers.Number):
            # Passed list of color for each box
            colors = [kwimage.Color(c)._forimage(image) for c in color]
        else:
            # Passed a single color
            colors = [kwimage.Color(color)._forimage(image)] * num

        if alpha is None:
            alpha = [1.0] * num
        elif isinstance(alpha, (float, np.float32, np.float64)):
            alpha = [alpha] * num

        if labels is None or labels is False:
            labels = [None] * num

        if label_loc == 'top_left':
            # Create a relative origin for the text
            text_relxy_org = np.array([0, 0])
            valign = 'bottom'
            halign = 'left'
            y_shift_sign = -1
        elif label_loc == 'bottom_left':
            # Create a relative origin for the text
            text_relxy_org = np.array([0, 1])
            valign = 'top'
            halign = 'left'
            y_shift_sign = +1
        else:
            raise KeyError(label_loc)

        rel_x, rel_y = text_relxy_org

        for ltrb, label, alpha_, col in zip(ltrb_list, labels, alpha, colors):
            x1, y1, x2, y2 = ltrb
            pt1 = _coords(x1, y1)
            pt2 = _coords(x2, y2)

            # Note cv2.rectangle does work inplace
            if alpha_ < 1.0:
                background = image.copy()

            # while cv2.rectangle will accept an alpha color it will not do any
            # blending with the background image.
            image = cv2.rectangle(image, pt1, pt2, color=col, **rectkw)
            if label:
                # Compute the prefered location of the text origin
                x1, y1 = pt1
                x2, y2 = pt2
                y_shift = y_shift_sign * (rectkw['thickness'] * 2)
                org_x = (x1 * (1 - rel_x)) + (x2 * rel_x)
                org_y = (y1 * (1 - rel_y)) + (y2 * rel_y) + y_shift
                org = (org_x, org_y)
                image = kwimage.draw_text_on_image(
                    image, text=label, org=org, color=col, valign=valign,
                    halign=halign, **fontkw)
            if alpha_ < 1.0:
                # We could get away with only doing this to a slice of the
                # image. It might result in a significant speedup. We would
                # need to know the bounding region of the modified pixels,
                # which could be tricky if there are labels.
                cv2.addWeighted(
                    src1=image, alpha=alpha_,
                    src2=background, beta=1 - alpha_,
                    gamma=0, dst=image)

        image = dtype_fixer(image, copy=False)
        return image


class Boxes(_BoxConversionMixins, _BoxPropertyMixins, _BoxTransformMixins,
            _BoxDrawMixins, ub.NiceRepr):  # _generic.Spatial
    r"""
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    This is a convinience class, and should not not store the data for very
    long. The general idiom should be create class, convert data, and then get
    the raw data and let the class be garbage collected. This will help ensure
    that your code is portable and understandable if this class is not
    available.

    This class is meant to efficiently store and manipulate multiple boxes. In
    the case of a single box the :class:`kwimage.structs.single_box.Box` class
    can be used instead.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import kwimage
        >>> import numpy as np
        >>> # Given an array / tensor that represents one or more boxes
        >>> data = np.array([[ 0,  0, 10, 10],
        >>>                  [ 5,  5, 50, 50],
        >>>                  [20,  0, 30, 10]])
        >>> # The kwimage.Boxes data structure is a thin fast wrapper
        >>> # that provides methods for operating on the boxes.
        >>> # It requires that the user explicitly provide a code that denotes
        >>> # the format of the boxes (i.e. what each column represents)
        >>> boxes = kwimage.Boxes(data, 'ltrb')
        >>> # This means that there is no ambiguity about box format
        >>> # The representation string of the Boxes object demonstrates this
        >>> print('boxes = {!r}'.format(boxes))
        boxes = <Boxes(ltrb,
            array([[ 0,  0, 10, 10],
                   [ 5,  5, 50, 50],
                   [20,  0, 30, 10]]))>
        >>> # if you pass this data around. You can convert to other formats
        >>> # For docs on available format codes see :class:`BoxFormat`.
        >>> # In this example we will convert (left, top, right, bottom)
        >>> # to (left-x, top-y, width, height).
        >>> boxes.toformat('xywh')
        <Boxes(xywh,
            array([[ 0,  0, 10, 10],
                   [ 5,  5, 45, 45],
                   [20,  0, 10, 10]]))>
        >>> # In addition to format conversion there are other operations
        >>> # We can quickly (using a C-backend) find IoUs
        >>> ious = boxes.ious(boxes)
        >>> print('{}'.format(ub.repr2(ious, nl=1, precision=2, with_dtype=False)))
        np.array([[1.  , 0.01, 0.  ],
                  [0.01, 1.  , 0.02],
                  [0.  , 0.02, 1.  ]])
        >>> # We can ask for the area of each box
        >>> print('boxes.area = {}'.format(ub.repr2(boxes.area, nl=0, with_dtype=False)))
        boxes.area = np.array([[ 100],[2025],[ 100]])
        >>> # We can ask for the center of each box
        >>> print('boxes.center = {}'.format(ub.repr2(boxes.center, nl=1, with_dtype=False)))
        boxes.center = (
            np.array([[ 5. ],[27.5],[25. ]]),
            np.array([[ 5. ],[27.5],[ 5. ]]),
        )
        >>> # We can translate / scale the boxes
        >>> boxes.translate((10, 10)).scale(100)
        <Boxes(ltrb,
            array([[1000., 1000., 2000., 2000.],
                   [1500., 1500., 6000., 6000.],
                   [3000., 1000., 4000., 2000.]]))>
        >>> # We can clip the bounding boxes
        >>> boxes.translate((10, 10)).scale(100).clip(1200, 1200, 1700, 1800)
        <Boxes(ltrb,
            array([[1200., 1200., 1700., 1800.],
                   [1500., 1500., 1700., 1800.],
                   [1700., 1200., 1700., 1800.]]))>
        >>> # We can perform arbitrary warping of the boxes
        >>> # (note that if the transform is not axis aligned, the axis aligned
        >>> #  bounding box of the transform result will be returned)
        >>> transform = np.array([[-0.83907153,  0.54402111,  0. ],
        >>>                       [-0.54402111, -0.83907153,  0. ],
        >>>                       [ 0.        ,  0.        ,  1. ]])
        >>> boxes.warp(transform)
        <Boxes(ltrb,
            array([[ -8.3907153 , -13.8309264 ,   5.4402111 ,   0.        ],
                   [-39.23347095, -69.154632  ,  23.00569785,  -6.9154632 ],
                   [-25.1721459 , -24.7113486 , -11.3412195 , -10.8804222 ]]))>
        >>> # Note, that we can transform the box to a Polygon for more
        >>> # accurate warping.
        >>> transform = np.array([[-0.83907153,  0.54402111,  0. ],
        >>>                       [-0.54402111, -0.83907153,  0. ],
        >>>                       [ 0.        ,  0.        ,  1. ]])
        >>> warped_polys = boxes.to_polygons().warp(transform)
        >>> print(ub.repr2(warped_polys.data, sv=1))
        [
            <Polygon({
                'exterior': <Coords(data=
                                array([[  0.       ,   0.       ],
                                       [  5.4402111,  -8.3907153],
                                       [ -2.9505042, -13.8309264],
                                       [ -8.3907153,  -5.4402111],
                                       [  0.       ,   0.       ]]))>,
                'interiors': [],
            })>,
            <Polygon({
                'exterior': <Coords(data=
                                array([[ -1.4752521 ,  -6.9154632 ],
                                       [ 23.00569785, -44.67368205],
                                       [-14.752521  , -69.154632  ],
                                       [-39.23347095, -31.39641315],
                                       [ -1.4752521 ,  -6.9154632 ]]))>,
                'interiors': [],
            })>,
            <Polygon({
                'exterior': <Coords(data=
                                array([[-16.7814306, -10.8804222],
                                       [-11.3412195, -19.2711375],
                                       [-19.7319348, -24.7113486],
                                       [-25.1721459, -16.3206333],
                                       [-16.7814306, -10.8804222]]))>,
                'interiors': [],
            })>,
        ]
        >>> # The kwimage.Boxes data structure is also convertable to
        >>> # several alternative data structures, like shapely, coco, and imgaug.
        >>> print(ub.repr2(boxes.to_shapely(), sv=1))
        [
            POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0)),
            POLYGON ((5 5, 5 50, 50 50, 50 5, 5 5)),
            POLYGON ((20 0, 20 10, 30 10, 30 0, 20 0)),
        ]
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> print(ub.repr2(boxes[0:1].to_imgaug(shape=(100, 100)), sv=1))
        BoundingBoxesOnImage([BoundingBox(x1=0.0000, y1=0.0000, x2=10.0000, y2=10.0000, label=None)], shape=(100, 100))
        >>> # xdoctest: -REQUIRES(module:imgaug)
        >>> print(ub.repr2(list(boxes.to_coco()), sv=1))
        [
            [0, 0, 10, 10],
            [5, 5, 45, 45],
            [20, 0, 10, 10],
        ]
        >>> # Finally, when you are done with your boxes object, you can
        >>> # unwrap the raw data by using the ``.data`` attribute
        >>> # all operations are done on this data, which gives the
        >>> # kwiamge.Boxes data structure almost no overhead when
        >>> # inserted into existing code.
        >>> print('boxes.data =\n{}'.format(ub.repr2(boxes.data, nl=1)))
        boxes.data =
        np.array([[ 0,  0, 10, 10],
                  [ 5,  5, 50, 50],
                  [20,  0, 30, 10]], dtype=np.int64)
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> # This data structure was designed for use with both torch
        >>> # and numpy, the underlying data can be either an array or tensor.
        >>> boxes.tensor()
        <Boxes(ltrb,
            tensor([[ 0,  0, 10, 10],
                    [ 5,  5, 50, 50],
                    [20,  0, 30, 10]]))>
        >>> boxes.numpy()
        <Boxes(ltrb,
            array([[ 0,  0, 10, 10],
                   [ 5,  5, 50, 50],
                   [20,  0, 30, 10]]))>

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> from kwimage.structs.boxes import *  # NOQA
        >>> # Demo of conversion methods
        >>> import kwimage
        >>> kwimage.Boxes([[25, 30, 15, 10]], 'xywh')
        <Boxes(xywh, array([[25, 30, 15, 10]]))>
        >>> kwimage.Boxes([[25, 30, 15, 10]], 'xywh').to_xywh()
        <Boxes(xywh, array([[25, 30, 15, 10]]))>
        >>> kwimage.Boxes([[25, 30, 15, 10]], 'xywh').to_cxywh()
        <Boxes(cxywh, array([[32.5, 35. , 15. , 10. ]]))>
        >>> kwimage.Boxes([[25, 30, 15, 10]], 'xywh').to_ltrb()
        <Boxes(ltrb, array([[25, 30, 40, 40]]))>
        >>> kwimage.Boxes([[25, 30, 15, 10]], 'xywh').scale(2).to_ltrb()
        <Boxes(ltrb, array([[50., 60., 80., 80.]]))>
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> kwimage.Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'xywh').scale(.1).to_ltrb()
        <Boxes(ltrb, tensor([[ 2.5000,  3.0000,  4.0000,  5.0000]]))>

    Note:
        In the following examples we show cases where :class:`Boxes` can hold a
        single 1-dimensional box array. This is a holdover from an older
        codebase, and some functions may assume that the input is at least 2-D.
        Thus when representing a single bounding box it is best practice to
        view it as a list of 1 box. While many function will work in the 1-D
        case, not all functions have been tested and thus we cannot gaurentee
        correctness.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> Boxes([25, 30, 15, 10], 'xywh')
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_xywh()
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_ltrb()
        <Boxes(ltrb, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').scale(2).to_ltrb()
        <Boxes(ltrb, array([50., 60., 80., 80.]))>
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'xywh').scale(.1).to_ltrb()
        <Boxes(ltrb, tensor([[ 2.5000,  3.0000,  4.0000,  5.0000]]))>

    Example:
        >>> datas = [
        >>>     [1, 2, 3, 4],
        >>>     [[1, 2, 3, 4], [4, 5, 6, 7]],
        >>>     [[[1, 2, 3, 4], [4, 5, 6, 7]]],
        >>> ]
        >>> formats = BoxFormat.cannonical
        >>> for format1 in formats:
        >>>     for data in datas:
        >>>         self = box1 = Boxes(data, format1)
        >>>         for format2 in formats:
        >>>             box2 = box1.toformat(format2)
        >>>             back = box2.toformat(format1)
        >>>             assert box1 == back
    """
    # __slots__ = ('data', 'format',)

    def __init__(self, data, format=None, check=True):
        """
        Args:
            data (ndarray | Tensor | Boxes) : Either an ndarray or Tensor
                with trailing shape of 4, or an existing Boxes object.

            format (str): format code indicating which coordinates are
                represented by data. If data is a Boxes object then this is
                not necessary.

            check (bool) : if True runs input checks on raw data.

        Raises:
            ValueError : if data is specified without a format
        """
        if (isinstance(data, Boxes) or (data.__class__.__name__ == 'Boxes' and
                                        hasattr(data, 'format') and
                                        hasattr(data, 'data'))):
            if format is not None:
                data = data.toformat(format).data
            else:
                data = data.data
                format = data.format
        elif isinstance(data, (list, tuple)):
            data = np.asarray(data)

        if format is None:
            raise ValueError('Must specify format of raw box data')

        format = BoxFormat.aliases.get(format, format)

        if check:
            if _numel(data) > 0 and data.shape[-1] != 4:
                got = data.shape[-1]
                raise ValueError(
                    'Trailing dimension of boxes must be 4. Got {}'.format(got)
                )

        self.data = data
        self.format = format

    def __getitem__(self, index):
        cls = self.__class__
        subset = cls(self.data[index], self.format)
        return subset

    def __eq__(self, other):
        """
        Tests equality of two Boxes objects

        Example:
            >>> box0 = box1 = Boxes([[1, 2, 3, 4]], 'xywh')
            >>> box2 = Boxes(box0.data, 'ltrb')
            >>> box3 = Boxes([[0, 2, 3, 4]], box0.format)
            >>> box4 = Boxes(box0.data, box2.format)
            >>> assert box0 == box1
            >>> assert not box0 == box2
            >>> assert not box2 == box3
            >>> assert box2 == box4
        """
        return (np.array_equal(self.data, other.data) and
                self.format == other.format)

    def __len__(self):
        """
        Returns:
            int: number of boxes
        """
        return len(self.data)

    def __nice__(self):
        """
        Returns:
            str:  the nice repr
        """
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        nice = '{}, {}'.format(self.format, data_repr)
        return nice

    def __repr__(self):
        return super().__str__()

    @classmethod
    def random(Boxes, num=1, scale=1.0, format=BoxFormat.XYWH, anchors=None,
               anchor_std=1.0 / 6, tensor=False, rng=None):
        """
        Makes random boxes; typically for testing purposes

        Args:
            num (int): number of boxes to generate
            scale (float | Tuple[float, float]): size of imgdims
            format (str): format of boxes to be created (e.g. ltrb, xywh)
            anchors (ndarray): normalized width / heights of anchor boxes to
                perterb and randomly place. (must be in range 0-1)
            anchor_std (float): magnitude of noise applied to anchor shapes
            tensor (bool): if True, returns boxes in tensor format
            rng (None | int | RandomState): initial random seed

        Returns:
            Boxes: random boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes.random(3, rng=0, scale=100)
            <Boxes(xywh,
                array([[54, 54,  6, 17],
                       [42, 64,  1, 25],
                       [79, 38, 17, 14]]))>
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> Boxes.random(3, rng=0, scale=100).tensor()
            <Boxes(xywh,
                tensor([[ 54,  54,   6,  17],
                        [ 42,  64,   1,  25],
                        [ 79,  38,  17,  14]]))>
            >>> anchors = np.array([[.5, .5], [.3, .3]])
            >>> Boxes.random(3, rng=0, scale=100, anchors=anchors)
            <Boxes(xywh,
                array([[ 2, 13, 51, 51],
                       [32, 51, 32, 36],
                       [36, 28, 23, 26]]))>

        Example:
            >>> # Boxes position/shape within 0-1 space should be uniform.
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> fig.gca().set_xlim(0, 128)
            >>> fig.gca().set_ylim(0, 128)
            >>> import kwimage
            >>> kwimage.Boxes.random(num=10).scale(128).draw()
        """
        rng = kwarray.ensure_rng(rng)

        if ub.iterable(scale):
            as_integer = all(isinstance(s, int) for s in scale)
        else:
            as_integer = isinstance(scale, int)

        if anchors is None:
            ltrb = rng.rand(num, 4).astype(np.float32)

            tl_x = np.minimum(ltrb[:, 0], ltrb[:, 2])
            tl_y = np.minimum(ltrb[:, 1], ltrb[:, 3])
            br_x = np.maximum(ltrb[:, 0], ltrb[:, 2])
            br_y = np.maximum(ltrb[:, 1], ltrb[:, 3])

            ltrb[:, 0] = tl_x
            ltrb[:, 1] = tl_y
            ltrb[:, 2] = br_x
            ltrb[:, 3] = br_y
        else:
            anchors = np.asarray(anchors, dtype=np.float32)
            if np.any(anchors > 1.0) or np.any(anchors < 0.0):
                raise ValueError('anchors must be normalized')
            anchor_xs = rng.randint(0, len(anchors), size=num)
            base_whs = anchors[anchor_xs]
            rand_whs = np.clip(
                base_whs * np.exp(rng.randn(num, 2) * anchor_std), 0, 1)
            # Allow cxy to vary within the allowed range
            min_cxy = rand_whs / 2
            max_cxy = (1 - min_cxy)
            rel_cxy = rng.rand(num, 2).astype(np.float32) * .99
            rand_cxwy = rel_cxy * (max_cxy - min_cxy) + min_cxy
            cxywh = np.hstack([rand_cxwy, rand_whs])
            ltrb = Boxes(cxywh, BoxFormat.CXYWH, check=False).to_ltrb().data

        boxes = Boxes(ltrb, format=BoxFormat.LTRB, check=False)
        boxes = boxes.scale(scale, inplace=True)
        if as_integer:
            boxes.data = boxes.data.astype(int)
        boxes = boxes.toformat(format, copy=False)
        if tensor:
            boxes = boxes.tensor()
        return boxes

    def copy(self):
        """
        Returns:
            Boxes: a copy of these boxes
        """
        new_data = _copy(self.data)
        return Boxes(new_data, self.format, check=False)

    @classmethod
    def concatenate(cls, boxes, axis=0):
        """
        Concatenates multiple boxes together

        Args:
            boxes (Sequence[Boxes]): list of boxes to concatenate
            axis (int): axis to stack on. Defaults to 0.

        Returns:
            Boxes: stacked boxes

        Example:
            >>> boxes = [Boxes.random(3) for _ in range(3)]
            >>> new = Boxes.concatenate(boxes)
            >>> assert len(new) == 9
            >>> assert np.all(new.data[3:6] == boxes[1].data)

        Example:
            >>> boxes = [Boxes.random(3) for _ in range(3)]
            >>> boxes[0].data = boxes[0].data[0]
            >>> boxes[1].data = boxes[0].data[0:0]
            >>> new = Boxes.concatenate(boxes)
            >>> assert len(new) == 4
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> new = Boxes.concatenate([b.tensor() for b in boxes])
            >>> assert len(new) == 4
        """
        if len(boxes) == 0:
            raise ValueError('need at least one box to concatenate')
        if axis != 0:
            raise ValueError('can only concatenate along axis=0')
        format = boxes[0].format
        datas = [_view(b.toformat(format).data, -1, 4) for b in boxes]
        newdata = _cat(datas, axis=0)
        new = cls(newdata, format)
        return new

    def compress(self, flags, axis=0, inplace=False):
        """
        Filters boxes based on a boolean criterion

        Args:
            flags (ArrayLike): true for items to be kept.
                Extended type: ArrayLike[bool]
            axis (int): you usually want this to be 0
            inplace (bool): if True, modifies this object

        Returns:
            Boxes: the boxes corresponding to where flags were true

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'ltrb')
            >>> self.compress([True])
            <Boxes(ltrb, array([[25, 30, 15, 10]]))>
            >>> self.compress([False])
            <Boxes(ltrb, array([], shape=(0, 4), dtype=int64))>
        """
        if len(self.data.shape) != 2 and _numel(self.data) > 0:
            raise ValueError('data must be 2d got {}d'.format(
                len(self.data.shape)))
        newdata = _compress(self.data, flags, axis=axis)
        if inplace:
            self.data = newdata
            new = self
        else:
            new = self.__class__(newdata, self.format)
        return new

    def take(self, idxs, axis=0, inplace=False):
        """
        Takes a subset of items at specific indices

        Args:
            indices (ArrayLike):
                Indexes of items to take.  Extended type ArrayLike[int].
            axis (int): you usually want this to be 0
            inplace (bool): if True, modifies this object

        Returns:
            Boxes: the boxes corresponding to the specified indices

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'ltrb')
            >>> self.take([0])
            <Boxes(ltrb, array([[25, 30, 15, 10]]))>
            >>> self.take([])
            <Boxes(ltrb, array([], shape=(0, 4), dtype=int64))>
        """
        if len(self.data.shape) != 2 and _numel(self.data) > 0:
            raise ValueError('data must be 2d got {}d'.format(
                len(self.data.shape)))
        if inplace:
            newdata = _take(self.data, idxs, axis=axis)
            self.data = newdata
            new = self
        else:
            newdata = _take(self.data, idxs, axis=axis)
            new = self.__class__(newdata, self.format)
        return new

    def is_tensor(self):
        """
        is the backend fueled by torch?

        Returns:
            bool: True if the Boxes are torch tensors
        """
        return torch is not None and torch.is_tensor(self.data)

    def is_numpy(self):
        """
        is the backend fueled by numpy?

        Returns:
            bool: True if the Boxes are numpy arrays
        """
        return isinstance(self.data, np.ndarray)

    @ub.memoize_property
    def _impl(self):
        """
        returns the kwarray.ArrayAPI implementation for the data

        Returns:
            kwarray.ArrayAPI: the array API for the box backend

        Example:
            >>> assert Boxes.random().numpy()._impl.is_numpy
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> assert Boxes.random().tensor()._impl.is_tensor
        """
        return kwarray.ArrayAPI.coerce(self.data)

    @property
    def device(self):
        """
        If the backend is torch returns the data device, otherwise None
        """
        try:
            return self.data.device
        except AttributeError:
            return None

    def astype(self, dtype):
        """
        Changes the type of the internal array used to represent the boxes

        Note:
            this operation is not inplace

        Returns:
            Boxes: the boxes with the chosen type

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> Boxes.random(3, 100, rng=0).tensor().astype('int32')
            <Boxes(xywh,
                tensor([[54, 54,  6, 17],
                        [42, 64,  1, 25],
                        [79, 38, 17, 14]], dtype=torch.int32))>
            >>> Boxes.random(3, 100, rng=0).numpy().astype('int32')
            <Boxes(xywh,
                array([[54, 54,  6, 17],
                       [42, 64,  1, 25],
                       [79, 38, 17, 14]], dtype=int32))>
            >>> Boxes.random(3, 100, rng=0).tensor().astype('float32')
            >>> Boxes.random(3, 100, rng=0).numpy().astype('float32')
        """
        data = self.data
        if torch is not None and torch.is_tensor(data):
            dtype = _rectify_torch_dtype(dtype)
            newself = self.__class__(data.to(dtype), self.format)
        else:
            newself = self.__class__(data.astype(dtype), self.format)
        return newself

    def round(self, inplace=False):
        """
        Rounds data coordinates to the nearest integer.

        This operation is applied directly to the box coordinates, so its
        output will depend on the format the boxes are stored in.

        Args:
            inplace (bool): if True, modifies this object. Defaults to False.

        Returns:
            Boxes: the boxes with rounded coordinates

        SeeAlso:
            :func:`Boxes.quantize`

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(3, rng=0).scale(10)
            >>> new = self.round()
            >>> print('self = {!r}'.format(self))
            >>> print('new = {!r}'.format(new))
            self = <Boxes(xywh,
                array([[5.48813522, 5.44883192, 0.53949833, 1.70306146],
                       [4.23654795, 6.4589411 , 0.13932407, 2.45878875],
                       [7.91725039, 3.83441508, 1.71937704, 1.45453393]]))>
            new = <Boxes(xywh,
                array([[5., 5., 1., 2.],
                       [4., 6., 0., 2.],
                       [8., 4., 2., 1.]]))>
        """
        new = self if inplace else self.__class__(self.data, self.format)
        new.data = self._impl.round(new.data)
        return new

    def quantize(self, inplace=False, dtype=np.int32):
        """
        Converts the box to integer coordinates.

        This operation takes the floor of the left side and the ceil of the
        right side. Thus the area of the box will never decreases. But this
        will often increase the width / height of the box by a pixel.

        Args:
            inplace (bool): if True, modifies this object
            dtype (type): type to cast as

        Returns:
            Boxes: the boxes with quantized coordinates

        SeeAlso:
            :func:`Boxes.round`
            :func:`Boxes.resize` if you need to ensure the size does not change

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(3, rng=0).scale(10)
            >>> new = self.quantize()
            >>> print('self = {!r}'.format(self))
            >>> print('new = {!r}'.format(new))
            self = <Boxes(xywh,
                array([[5.48813522, 5.44883192, 0.53949833, 1.70306146],
                       [4.23654795, 6.4589411 , 0.13932407, 2.45878875],
                       [7.91725039, 3.83441508, 1.71937704, 1.45453393]]))>
            new = <Boxes(xywh,
                array([[5, 5, 2, 3],
                       [4, 6, 1, 3],
                       [7, 3, 3, 3]], dtype=int32))>

        Example:
            >>> import kwimage
            >>> # Be careful if it is important to preserve the width/height
            >>> self = kwimage.Boxes([[0, 0, 10, 10]], 'xywh')
            >>> aff = kwimage.Affine.coerce(offset=(0.5, 0.0))
            >>> warped = self.warp(aff)
            >>> new = warped.quantize(dtype=int)
            >>> print('self   = {!r}'.format(self))
            >>> print('warped = {!r}'.format(warped))
            >>> print('new    = {!r}'.format(new))
            self   = <Boxes(xywh, array([[ 0,  0, 10, 10]]))>
            warped = <Boxes(xywh, array([[ 0.5,  0. , 10. , 10. ]]))>
            new    = <Boxes(xywh, array([[ 0,  0, 11, 10]]))>

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(3, rng=0)
            >>> orig = self.copy()
            >>> self.quantize(inplace=True)
            >>> assert np.any(self.data != orig.data)
        """
        new = self if inplace else self.__class__(self.data, self.format)
        _impl = self._impl
        _ceil = _impl.ceil
        _floor = _impl.floor
        _astype = _impl.astype

        ltrb_box = new.to_ltrb(copy=False)
        ltrb = ltrb_box.data
        new_ltrb = _impl.empty_like(ltrb, dtype=dtype)

        new_ltrb[..., 0:2] = _astype(_floor(ltrb[..., 0:2]), dtype=dtype)
        new_ltrb[..., 2:4] = _astype(_ceil(ltrb[..., 2:4]), dtype=dtype)

        # new_ltrb[..., 0] = _astype(_floor(ltrb[..., 0]), dtype=dtype)
        # new_ltrb[..., 1] = _astype(_floor(ltrb[..., 1]), dtype=dtype)
        # new_ltrb[..., 2] = _astype(_ceil(ltrb[..., 2]), dtype=dtype)
        # new_ltrb[..., 3] = _astype(_ceil(ltrb[..., 3]), dtype=dtype)

        ltrb_box.data = new_ltrb
        new.data = ltrb_box.toformat(new.format, copy=False).data
        return new

    def numpy(self):
        """
        Converts tensors to numpy. Does not change memory if possible.

        Returns:
            Boxes: the boxes with a numpy backend

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Boxes.random(3).tensor()
            >>> newself = self.numpy()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        data = self.data
        if torch is not None and torch.is_tensor(data):
            data = self._impl.numpy(data.data)
            # data = data.data.cpu().numpy()
        newself = self.__class__(data, self.format)
        return newself

    def tensor(self, device=ub.NoParam):
        """
        Converts numpy to tensors. Does not change memory if possible.

        Args:
            device (int | None | torch.device):
                The torch device to put the backend tensors on

        Returns:
            Boxes: the boxes with a torch backend

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Boxes.random(3)
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> newself = self.tensor()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        if torch is None:
            raise Exception('torch is not available')
        data = self.data
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if device is not ub.NoParam:
            data = data.to(device)
        newself = self.__class__(data, self.format)
        return newself

    def ious(self, other, bias=0, impl='auto', mode=None):
        """
        Intersection over union.

        Compute IOUs (intersection area over union area) between these boxes
        and another set of boxes. This is a symmetric measure of similarity
        between boxes.

        TODO:
            - [ ] Add pairwise flag to toggle between one-vs-one and all-vs-all
                  computation. I.E. Add option for componentwise calculation.

        Args:
            other (Boxes): boxes to compare IoUs against

            bias (int):
                either 0 or 1, does TL=BR have area of 0 or 1?
                Defaults to 0.

            impl (str): code to specify implementation used to
                ious. Can be either torch, py, c, or auto. Efficiency and the
                exact result will vary by implementation, but they will always
                be close.  Some implementations only accept certain data types
                (e.g.  impl='c', only accepts float32 numpy arrays).  See
                ~/code/kwimage/dev/bench_bbox.py for benchmark details. On my
                system the torch impl was fastest (when the data was on the
                GPU). Defaults to 'auto'

            mode (str) : depricated, use impl

        Returns:
            ndarray: the ious

        SeeAlso:
            iooas - for a measure of coverage between boxes

        Examples:
            >>> import kwimage
            >>> self = kwimage.Boxes(np.array([[ 0,  0, 10, 10],
            >>>                                [10,  0, 20, 10],
            >>>                                [20,  0, 30, 10]]), 'ltrb')
            >>> other = kwimage.Boxes(np.array([6, 2, 20, 10]), 'ltrb')
            >>> overlaps = self.ious(other, bias=1).round(2)
            >>> assert np.all(np.isclose(overlaps, [0.21, 0.63, 0.04])), repr(overlaps)


        Examples:
            >>> import kwimage
            >>> boxes1 = kwimage.Boxes(np.array([[ 0,  0, 10, 10],
            >>>                                  [10,  0, 20, 10],
            >>>                                  [20,  0, 30, 10]]), 'ltrb')
            >>> other = kwimage.Boxes(np.array([[6, 2, 20, 10],
            >>>                                 [100, 200, 300, 300]]), 'ltrb')
            >>> overlaps = boxes1.ious(other)
            >>> print('{}'.format(ub.repr2(overlaps, precision=2, nl=1)))
            np.array([[0.18, 0.  ],
                      [0.61, 0.  ],
                      [0.  , 0.  ]]...)

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes(np.empty(0), 'xywh').ious(Boxes(np.empty(4), 'xywh')).shape
            (0,)
            >>> #Boxes(np.empty(4), 'xywh').ious(Boxes(np.empty(0), 'xywh')).shape
            >>> Boxes(np.empty((0, 4)), 'xywh').ious(Boxes(np.empty((0, 4)), 'xywh')).shape
            (0, 0)
            >>> Boxes(np.empty((1, 4)), 'xywh').ious(Boxes(np.empty((0, 4)), 'xywh')).shape
            (1, 0)
            >>> Boxes(np.empty((0, 4)), 'xywh').ious(Boxes(np.empty((1, 4)), 'xywh')).shape
            (0, 1)

        Examples:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> formats = BoxFormat.cannonical
            >>> istensors = [False, True]
            >>> results = {}
            >>> for format in formats:
            >>>     for tensor in istensors:
            >>>         boxes1 = Boxes.random(5, scale=10.0, rng=0, format=format, tensor=tensor)
            >>>         boxes2 = Boxes.random(7, scale=10.0, rng=1, format=format, tensor=tensor)
            >>>         ious = boxes1.ious(boxes2)
            >>>         results[(format, tensor)] = ious
            >>> results = {k: v.numpy() if torch.is_tensor(v) else v for k, v in results.items() }
            >>> results = {k: v.tolist() for k, v in results.items()}
            >>> print(ub.repr2(results, sk=True, precision=3, nl=2))
            >>> from functools import partial
            >>> assert ub.allsame(results.values(), partial(np.allclose, atol=1e-07))

        Ignore:
            >>> # does this work with backprop?
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import torch
            >>> import kwimage
            >>> num = 1000
            >>> true_boxes = kwimage.Boxes.random(num).tensor()
            >>> inputs = torch.rand(num, 10)
            >>> regress = torch.nn.Linear(10, 4)
            >>> energy = regress(inputs)
            >>> energy.retain_grad()
            >>> outputs = energy.sigmoid()
            >>> outputs.retain_grad()
            >>> out_boxes = kwimage.Boxes(outputs, 'cxywh')
            >>> ious = out_boxes.ious(true_boxes)
            >>> loss = ious.sum()
            >>> loss.backward()

        """
        other_is_1d = len(other) > 0 and (len(other.shape) == 1)
        if other_is_1d:
            # `box_ious` expects 2d input
            other = other[None, :]

        # self_is_1d = (len(self.shape) == 1)
        # if self_is_1d:
        #     self = self[None, :]

        if len(other) == 0 or len(self) == 0:
            if torch is not None and (torch.is_tensor(self.data) or torch.is_tensor(other.data)):
                if _TORCH_HAS_EMPTY_SHAPE:
                    ious = torch.empty((len(self), len(other)))
                else:
                    ious = torch.empty(0)
            else:
                ious = np.empty((len(self), len(other)))
        else:
            self_ltrb = self.to_ltrb(copy=False)
            other_ltrb = other.to_ltrb(copy=False)

            if mode is not None:
                ub.schedule_deprecation(
                    'kwimage', 'mode', 'argument to Boxes.ious',
                    migration='Use impl instead.', deprecate='0.9.0',
                    error='0.10.0', remove='0.11.0')
                impl = mode

            ious = box_ious(self_ltrb.data, other_ltrb.data, bias=bias,
                            impl=impl)

        if other_is_1d:
            ious = ious[..., 0]

        # if self_is_1d:
        #     ious = ious[0, ...]
        return ious

    def iooas(self, other, bias=0):
        """
        Intersection over other area.

        This is an asymetric measure of coverage. How much of the "other" boxes
        are covered by these boxes. It is the area of intersection between each
        pair of boxes and the area of the "other" boxes.

        SeeAlso:
            ious - for a measure of similarity between boxes

        Args:
            other (Boxes):
                boxes to compare IoOA against

            bias (int):
                either 0 or 1, does TL=BR have area of 0 or 1? Defaults to 0.

        Returns:
            ndarray: the iooas

        Examples:
            >>> self = Boxes(np.array([[ 0,  0, 10, 10],
            >>>                        [10,  0, 20, 10],
            >>>                        [20,  0, 30, 10]]), 'ltrb')
            >>> other = Boxes(np.array([[6, 2, 20, 10], [0, 0, 0, 3]]), 'xywh')
            >>> coverage = self.iooas(other, bias=0).round(2)
            >>> print('coverage = {!r}'.format(coverage))
        """
        numer = self.isect_area(other, bias=bias)
        denom = other.area.T
        # If the denom is zero the numer must also be zero, and the overlap is
        # zero, so this is safe.
        denom[denom == 0] = 1
        iooas = numer / denom
        return iooas

    def isect_area(self, other, bias=0):
        """
        Intersection part of intersection over union computation

        Args:
            other (Boxes):
                boxes to compare IoOA against

            bias (int):
                either 0 or 1, does TL=BR have area of 0 or 1? Defaults to 0.

        Returns:
            ndarray: the iooas

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> self = Boxes.random(5, scale=10.0, rng=0, format='ltrb')
            >>> other = Boxes.random(3, scale=10.0, rng=1, format='ltrb')
            >>> isect = self.isect_area(other, bias=0)
            >>> ious_v1 = isect / ((self.area + other.area.T) - isect)
            >>> ious_v2 = self.ious(other, bias=0)
            >>> assert np.allclose(ious_v1, ious_v2)
        """
        other_is_1d = (len(other.shape) == 1)
        if other_is_1d:
            other = other[None, :]

        if len(other) == 0 or len(self) == 0:
            if torch is not None and (torch.is_tensor(self.data) or torch.is_tensor(other.data)):
                if _TORCH_HAS_EMPTY_SHAPE:
                    isect = torch.empty((len(self), len(other)))
                else:
                    isect = torch.empty(0)
            else:
                isect = np.empty((len(self), len(other)))
        else:
            self_ltrb = self.to_ltrb(copy=False)
            other_ltrb = other.to_ltrb(copy=False)

            _impl = self._impl
            isect = _isect_areas(self_ltrb.data, other_ltrb.data, _impl=_impl)

        if other_is_1d:
            isect = isect[..., 0]
        return isect

    def intersection(self, other):
        """
        Componentwise intersection between two sets of Boxes

        intersections of boxes are always boxes, so this works

        Args:
            other (Boxes): boxes to intersect with this object.
                (must be of same length)

        Returns:
            Boxes: the intersection geometry

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(5, rng=0).scale(10.)
            >>> other = self.translate(1)
            >>> new = self.intersection(other)
            >>> new_area = np.nan_to_num(new.area).ravel()
            >>> alt_area = np.diag(self.isect_area(other))
            >>> close = np.isclose(new_area, alt_area)
            >>> assert np.all(close)
        """
        other_is_1d = (len(other.shape) == 1)
        if other_is_1d:
            other = other[None, :]

        self_ltrb = self.to_ltrb(copy=False).data
        other_ltrb = other.to_ltrb(copy=False).data

        tl = np.maximum(self_ltrb[..., :2], other_ltrb[..., :2])
        br = np.minimum(self_ltrb[..., 2:], other_ltrb[..., 2:])

        is_bad = np.any(tl > br, axis=1)
        ltrb = np.concatenate([tl, br], axis=-1)

        ltrb[is_bad] = np.nan

        isect = Boxes(ltrb, 'ltrb')

        return isect

    def union_hull(self, other):
        """
        Componentwise hull union between two sets of Boxes

        NOTE: convert to polygon to do a real union.

        Args:
            other (Boxes): boxes to union with this object.
                (must be of same length)

        Returns:
            Boxes: unioned boxes

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(5, rng=0).scale(10.)
            >>> other = self.translate(1)
            >>> new = self.union_hull(other)
            >>> new_area = np.nan_to_num(new.area).ravel()
        """
        other_is_1d = (len(other.shape) == 1)
        if other_is_1d:
            other = other[None, :]

        self_ltrb = self.to_ltrb(copy=False).data
        other_ltrb = other.to_ltrb(copy=False).data

        tl = np.minimum(self_ltrb[..., :2], other_ltrb[..., :2])
        br = np.maximum(self_ltrb[..., 2:], other_ltrb[..., 2:])

        is_bad = np.any(tl > br, axis=1)
        ltrb = np.concatenate([tl, br], axis=-1)

        ltrb[is_bad] = np.nan

        isect = Boxes(ltrb, 'ltrb')

        return isect

    def bounding_box(self):
        """
        Returns the box that bounds all of the contained boxes

        Returns:
            Boxes: a single box

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(5, rng=0).scale(10.)
            >>> other = self.translate(1)
            >>> new = self.union_hull(other)
            >>> new_area = np.nan_to_num(new.area).ravel()
        """
        min_xs, min_ys, max_xs, max_ys = self.to_ltrb().components
        min_x = min_xs.min()
        min_y = min_ys.min()
        max_x = max_xs.max()
        max_y = max_ys.max()
        new_ltrb = np.array([[min_x, min_y, max_x, max_y]])
        new = Boxes(new_ltrb, format='ltrb')
        return new

    def contains(self, other):
        """
        Determine of points are completely contained by these boxes

        Args:
            other (kwimage.Points): points to test for containment.
                TODO: support generic data types

        Returns:
            ArrayLike: flags - N x M boolean matrix indicating which box
                contains which points, where N is the number of boxes and
                M is the number of points.

        Examples:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(10).scale(10).round()
            >>> other = kwimage.Points.random(10).scale(10).round()
            >>> flags = self.contains(other)
            >>> flags = self.contains(self.xy_center)
            >>> assert np.all(np.diag(flags))
        """
        ltrb = self.to_ltrb()

        try:
            # other = Points.coerce(other)?
            # points
            pt_x, pt_y = other.xy.T
        except AttributeError:
            # ndarray
            pt_x, pt_y = other.T

        flags = (
            (ltrb.tl_x <= pt_x) &
            (ltrb.tl_y <= pt_y) &
            (ltrb.br_x >= pt_x) &
            (ltrb.br_y >= pt_y)
        )
        return flags

    def view(self, *shape):
        """
        Passthrough method to view or reshape

        Args:
            *shape (Tuple[int, ...]): new shape

        Returns:
            Boxes: data with a different view

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='xywh').tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='ltrb').tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
        """
        data_ = _view(self.data, *shape)
        return self.__class__(data_, self.format)


def _copy(data):
    if torch is not None and torch.is_tensor(data):
        return data.clone()
    else:
        return data.copy()


def _view(data, *shape):
    if torch is not None and torch.is_tensor(data):
        data_ = data.view(*shape)
    else:
        data_ = data.reshape(*shape)
    return data_


def _cat(datas, axis=-1):
    if torch is not None and torch.is_tensor(datas[0]):
        return torch.cat(datas, dim=axis)
    else:
        return np.concatenate(datas, axis=axis)


def _take(data, indices, axis=None):
    """
    compatable take-API between torch and numpy

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> np_data = np.arange(0, 143).reshape(11, 13)
        >>> pt_data = torch.LongTensor(np_data)
        >>> indices = [1, 3, 5, 7, 11, 13, 17, 21]
        >>> idxs0 = [1, 3, 5, 7]
        >>> idxs1 = [1, 3, 5, 7, 11]
        >>> assert np.allclose(_take(np_data, indices), _take(pt_data, indices))
        >>> assert np.allclose(_take(np_data, idxs0, 0), _take(pt_data, idxs0, 0))
        >>> assert np.allclose(_take(np_data, idxs1, 1), _take(pt_data, idxs1, 1))
    """
    if isinstance(data, np.ndarray):
        return data.take(indices, axis=axis)
    elif torch is not None and torch.is_tensor(data):
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(data.device)
        if axis is None:
            return data.take(indices)
        else:
            return torch.index_select(data, dim=axis, index=indices)
    else:
        raise TypeError(type(data))


def _compress(data, flags, axis=None):
    """
    compatable take-API between torch and numpy

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> np_data = np.arange(0, 143).reshape(11, 13)
        >>> pt_data = torch.LongTensor(np_data)
        >>> flags = (np_data % 2 == 0).ravel()
        >>> f0 = (np_data % 2 == 0)[:, 0]
        >>> f1 = (np_data % 2 == 0)[0, :]
        >>> assert np.allclose(_compress(np_data, flags), _compress(pt_data, flags))
        >>> assert np.allclose(_compress(np_data, f0, 0), _compress(pt_data, f0, 0))
        >>> assert np.allclose(_compress(np_data, f1, 1), _compress(pt_data, f1, 1))
    """
    if isinstance(data, np.ndarray):
        return data.compress(flags, axis=axis)
    elif torch is not None and torch.is_tensor(data):
        if not torch.is_tensor(flags):
            if _TORCH_HAS_BOOL_COMP:
                flags = np.asarray(flags, dtype=np.bool)
                flags = torch.BoolTensor(flags).to(data.device)
            else:
                flags = np.asarray(flags).astype(np.uint8)
                flags = torch.ByteTensor(flags).to(data.device)
        if flags.ndimension() != 1:
            raise ValueError('condition must be a 1-d tensor')
        if axis is None:
            return torch.masked_select(data.view(-1), flags)
        else:
            out_shape = list(data.shape)
            fancy_shape = [-1] * len(out_shape)
            out_shape[axis] = int(flags.sum())
            fancy_shape[axis] = flags.numel()
            explicit_flags = flags.view(*fancy_shape).expand_as(data)
            out = torch.masked_select(data, explicit_flags).view(*out_shape)
            return out
    else:
        raise TypeError(type(data))


def _numel(data):
    """ compatable numel-API between torch and numpy """
    if isinstance(data, np.ndarray):
        return data.size
    else:
        return data.numel()


@ub.memoize
def _torch_dtype_lut():
    lut = {}

    # Handle nonstandard alias dtype names
    lut['double'] = torch.double
    lut['long'] = torch.long

    # Handle floats
    for k in [np.float16, 'float16']:
        lut[k] = torch.float16
    for k in [np.float32, 'float32']:
        lut[k] = torch.float32
    for k in [np.float64, 'float64']:
        lut[k] = torch.float64

    if torch.float == torch.float32:
        lut['float'] = torch.float32
    else:
        raise AssertionError('dont think this can happen')

    if np.float_ == np.float32:
        lut[float] = torch.float32
    elif np.float_ == np.float64:
        lut[float] = torch.float64
    else:
        raise AssertionError('dont think this can happen')

    # Handle signed integers
    for k in [np.int8, 'int8']:
        lut[k] = torch.int8
    for k in [np.int16, 'int16']:
        lut[k] = torch.int16
    for k in [np.int32, 'int32']:
        lut[k] = torch.int32
    for k in [np.int64, 'int64']:
        lut[k] = torch.int64

    if np.int_ == np.int32:
        lut[int] = torch.int32
    elif np.int_ == np.int64:
        lut[int] = torch.int64
    else:
        raise AssertionError('dont think this can happen')

    if torch.int == torch.int32:
        lut['int'] = torch.int32
    else:
        raise AssertionError('dont think this can happen')

    # Handle unsigned integers
    for k in [np.uint8, 'uint8']:
        lut[k] = torch.uint8
    return lut


def _rectify_torch_dtype(dtype):
    return _torch_dtype_lut().get(dtype, dtype)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.structs.boxes
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
