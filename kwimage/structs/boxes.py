# -*- coding: utf-8 -*-
"""
Vectorized Bounding Boxes

kwimage.Boxes is a tool for efficiently transporting a set of bounding boxes
within python as well as methods for operating on bounding boxes. It is a VERY
thin wrapper around a pure numpy/torch array/tensor representation, and thus it
is very fast.

Raw bounding boxes come in lots of different formats. There are lots of ways to
parameterize two points! Because of this THE USER MUST ALWAYS BE EXPLICIT ABOUT
THE BOX FORMAT.


There are 3 main bounding box formats:
    xywh: top left xy-coordinates and width height offsets
    cxywh: center xy-coordinates and width height offsets
    tlbr: top left and bottom right xy coordinates

Here is some example usage

Example:
    >>> from kwimage.structs.boxes import Boxes
    >>> data = np.array([[ 0,  0, 10, 10],
    >>>                  [ 5,  5, 50, 50],
    >>>                  [10,  0, 20, 10],
    >>>                  [20,  0, 30, 10]])
    >>> # Note that the format of raw data is ambiguous, so you must specify
    >>> boxes = Boxes(data, 'tlbr')
    >>> print('boxes = {!r}'.format(boxes))
    boxes = <Boxes(tlbr,
        array([[ 0,  0, 10, 10],
               [ 5,  5, 50, 50],
               [10,  0, 20, 10],
               [20,  0, 30, 10]]))>

    >>> # Now you can operate on those boxes easily
    >>> print(boxes.translate((10, 10)))
    <Boxes(tlbr,
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
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import ubelt as ub
import warnings
import skimage
import kwarray
from distutils.version import LooseVersion
from . import _generic  # NOQA

__all__ = ['Boxes']

import os
val = os.environ.get('KWIMAGE_DISABLE_C_EXTENSIONS', '').lower()
DISABLE_C_EXTENSIONS = val in {'true', 'on', 'yes', '1'}

if not DISABLE_C_EXTENSIONS:
    try:
        from ._boxes_backend.cython_boxes import bbox_ious_c as _bbox_ious_c
    except ImportError:
        _bbox_ious_c = None
else:
    _bbox_ious_c = None

_TORCH_HAS_EMPTY_SHAPE = LooseVersion(torch.__version__) >= LooseVersion('1.0.0')
_TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


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
    # xy1xy2 = 'tlbr'
    # xx1yy2 = 'extent'

    # Note: keep the strings as the "old"-style names for now
    # TODO: change the string values to match their associated NameConstant
    #     - [x] bump versions
    #     - [x] use the later in the or statements
    #     - [ ] ensur nothing depends on the old values.

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
    TLBR  = _register('tlbr')   # (x1, y1, x2, y2)
    XXYY  = _register('xxyy')   # (x1, x2, y1, y2)

    # Row-Major-Formats
    # Note: prefix row major format with an underscore.
    # Reason: Boxes prefers column-major formats
    _YYXX  = _register('_yyxx')   # (y1, y2, x1, x2)
    _RCHW  = _register('_rchw')   # (y1, y2, x1, x2)

    aliases = {
        # NOTE: Once a name enters here it is very difficult to remove
        # Once we hit version 1.0, this table cannot be removed from without
        # bumping a major version.
        'xywh': XYWH,
        'tlhw': XYWH,  # todo: remove: does not follow the pattern

        'cxywh': CXYWH,

        'tlbr': TLBR,
        'xyxy': TLBR,
        'xy1xy2': TLBR,

        'xxyy': XXYY,
        'xx1yy2': XXYY,

        # Transposed formats
        '_yyxx': _YYXX,
        '_yy1xx2': _YYXX,

        # Explicit tuple format
        'cx,cy,w,h'           : CXYWH,
        'tl_x,tl_y,w,h'       : XYWH,
        'tl_x,tl_y,br_x,br_y' : TLBR,
        'x1,x2,y1,y2'         : XXYY,
    }
    for key in cannonical:
        aliases[key] = key


def box_ious(tlbr1, tlbr2, bias=0, impl=None):
    """
    Args:
        tlbr1 (ndarray): (N, 4) tlbr format
        tlbr2 (ndarray): (K, 4) tlbr format
        bias (int): either 0 or 1, does tl=br have area of 0 or 1?

    Benchmark:
        See ~/code/kwarray/dev/bench_bbox.py

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> tlbr1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr').data
        >>> tlbr2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr').data
        >>> ious = box_ious(tlbr1, tlbr2)
        >>> print(ub.repr2(ious.tolist(), precision=2))
        [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.01],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02],
            [0.32, 0.02, 0.01, 0.07, 0.24, 0.12, 0.55],
            [0.00, 0.00, 0.00, 0.11, 0.00, 0.12, 0.04],
        ]

    Example:
        >>> tlbr1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr').data
        >>> tlbr2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr').data
        >>> if _bbox_ious_c is not None:
        >>>     ious_c = box_ious(tlbr1, tlbr2, bias=0, impl='c')
        >>>     ious_py = box_ious(tlbr1, tlbr2, bias=0, impl='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
        >>>     ious_c = box_ious(tlbr1, tlbr2, bias=1, impl='c')
        >>>     ious_py = box_ious(tlbr1, tlbr2, bias=1, impl='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
    """
    if impl is None or impl == 'auto':
        if torch.is_tensor(tlbr1):
            impl = 'torch'
        else:
            impl = 'py' if _bbox_ious_c is None else 'c'

    if impl == 'torch' or torch.is_tensor(tlbr1):
        # TODO: add tests for equality with other methods or show why it should
        # be different.
        # NOTE: this is done in boxes.ious
        return _box_ious_torch(tlbr1, tlbr2, bias)
    elif impl == 'c':
        if _bbox_ious_c is None:
            raise Exception('The Boxes C module is not available')
        return _bbox_ious_c(tlbr1.astype(np.float32),
                            tlbr2.astype(np.float32), bias)
    elif impl == 'py':
        return _box_ious_py(tlbr1, tlbr2, bias)
    else:
        raise KeyError(impl)


def _box_ious_torch(tlbr1, tlbr2, bias=0):
    """
    Example:
        >>> tlbr1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr').tensor().data
        >>> tlbr2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr').tensor().data
        >>> bias = 0
        >>> ious = _box_ious_torch(tlbr1, tlbr2, bias)
        >>> ious_np = _box_ious_py(tlbr1.numpy(), tlbr2.numpy(), bias)
        >>> assert np.all(ious_np == ious.numpy())
    """
    # tlbr1 = tlbr1.view(-1, 4)
    # tlbr2 = tlbr2.view(-1, 4)

    w1 = tlbr1[..., 2] - tlbr1[..., 0] + bias
    h1 = tlbr1[..., 3] - tlbr1[..., 1] + bias
    w2 = tlbr2[..., 2] - tlbr2[..., 0] + bias
    h2 = tlbr2[..., 3] - tlbr2[..., 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = torch.min(tlbr1[..., 2][..., None], tlbr2[..., 2])
    x_mins = torch.max(tlbr1[..., 0][..., None], tlbr2[..., 0])

    iws = (x_maxs - x_mins + bias).clamp(0, float('inf'))

    y_maxs = torch.min(tlbr1[..., 3][..., None], tlbr2[..., 3])
    y_mins = torch.max(tlbr1[..., 1][..., None], tlbr2[..., 1])

    ihs = (y_maxs - y_mins + bias).clamp(0, float('inf'))

    areas_sum = (areas1[..., None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


def _box_ious_py(tlbr1, tlbr2, bias=0):
    """
    This is the fastest python implementation of bbox_ious I found
    """
    w1 = tlbr1[:, 2] - tlbr1[:, 0] + bias
    h1 = tlbr1[:, 3] - tlbr1[:, 1] + bias
    w2 = tlbr2[:, 2] - tlbr2[:, 0] + bias
    h2 = tlbr2[:, 3] - tlbr2[:, 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = np.minimum(tlbr1[:, 2][:, None], tlbr2[:, 2])
    x_mins = np.maximum(tlbr1[:, 0][:, None], tlbr2[:, 0])

    iws = np.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = np.minimum(tlbr1[:, 3][:, None], tlbr2[:, 3])
    y_mins = np.maximum(tlbr1[:, 1][:, None], tlbr2[:, 1])

    ihs = np.maximum(y_maxs - y_mins + bias, 0)

    areas_sum = (areas1[:, None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


def _isect_areas(tlbr1, tlbr2, bias=0):
    """
    Returns only the area of the intersection
    """
    x_maxs = np.minimum(tlbr1[:, 2][:, None], tlbr2[:, 2])
    x_mins = np.maximum(tlbr1[:, 0][:, None], tlbr2[:, 0])

    iws = np.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = np.minimum(tlbr1[:, 3][:, None], tlbr2[:, 3])
    y_mins = np.maximum(tlbr1[:, 1][:, None], tlbr2[:, 1])

    ihs = np.maximum(y_maxs - y_mins + bias, 0)

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
            copy (bool, default=True):
                if False, the conversion is done inplace, but only if possible.

        Returns:
            Boxes : transformed boxes

        CommandLine:
            xdoctest -m kwimage.structs.boxes _BoxConversionMixins.toformat

        Example:
            >>> boxes = Boxes.random(2, scale=10, rng=0)

            >>> boxes.toformat('tlbr')
            <Boxes(tlbr,
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
            >>> boxes.toformat('tlbr')
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
            # Only difference between tlbr and extent=xxyy is the column order
            # xxyy: is x1, x2, y1, y2
            tlbr = self.to_tlbr().data
            xxyy = tlbr[..., [0, 2, 1, 3]]
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
        elif self.format == BoxFormat.TLBR:
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
            return self.to_tlbr(copy=copy).to_xywh(copy=copy)
        else:
            raise KeyError(self.format)
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
        elif self.format == BoxFormat.TLBR:
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
            return self.to_tlbr(copy=copy).to_cxywh(copy=copy)
        elif self.format == BoxFormat._RCHW:
            return self.to_tlbr(copy=copy).to_cxywh(copy=copy)
        else:
            raise KeyError(self.format)
        cxywh = _cat([cx, cy, w, h])
        return Boxes(cxywh, BoxFormat.CXYWH, check=False)

    @_register_convertor(BoxFormat.TLBR)
    def to_tlbr(self, copy=True):
        if self.format == BoxFormat.TLBR:
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
            raise KeyError(self.format)
        tlbr = _cat([x1, y1, x2, y2])
        return Boxes(tlbr, BoxFormat.TLBR, check=False)

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
        if self.format == BoxFormat.TLBR:
            _yyxx = self.data[..., [1, 3, 0, 2]]
        else:
            _yyxx = self.to_tlbr(copy)._to_yyxx(copy)
        return Boxes(_yyxx, BoxFormat._YYXX, check=False)

    def to_imgaug(self, shape):
        """
        Args:
            shape (tuple): shape of image that boxes belong to

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> bboi = self.to_imgaug((10, 10))
        """
        import imgaug
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))

        tlbr = self.to_tlbr(copy=False).data
        bbs = [imgaug.BoundingBox(x1, y1, x2, y2) for x1, y1, x2, y2 in tlbr]
        bboi = imgaug.BoundingBoxesOnImage(bbs, shape=shape)
        return bboi

    def to_shapley(self):
        """
        Convert boxes to a list of shapely polygons
        """
        from shapely.geometry import Polygon
        x1, y1, x2, y2 = self.to_tlbr(copy=False).components
        a = _cat([x1, y1]).tolist()
        b = _cat([x1, y2]).tolist()
        c = _cat([x2, y2]).tolist()
        d = _cat([x2, y1]).tolist()
        polygons = [Polygon(points) for points in zip(a, b, c, d, a)]
        return polygons

    @classmethod
    def from_imgaug(Boxes, bboi):
        """
        Args:
            bboi (ia.BoundingBoxesOnImage):

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> orig = Boxes.random(5, format='tlbr')
            >>> bboi = orig.to_imgaug(shape=(500, 500))
            >>> self = Boxes.from_imgaug(bboi)
            >>> assert np.all(self.data == orig.data)
        """
        tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        tlbr = tlbr.reshape(-1, 4)
        return Boxes(tlbr, format=BoxFormat.TLBR, check=False)

    def to_coco(self, style='orig'):
        """
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

        Example:
            >>> import kwimage
            >>> self = kwimage.Boxes.random(5)
            >>> polys = self.to_polygons()
            >>> print('polys = {!r}'.format(polys))
        """
        import kwimage
        poly_list = []
        for tlbr in self.to_tlbr().data:
            x1, y1, x2, y2 = tlbr
            # Exteriors are counterlockwise
            exterior = np.array([
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1],
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

        Notes:
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
            >>> Boxes([25, 30, 35, 40], 'tlbr').tl_x
            array([25])
        """
        return self.to_tlbr(copy=False)._component(0)

    @property
    def tl_y(self):
        """
        Top left y coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'tlbr').tl_y
            array([30])
        """
        return self.to_tlbr(copy=False)._component(1)

    @property
    def br_x(self):
        """
        Bottom right x coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'tlbr').br_x
            array([35])
        """
        return self.to_tlbr(copy=False)._component(2)

    @property
    def br_y(self):
        """
        Bottom right y coordinate

        Example:
            >>> Boxes([25, 30, 35, 40], 'tlbr').br_y
            array([40])
        """
        return self.to_tlbr(copy=False)._component(3)

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
            augmenter (imgaug.augmenters.Augmenter): an imgaug augmenter
            input_dims (Tuple): h/w of the input image
            inplace (bool, default=False): if True, modifies data inplace

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
        tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        tlbr = tlbr.reshape(-1, 4)
        new.data = tlbr
        new.format = BoxFormat.TLBR
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
            transform (skimage.transform._geometric.GeometricTransform | ArrayLike):
                scikit-image tranform or a 3x3 transformation matrix

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused in non-raster spatial structures

            inplace (bool, default=False): if True, modifies data inplace

        TODO:
            - [ ] Generalize so the transform can be an arbitrary matrix

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> transform = skimage.transform.AffineTransform(scale=(2, 3), translation=(4, 5))
            >>> Boxes([25, 30, 15, 10], 'xywh').warp(transform)
            <Boxes(xywh, array([54., 95., 30., 30.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').warp(transform.params)
            <Boxes(xywh, array([54., 95., 30., 30.]))>
        """

        if inplace:
            new = self
            new_data = self.data
        else:
            if torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(np.float, copy=True)
            new = Boxes(new_data, self.format)

        try:
            # First try to warp using simple calls to axis-aligned operations
            rotation = 0
            shear = 0
            scale = 0
            translation = 0
            matrix = None

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
            elif isinstance(transform, (np.ndarray, torch.Tensor)):
                matrix = transform
            else:
                try:
                    import imgaug
                except ImportError:
                    import warnings
                    warnings.warn('imgaug is not installed')
                    raise TypeError(type(transform))
                if isinstance(transform, imgaug.augmenters.Augmenter):
                    aug = new._warp_imgaug(transform, input_dims=input_dims, inplace=True)
                    return aug
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
            raise NotImplementedError('Corner warping is not implemented yet')

        return new

    def scale(self, factor, output_dims=None, inplace=False):
        """
        Scale a bounding boxes by a factor.

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        TODO:
            it might be useful to have an argument `origin`, so everything
            is scaled about that origin.

            works natively with tlbr, cxywh, xywh, xy, or wh formats

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes(np.array([1, 1, 10, 10]), 'xywh').scale(2).data
            array([ 2.,  2., 20., 20.])
            >>> Boxes(np.array([[1, 1, 10, 10]]), 'xywh').scale((2, .5)).data
            array([[ 2. ,  0.5, 20. ,  5. ]])

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> x = Boxes([25., 30., 15., 10.], 'tlbr')
            >>> x.scale(2)
            >>> print(x)
            <Boxes(tlbr, array([25., 30., 15., 10.]))>
            >>> x.scale(2.0, inplace=True)
            >>> print(x)
            <Boxes(tlbr, array([50., 60., 30., 20.]))>

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
            >>> y1 = boxes.toformat('tlbr').scale(scale_xy).toformat('xywh')
            >>> y2 = boxes.toformat('xxyy').scale(scale_xy).toformat('xywh')
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
            if torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(np.float, copy=True)
            new = Boxes(new_data, self.format)
        if _numel(new_data) > 0:
            if self.format in [BoxFormat.XYWH, BoxFormat.CXYWH, BoxFormat.TLBR]:
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
        return new

    def translate(self, amount, output_dims=None, inplace=False):
        """
        Shift the boxes up/down left/right

        Args:
            factor (float or Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes([25, 30, 15, 10], 'xywh').translate(10)
            <Boxes(xywh, array([35., 40., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').translate((10, 0))
            <Boxes(xywh, array([35., 30., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'tlbr').translate((10, 5))

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> x = Boxes([25, 30, 15, 10], 'tlbr')
            >>> x.translate((10, 5))
            >>> print(x)
            <Boxes(tlbr, array([25, 30, 15, 10]))>
            >>> x.translate((10, 5), inplace=True)
            >>> print(x)
            <Boxes(tlbr, array([35, 35, 25, 15]))>

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> import kwimage
            >>> rng = kwarray.ensure_rng(0)
            >>> boxes = kwimage.Boxes.random(num=3, scale=10, rng=rng)
            >>> dxdy = (10 * rng.randn(len(boxes), 2)).astype(np.int)
            >>> boxes.translate(dxdy)
            <Boxes(xywh,
                array([[12.,  6.,  1.,  2.],
                       [ 8.,  9.,  0.,  2.],
                       [21.,  1.,  2.,  2.]]))>
            >>> y0 = boxes.toformat('xywh').translate(dxdy).toformat('xywh')
            >>> y1 = boxes.toformat('tlbr').translate(dxdy).toformat('xywh')
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

        kwarray.ArrayAPI.impl(self.data)

        if inplace:
            new = self
            new_data = self.data
        else:
            if torch.is_tensor(self.data):
                new_data = self.data.float().clone()
            else:
                new_data = self.data.astype(np.float, copy=True)
            new = Boxes(new_data, self.format)

        if _numel(new_data) > 0:
            if self.format in [BoxFormat.XYWH, BoxFormat.CXYWH]:
                new_data[..., 0] += tx
                new_data[..., 1] += ty
            elif self.format in [BoxFormat.TLBR]:
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
        Clip boxes to image boundaries.  If box is in tlbr format, inplace
        operation is an option.

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> self = boxes = Boxes(np.array([[-10, -10, 120, 120], [1, -2, 30, 50]]), 'tlbr')
            >>> clipped = boxes.clip(0, 0, 110, 100, inplace=False)
            >>> assert np.any(boxes.data != clipped.data)
            >>> clipped2 = boxes.clip(0, 0, 110, 100, inplace=True)
            >>> assert clipped2.data is boxes.data
            >>> assert np.all(clipped2.data == clipped.data)
            >>> print(clipped)
            <Boxes(tlbr,
                array([[  0,   0, 110, 100],
                       [  1,   0,  30,  50]]))>
        """
        if inplace:
            if self.format != BoxFormat.TLBR:
                raise ValueError('Must be in tlbr format to operate inplace')
            self2 = self
        else:
            self2 = self.to_tlbr(copy=True)
        if len(self2) == 0:
            return self2

        if True:
            impl = self._impl
            x1, y1, x2, y2 = impl.T(self2.data)
            np.clip(x1, x_min, x_max, out=x1)
            np.clip(y1, y_min, y_max, out=y1)
            np.clip(x2, x_min, x_max, out=x2)
            np.clip(y2, y_min, y_max, out=y2)
        else:
            if torch.is_tensor(self2.data):
                x1, y1, x2, y2 = self2.data.t()
                x1.clamp_(x_min, x_max)
                y1.clamp_(y_min, y_max)
                x2.clamp_(x_min, x_max)
                y2.clamp_(y_min, y_max)
            else:
                x1, y1, x2, y2 = self2.data.T
                np.clip(x1, x_min, x_max, out=x1)
                np.clip(y1, y_min, y_max, out=y1)
                np.clip(x2, x_min, x_max, out=x2)
                np.clip(y2, y_min, y_max, out=y2)
        return self2

    def transpose(self):
        """
        Reflects box coordinates about the line y=x.

        Example:
            >>> Boxes([[0, 1, 2, 4]], 'tlbr').transpose()
            <Boxes(tlbr, array([[1, 0, 4, 2]]))>
        """
        x, y, w, h = self.to_xywh().components
        self2 = self.__class__(_cat([y, x, h, w]), format=BoxFormat.XYWH)
        self2 = self2.toformat(self.format)
        return self2


class _BoxDrawMixins(object):
    """
    Non-core functions for box visualization
    """

    def draw(self, color='blue', alpha=None, labels=None, centers=False,
             fill=False, lw=2, ax=None):
        """
        Draws boxes using matplotlib. Wraps around kwplot.draw_boxes

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> self = Boxes.random(num=10, scale=512.0, rng=0, format='tlbr')
            >>> self.translate((-128, -128), inplace=True)
            >>> self.data[0][:] = [3, 3, 253, 253]
            >>> image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(color='blue')
            >>> # xdoc: +REQUIRES(--show)
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()
        """
        import kwplot
        boxes = self.to_xywh()
        if len(boxes.shape) == 1 and boxes.shape[0] == 4:
            # Hack to draw non-2d boxes
            boxes = boxes[None, :]

        return kwplot.draw_boxes(boxes, color=color, labels=labels,
                                 alpha=alpha, centers=centers, fill=fill,
                                 lw=lw, ax=ax)

    def draw_on(self, image, color='blue', alpha=None, labels=None,
                copy=False):
        """
        Draws boxes directly on the image using OpenCV

        Args:
            image (ndarray): must be in uint8 format
            color (str | ColorLike): one color for all boxes
            copy (bool, default=False): if False only copies if necessary

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> self = Boxes.random(num=10, scale=256, rng=0, format='tlbr')
            >>> self.data[0][:] = [3, 3, 253, 253]
            >>> color = 'blue'
            >>> image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image.copy(), color=color)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image, fnum=2000, pnum=(1, 2, 1))
            >>> kwplot.imshow(image2, fnum=2000, pnum=(1, 2, 2))
            >>> kwplot.show_if_requested()

        Example:
            >>> from kwimage.structs.boxes import *  # NOQA
            >>> import kwimage
            >>> self = Boxes.random(num=10, rng=0).scale(128)
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

        Ignore:
            import cv2
            image = kwimage.ensure_uint255(np.random.rand(128, 128, 3))
            image = kwimage.ensure_alpha_channel(np.random.rand(128, 128, 3))
            canvas = cv2.rectangle(image, (10, 10), (100, 100), color=(0, 1.0, 0, 1.0), thickness=10)
            kwplot.imshow(image, fnum=2000, pnum=(1, 2, 1))
            kwplot.imshow(canvas, fnum=2000, pnum=(1, 2, 2))
        """
        import cv2
        import kwimage
        def _coords(x, y):
            # ensure coords don't go out of bounds or cv2 throws weird error
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            return tuple(map(int, map(round, (x, y))))

        dtype_fixer = _generic._consistent_dtype_fixer(image)
        h, w = image.shape[0:2]

        # Get the color that is compatible with the input image encoding
        rect_color = kwimage.Color(color)._forimage(image)

        # Parameters for drawing the box rectangles
        rectkw = {
            'thickness': int(2),
            'color': rect_color,
        }

        # Parameters for drawing the label text
        fontkw = {
            'color': rect_color,
            'thickness': int(2),
            'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
            'fontScale': 0.75,
            'lineType': cv2.LINE_AA,
        }

        tlbr_list = self.to_tlbr().data

        if alpha is None:
            alpha = [1.0] * len(tlbr_list)
        elif isinstance(alpha, (float, np.float32, np.float64)):
            alpha = [alpha] * len(tlbr_list)

        if labels is None or labels is False:
            labels = [None] * len(tlbr_list)

        image = kwimage.atleast_3channels(image, copy=copy)

        for tlbr, label, alpha_ in zip(tlbr_list, labels, alpha):
            x1, y1, x2, y2 = tlbr
            pt1 = _coords(x1, y1)
            pt2 = _coords(x2, y2)
            x, y = pt1
            org = (x, y - (rectkw['thickness'] * 2))
            # Note cv2.rectangle does work inplace
            if alpha_ < 1.0:
                background = image.copy()

            # while cv2.rectangle will accept an alpha color it will not do any
            # blending with the background image.
            image = cv2.rectangle(image, pt1, pt2, **rectkw)
            if label:
                image = kwimage.draw_text_on_image(
                    image, text=label, org=org, **fontkw)
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


# Note: should we inherit from nh.util.Boxes (which is basically the same
# object) so isinstance works outside of the internal lib?


class Boxes(_BoxConversionMixins, _BoxPropertyMixins, _BoxTransformMixins,
            _BoxDrawMixins, ub.NiceRepr):  # _generic.Spatial
    """
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    This is a convinience class, and should not not store the data for very
    long. The general idiom should be create class, convert data, and then get
    the raw data and let the class be garbage collected. This will help ensure
    that your code is portable and understandable if this class is not
    available.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> Boxes([25, 30, 15, 10], 'xywh')
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_xywh()
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_tlbr()
        <Boxes(tlbr, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').scale(2).to_tlbr()
        <Boxes(tlbr, array([50., 60., 80., 80.]))>
        >>> Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'xywh').scale(.1).to_tlbr()
        <Boxes(tlbr, tensor([[ 2.5000,  3.0000,  4.0000,  5.0000]]))>

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
            >>> box2 = Boxes(box0.data, 'tlbr')
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
        return len(self.data)

    def __nice__(self):
        # return self.format + ', shape=' + str(list(self.data.shape))
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        nice = '{}, {}'.format(self.format, data_repr)
        return nice

    def __repr__(self):
        return super(Boxes, self).__str__()

    @classmethod
    def random(Boxes, num=1, scale=1.0, format=BoxFormat.XYWH, anchors=None,
               anchor_std=1.0 / 6, tensor=False, rng=None):
        """
        Makes random boxes; typically for testing purposes

        Args:
            num (int): number of boxes to generate
            scale (float | Tuple[float, float]): size of imgdims
            format (str): format of boxes to be created (e.g. tlbr, xywh)
            anchors (ndarray): normalized width / heights of anchor boxes to
                perterb and randomly place. (must be in range 0-1)
            anchor_std (float): magnitude of noise applied to anchor shapes
            tensor (bool): if True, returns boxes in tensor format
            rng (None | int | RandomState): initial random seed

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes.random(3, rng=0, scale=100)
            <Boxes(xywh,
                array([[54, 54,  6, 17],
                       [42, 64,  1, 25],
                       [79, 38, 17, 14]]))>
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
            tlbr = rng.rand(num, 4).astype(np.float32)

            tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
            tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
            br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
            br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

            tlbr[:, 0] = tl_x
            tlbr[:, 1] = tl_y
            tlbr[:, 2] = br_x
            tlbr[:, 3] = br_y
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
            tlbr = Boxes(cxywh, BoxFormat.CXYWH, check=False).to_tlbr().data

        boxes = Boxes(tlbr, format=BoxFormat.TLBR, check=False)
        boxes = boxes.scale(scale, inplace=True)
        if as_integer:
            boxes.data = boxes.data.astype(np.int)
        boxes = boxes.toformat(format, copy=False)
        if tensor:
            boxes = boxes.tensor()
        return boxes

    def copy(self):
        new_data = _copy(self.data)
        return Boxes(new_data, self.format, check=False)

    @classmethod
    def concatenate(cls, boxes, axis=0):
        """
        Concatenates multiple boxes together

        Args:
            boxes (Sequence[Boxes]): list of boxes to concatenate
            axis (int, default=0): axis to stack on

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
            flags (ArrayLike[bool]): true for items to be kept
            axis (int): you usually want this to be 0
            inplace (bool): if True, modifies this object

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> self.compress([True])
            <Boxes(tlbr, array([[25, 30, 15, 10]]))>
            >>> self.compress([False])
            <Boxes(tlbr, array([], shape=(0, 4), dtype=int64))>
        """
        if len(self.data.shape) != 2 and _numel(self.data) > 0:
            raise ValueError('data must be 2d got {}d'.format(
                len(self.data.shape)))
        newdata = _compress(self.data, flags, axis=axis)
        if inplace:
            self.data = newdata
            self2 = self
        else:
            self2 = self.__class__(newdata, self.format)
        return self2

    def take(self, idxs, axis=0, inplace=False):
        """
        Takes a subset of items at specific indices

        Args:
            indices (ArrayLike[int]): indexes of items to take
            axis (int): you usually want this to be 0
            inplace (bool): if True, modifies this object

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> self.take([0])
            <Boxes(tlbr, array([[25, 30, 15, 10]]))>
            >>> self.take([])
            <Boxes(tlbr, array([], shape=(0, 4), dtype=int64))>
        """
        if len(self.data.shape) != 2 and _numel(self.data) > 0:
            raise ValueError('data must be 2d got {}d'.format(
                len(self.data.shape)))
        if inplace:
            newdata = _take(self.data, idxs, axis=axis)
            self.data = newdata
            self2 = self
        else:
            newdata = _take(self.data, idxs, axis=axis)
            self2 = self.__class__(newdata, self.format)
        return self2

    def is_tensor(self):
        """ is the backend fueled by torch? """
        return torch.is_tensor(self.data)

    def is_numpy(self):
        """ is the backend fueled by numpy? """
        return isinstance(self.data, np.ndarray)

    @ub.memoize_property
    def _impl(self):
        """
        returns the kwarray.ArrayAPI implementation for the data

        Example:
            >>> assert Boxes.random().numpy()._impl.is_numpy
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

        Notes:
            this operation is not inplace

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
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
        if torch.is_tensor(data):
            dtype = _rectify_torch_dtype(dtype)
            newself = self.__class__(data.to(dtype), self.format)
        else:
            newself = self.__class__(data.astype(dtype), self.format)
        return newself

    def numpy(self):
        """
        Converts tensors to numpy. Does not change memory if possible.

        Example:
            >>> self = Boxes.random(3).tensor()
            >>> newself = self.numpy()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        data = self.data
        if torch.is_tensor(data):
            data = data.data.cpu().numpy()
        newself = self.__class__(data, self.format)
        return newself

    def tensor(self, device=ub.NoParam):
        """
        Converts numpy to tensors. Does not change memory if possible.

        Example:
            >>> self = Boxes.random(3)
            >>> newself = self.tensor()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        data = self.data
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if device is not ub.NoParam:
            data = data.to(device)
        newself = self.__class__(data, self.format)
        return newself

    def ious(self, other, bias=0, impl='auto', mode=None):
        """
        Compute IOUs (intersection area over union area) between these boxes
        and another set of boxes.

        Args:
            other (Boxes): boxes to compare IoUs against
            bias (int, default=0): either 0 or 1, does TL=BR have area of 0 or 1?
            impl (str, default='auto'): code to specify implementation used to
                ious. Can be either torch, py, c, or auto. Efficiency and the
                exact result will vary by implementation, but they will always
                be close.  Some implementations only accept certain data types
                (e.g.  impl='c', only accepts float32 numpy arrays).  See
                ~/code/kwimage/dev/bench_bbox.py for benchmark details. On my
                system the torch impl was fastest (when the data was on the
                GPU).
            mode : depricated, use impl

        Examples:
            >>> self = Boxes(np.array([[ 0,  0, 10, 10],
            >>>                        [10,  0, 20, 10],
            >>>                        [20,  0, 30, 10]]), 'tlbr')
            >>> other = Boxes(np.array([6, 2, 20, 10]), 'tlbr')
            >>> overlaps = self.ious(other, bias=1).round(2)
            >>> assert np.all(np.isclose(overlaps, [0.21, 0.63, 0.04])), repr(overlaps)

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
        """
        other_is_1d = len(other) > 0 and (len(other.shape) == 1)
        if other_is_1d:
            # `box_ious` expects 2d input
            other = other[None, :]

        # self_is_1d = (len(self.shape) == 1)
        # if self_is_1d:
        #     self = self[None, :]

        if len(other) == 0 or len(self) == 0:
            if torch.is_tensor(self.data) or torch.is_tensor(other.data):
                if _TORCH_HAS_EMPTY_SHAPE:
                    torch.empty((len(self), len(other)))
                else:
                    ious = torch.empty(0)
            else:
                ious = np.empty((len(self), len(other)))
        else:
            self_tlbr = self.to_tlbr(copy=False)
            other_tlbr = other.to_tlbr(copy=False)

            if mode is not None:
                warnings.warn('mode is depricated use impl', DeprecationWarning)
                impl = mode

            ious = box_ious(self_tlbr.data, other_tlbr.data, bias=bias,
                            impl=impl)

        if other_is_1d:
            ious = ious[..., 0]

        # if self_is_1d:
        #     ious = ious[0, ...]
        return ious

    def isect_area(self, other, bias=0):
        """
        Intersection part of intersection over union computation

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> self = Boxes.random(5, scale=10.0, rng=0, format='tlbr')
            >>> other = Boxes.random(3, scale=10.0, rng=1, format='tlbr')
            >>> isect = self.isect_area(other, bias=0)
            >>> ious_v1 = isect / ((self.area + other.area.T) - isect)
            >>> ious_v2 = self.ious(other, bias=0)
            >>> assert np.allclose(ious_v1, ious_v2)
        """
        other_is_1d = (len(other.shape) == 1)
        if other_is_1d:
            other = other[None, :]
        self_tlbr = self.to_tlbr(copy=False)
        other_tlbr = other.to_tlbr(copy=False)

        isect = _isect_areas(self_tlbr.data, other_tlbr.data)
        if other_is_1d:
            isect = isect[..., 0]
        return isect

    def intersection(self, other):
        """
        Pairwise intersection between two sets of Boxes

        Returns:
            Boxes: intersected boxes

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

        self_tlbr = self.to_tlbr(copy=False).data
        other_tlbr = other.to_tlbr(copy=False).data

        tl = np.maximum(self_tlbr[..., :2], other_tlbr[..., :2])
        br = np.minimum(self_tlbr[..., 2:], other_tlbr[..., 2:])

        is_bad = np.any(tl > br, axis=1)
        tlbr = np.concatenate([tl, br], axis=-1)

        tlbr[is_bad] = np.nan

        isect = Boxes(tlbr, 'tlbr')

        return isect

    def view(self, *shape):
        """
        Passthrough method to view or reshape

        Example:
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='xywh').tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='tlbr').tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
        """
        data_ = _view(self.data, *shape)
        return self.__class__(data_, self.format)


def _copy(data):
    if torch.is_tensor(data):
        return data.clone()
    else:
        return data.copy()


def _view(data, *shape):
    if torch.is_tensor(data):
        data_ = data.view(*shape)
    else:
        data_ = data.reshape(*shape)
    return data_


def _cat(datas, axis=-1):
    if torch.is_tensor(datas[0]):
        return torch.cat(datas, dim=axis)
    else:
        return np.concatenate(datas, axis=axis)


def _take(data, indices, axis=None):
    """
    compatable take-API between torch and numpy

    Example:
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
    elif torch.is_tensor(data):
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
    elif torch.is_tensor(data):
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
