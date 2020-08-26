"""
Generic segmentation object that can use either a Mask or (Multi)Polygon
backend.
"""
# from kwimage.structs import _generic
import numpy as np
import six
from . import _generic
import ubelt as ub


class _WrapperObject(ub.NiceRepr):

    def __nice__(self):
        return self.data.__nice__()

    def draw(self, *args, **kw):
        return self.data.draw(*args, **kw)

    def draw_on(self, *args, **kw):
        return self.data.draw_on(*args, **kw)

    def warp(self, *args, **kw):
        return self.data.warp(*args, **kw)

    def translate(self, *args, **kw):
        return self.data.translate(*args, **kw)

    def scale(self, *args, **kw):
        return self.data.scale(*args, **kw)

    def to_coco(self, *args, **kw):
        return self.data.to_coco(*args, **kw)

    def numpy(self, *args, **kw):
        return self.data.numpy(*args, **kw)

    def tensor(self, *args, **kw):
        return self.data.tensor(*args, **kw)


class Segmentation(_WrapperObject):
    """
    Either holds a MultiPolygon, Polygon, or Mask

    Args:
        data (object): the underlying object
        format (str): either 'mask', 'polygon', or 'multipolygon'
    """
    def __init__(self, data, format=None):
        self.data = data
        self.format = format

    @classmethod
    def random(cls, rng=None):
        """
        Example:
            >>> self = Segmentation.random()
            >>> print('self = {!r}'.format(self))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> self.draw()
            >>> kwplot.show_if_requested()
        """

        import kwarray
        import kwimage
        rng = kwarray.ensure_rng(rng)
        if rng.rand() > 0.5:
            data = kwimage.Polygon.random()
        else:
            data = kwimage.Mask.random()
        return cls.coerce(data)

    def to_multi_polygon(self):
        return self.data.to_multi_polygon()

    def to_mask(self, dims=None):
        return self.data.to_mask(dims=dims)

    @property
    def meta(self):
        return self.data.meta

    @classmethod
    def coerce(cls, data, dims=None):
        import kwimage
        if _generic._isinstance2(data, kwimage.Segmentation):
            self = data
        elif _generic._isinstance2(data, kwimage.Mask):
            self = Segmentation(data, 'mask')
        elif _generic._isinstance2(data, kwimage.Polygon):
            self = Segmentation(data, 'polygon')
        elif _generic._isinstance2(data, kwimage.MultiPolygon):
            self = Segmentation(data, 'multipolygon')
        else:
            data = _coerce_coco_segmentation(data, dims=dims)
            self = cls.coerce(data, dims=dims)
        return self


class SegmentationList(_generic.ObjectList):
    """
    Store and manipulate multiple segmentations (masks or polygons), usually
    within the same image
    """

    def to_polygon_list(self):
        """
        Converts all mask objects to multi-polygon objects
        """
        import kwimage
        new = kwimage.PolygonList([
            None if item is None else item.to_multi_polygon()
            for item in self
        ])
        return new

    def to_mask_list(self, dims=None):
        """
        Converts all mask objects to multi-polygon objects
        """
        import kwimage
        new = kwimage.MaskList([
            None if item is None else item.to_mask(dims=dims)
            for item in self
        ])
        return new

    def to_segmentation_list(self):
        return self

    @classmethod
    def coerce(cls, data):
        """
        Interpret data as a list of Segmentations
        """
        if isinstance(data, (list, _generic.ObjectList)):
            data = [None if item is None else Segmentation.coerce(item)
                    for item in data]
        else:
            raise TypeError(data)
        self = cls(data)
        return self


def _coerce_coco_segmentation(data, dims=None):
    """
    Attempts to auto-inspect the format of segmentation data

    Args:
        data : the data to coerce

             2D-C-ndarray -> C_MASK
             2D-F-ndarray -> F_MASK

             Dict(counts=bytes) -> BYTES_RLE
             Dict(counts=ndarray) -> ARRAY_RLE

             Dict(exterior=ndarray) -> ARRAY_RLE

             # List[List[int]] -> Polygon
             List[int] -> Polygon
             List[Dict] -> MultPolygon

        dims (Tuple): required for certain formats like polygons
            height / width of the source image

    Returns:
        Mask | Polygon | MultiPolygon - depending on which is appropriate

    Example:
        >>> segmentation = {'size': [5, 9], 'counts': ';?1B10O30O4'}
        >>> dims = (9, 5)
        >>> raw_mask = (np.random.rand(32, 32) > .5).astype(np.uint8)
        >>> _coerce_coco_segmentation(segmentation)
        >>> _coerce_coco_segmentation(raw_mask)

        >>> coco_polygon = [
        >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
        >>>     np.array([[2, 1],[2, 2],[4, 2],[4, 1]]),
        >>> ]
        >>> self = _coerce_coco_segmentation(coco_polygon, dims)
        >>> print('self = {!r}'.format(self))
        >>> coco_polygon = [
        >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
        >>> ]
        >>> self = _coerce_coco_segmentation(coco_polygon, dims)
        >>> print('self = {!r}'.format(self))
    """
    import kwimage
    from kwimage.structs.mask import MaskFormat
    if isinstance(data, np.ndarray):
        # INPUT TYPE: RAW MASK
        if dims is not None:
            assert dims == data.shape[0:2]
        if data.flags['F_CONTIGUOUS']:
            self = kwimage.Mask(data, MaskFormat.F_MASK)
        else:
            self = kwimage.Mask(data, MaskFormat.C_MASK)
    elif isinstance(data, dict):
        if 'counts' in data:
            # INPUT TYPE: COCO RLE DICTIONARY
            if dims is not None:
                data_shape = data.get('dims', data.get('shape', data.get('size', None)))
                if data_shape is None:
                    data['shape'] = data_shape
                else:
                    assert tuple(map(int, dims)) == tuple(map(int, data_shape)), (
                        '{} {}'.format(dims, data_shape))
            if isinstance(data['counts'], (six.text_type, six.binary_type)):
                self = kwimage.Mask(data, MaskFormat.BYTES_RLE)
            else:
                self = kwimage.Mask(data, MaskFormat.ARRAY_RLE)
        elif 'exterior' in data:
            # TODO: kwimage.Polygon.from_coco
            self = kwimage.Polygon(**data)
            # raise NotImplementedError('explicit polygon coerce')
        else:
            raise TypeError(type(data))
    elif isinstance(data, list):
        # THIS IS NOT AN IDEAL FORMAT. IDEALLY WE WILL MODIFY COCO TO USE
        # DICTIONARIES FOR POLYGONS, WHICH ARE UNAMBIGUOUS
        if len(data) == 0:
            self = None
        else:
            first = ub.peek(data)
            if isinstance(first, dict):
                # TODO: kwimage.MultiPolygon.from_coco
                self = kwimage.MultiPolygon(
                    [kwimage.Polygon(**item) for item in data])
            elif isinstance(first, int):
                # TODO: kwimage.Polygon.from_coco
                exterior = np.array(data).reshape(-1, 2)
                self = kwimage.Polygon(exterior=exterior)
            elif isinstance(first, list):
                # TODO: kwimage.MultiPolygon.from_coco
                poly_list = [kwimage.Polygon(exterior=np.array(item).reshape(-1, 2))
                             for item in data]
                if len(poly_list) == 1:
                    self = poly_list[0]
                else:
                    self = kwimage.MultiPolygon(poly_list)
            elif isinstance(first, np.ndarray):
                poly_list = [kwimage.Polygon(exterior=item.reshape(-1, 2))
                             for item in data]
                if len(poly_list) == 1:
                    self = poly_list[0]
                else:
                    self = kwimage.MultiPolygon(poly_list)
            else:
                raise TypeError(type(data))
    else:
        raise TypeError(type(data))
    return self
