"""
Generic segmentation object that can use either a Mask or (Multi)Polygon
backend.
"""
# from kwimage.structs import _generic
import numpy as np
import numbers
from . import _generic
import ubelt as ub


class _WrapperObject(ub.NiceRepr):

    def __nice__(self):
        return self.data.__nice__()

    def draw(self, *args, **kw):
        return self.data.draw(*args, **kw)

    def draw_on(self, *args, **kw):
        """
        See help(self.data.draw_on)
        """
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
            >>> # xdoctest: +REQUIRES(module:cv2)
            >>> self = Segmentation.random()
            >>> print('self = {!r}'.format(self))
            >>> # xdoctest: +REQUIRES(--show)
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

    def to_mask(self, dims=None, pixels_are='points'):
        return self.data.to_mask(dims=dims, pixels_are=pixels_are)

    def box(self):
        return self.data.box()

    @property
    def area(self):
        return self.data.area

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
            if data is None:
                return None
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

    def to_mask_list(self, dims=None, pixels_are='points'):
        """
        Converts all mask objects to multi-polygon objects
        """
        import kwimage
        new = kwimage.MaskList([
            None if item is None else item.to_mask(dims=dims, pixels_are=pixels_are)
            for item in self
        ])
        return new

    def to_segmentation_list(self):
        return self

    @classmethod
    def coerce(cls, data, none_policy='raise'):
        """
        Interpret data as a list of Segmentations

        Args:
            none_policy (str):
                Determines how to handle None inputs.
                Can be: 'return-None', or 'raise'.
        """
        if isinstance(data, (list, _generic.ObjectList)):
            data = [None if item is None else Segmentation.coerce(item)
                    for item in data]
        else:
            if data is None:
                return _handle_null_policy(none_policy)
            else:
                raise TypeError(data)
        self = cls(data)
        return self


def _handle_null_policy(policy, ex_type=TypeError,
                        ex_msg='cannot accept null input'):
    """
    For handling a nan or None policy.

    Args:
        policy (str):
            How null inputs are handled. Can be:
                'return-None': returns None
                'return-nan': returns nan
                'raise': raises an error

        ex_type (type): Exception type to raise if policy is raise

        ex_msg (msg): Exception arguments

    TODO: rectify with similar logic in kwutil/util_time
    """
    if policy == 'return-None':
        return None
    elif policy == 'return-nan':
        return float('nan')
    elif policy == 'raise':
        raise ex_type(ex_msg)
    else:
        raise KeyError(ub.paragraph(
            f'''
            Unknown null policy={policy!r}.
            Valid choices are "return-None", "return-nan", and "raise".
            '''))


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

    TODO:
        - [ ] Handle WKT

    Returns:
        Mask | Polygon | MultiPolygon | Segmentation - depending on which is appropriate

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
            if isinstance(data['counts'], (str, bytes)):
                self = kwimage.Mask(data, MaskFormat.BYTES_RLE)
            else:
                self = kwimage.Mask(data, MaskFormat.ARRAY_RLE)
        elif 'exterior' in data:
            # TODO: kwimage.Polygon.from_coco
            self = kwimage.Polygon(**data)
            # raise NotImplementedError('explicit polygon coerce')
        elif 'type' in data:
            if data['type'] == 'Polygon':
                self = kwimage.Polygon.from_geojson(data)
            elif data['type'] == 'MultiPolygon':
                self = kwimage.MultiPolygon.from_geojson(data)
            else:
                raise NotImplementedError(data['type'])
        else:
            raise TypeError('Unable to interpret dictionary format {}'.format(data))
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
            elif isinstance(first, numbers.Number):
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
                raise TypeError('Unable to interpret list format {}'.format(data))
    elif isinstance(data, (kwimage.Polygon, kwimage.MultiPolygon, kwimage.Mask, kwimage.Segmentation)):
        self = data
    else:
        from shapely.geometry.polygon import Polygon
        from shapely.geometry.multipolygon import MultiPolygon
        if isinstance(data, MultiPolygon):
            self = kwimage.MultiPolygon.from_shapely(data)
        elif isinstance(data, Polygon):
            self = kwimage.Polygon.from_shapely(data)
        else:
            raise TypeError(f'Unable to coerce type={type(data)!r} into a segmentation. data={data!r}')
    return self
