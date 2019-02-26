"""
Data structure for Binary Masks

Structure for efficient encoding of per-annotation segmentation masks
Based on efficient cython/C code in the cocoapi [1].

THIS IS CURRENTLY A WORK IN PROGRESS.

References:
    ..[1] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/_mask.pyx
    ..[2] https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.c
    ..[3] https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.h
    ..[4] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py

Goals:
    The goal of this file is to create a datastructure that lets the developer
    seemlessly convert between:
        (1) raw binary uint8 masks
        (2) memory-efficient comprsssed run-length-encodings of binary
        segmentation masks.
        (3) convex polygons
        (4) convex hull polygons
        (5) bounding box

    It is not there yet, and the API is subject to change in order to better
    accomplish these goals.
"""
import cv2
import copy
import numpy as np
import ubelt as ub
import itertools as it
from kwimage.structs._mask_backend import cython_mask

__all__ = ['Mask']


class MaskFormat:
    """
    Defines valid formats and their aliases.

    Attrs:
        aliases (Mapping[str, str]):
            maps format aliases to their cannonical name.
    """
    cannonical = []

    def _register(k, cannonical=cannonical):
        cannonical.append(k)
        return k

    STRING_RLE   = _register('string_rle')    # cython compressed RLE
    ARRAY_RLE    = _register('array_rle')     # numpy uncompreesed RLE
    C_MASK       = _register('c_mask')        # row-major raw binary mask
    FORTRAN_MASK = _register('fortran_mask')  # column-major raw binary mask

    aliases = {
    }
    for key in cannonical:
        aliases[key] = key


class _MaskConversionMixin(object):
    """
    Mixin class registering conversion functions
    """
    convert_funcs = {}

    def _register_convertor(key, convert_funcs=convert_funcs):
        def _reg(func):
            convert_funcs[key] = func
            return func
        return _reg

    def toformat(self, format, copy=False):
        """
        Changes the internal representation using one of the registered
        convertor functions.

        Args:
            format (str):
                the string code for the format you want to transform into.

        Example:
            >>> from kwimage.structs.mask import MaskFormat  # NOQA
            >>> mask = Mask.random(shape=(8, 8), rng=0)
            >>> # Test that we can convert to and from all formats
            >>> for format1 in MaskFormat.cannonical:
            ...     mask1 = mask.toformat(format1)
            ...     for format2 in MaskFormat.cannonical:
            ...         mask2 = mask1.toformat(format2)
            ...         img1 = mask1.to_c_mask().data
            ...         img2 = mask2.to_c_mask().data
            ...         if not np.all(img1 == img2):
            ...             msg = 'Failed convert {} <-> {}'.format(format1, format2)
            ...             print(msg)
            ...             raise AssertionError(msg)
            ...         else:
            ...             msg = 'Passed convert {} <-> {}'.format(format1, format2)
            ...             print(msg)
        """
        key = MaskFormat.aliases.get(format, format)
        try:
            func = self.convert_funcs[key]
            return func(self, copy)
        except KeyError:
            raise KeyError('Cannot convert {} to {}'.format(self.format, format))

    @_register_convertor(MaskFormat.STRING_RLE)
    def to_string_rle(self, copy=False):
        """
        Example:
            >>> from kwimage.structs.mask import MaskFormat  # NOQA
            >>> mask = Mask.random(shape=(8, 8), rng=0)
            >>> print(mask.to_string_rle().data['counts'])
            ...'135000k0NWO0L'
            >>> print(mask.to_array_rle().data['counts'].tolist())
            [1, 3, 5, 3, 5, 3, 32, 1, 7, 1, 3]
            >>> print(mask.to_array_rle().to_string_rle().data['counts'])
            ...'135000k0NWO0L'
        """
        if self.format == MaskFormat.STRING_RLE:
            return self.copy() if copy else self
        if self.format == MaskFormat.ARRAY_RLE:
            w, h = self.data['size']
            if self.data.get('order', 'F') != 'F':
                raise ValueError('Expected column-major array RLE')
            newdata = cython_mask.frUncompressedRLE([self.data], h, w)[0]
            self = Mask(newdata, MaskFormat.STRING_RLE)

        elif self.format == MaskFormat.FORTRAN_MASK:
            fortran_masks = self.data[:, :, None]
            encoded = cython_mask.encode(fortran_masks)[0]
            self = Mask(encoded, format=MaskFormat.STRING_RLE)
        elif self.format == MaskFormat.C_MASK:
            c_mask = self.data
            fortran_masks = np.asfortranarray(c_mask)[:, :, None]
            encoded = cython_mask.encode(fortran_masks)[0]
            self = Mask(encoded, format=MaskFormat.STRING_RLE)
        else:
            raise NotImplementedError(self.format)
        return self

    @_register_convertor(MaskFormat.ARRAY_RLE)
    def to_array_rle(self, copy=False):
        if self.format == MaskFormat.ARRAY_RLE:
            return self.copy() if copy else self
        else:
            # NOTE: inefficient, could be improved
            import kwimage
            fortran_mask = self.to_fortran_mask().data
            encoded = kwimage.encode_run_length(
                fortran_mask, binary=True, order='F')
            encoded['size'] = encoded['shape'][0:2][::-1]  # hack in size
            self = Mask(encoded, format=MaskFormat.ARRAY_RLE)
        return self

    @_register_convertor(MaskFormat.FORTRAN_MASK)
    def to_fortran_mask(self, copy=False):
        if self.format == MaskFormat.FORTRAN_MASK:
            return self.copy() if copy else self
        elif self.format == MaskFormat.C_MASK:
            c_mask = self.data.copy() if copy else self.data
            fortran_mask = np.asfortranarray(c_mask)
        # elif self.format == MaskFormat.ARRAY_RLE:
        #     pass
        else:
            # NOTE: inefficient, could be improved
            self = self.to_string_rle(copy=False)
            fortran_mask = cython_mask.decode([self.data])[:, :, 0]
        self = Mask(fortran_mask, MaskFormat.FORTRAN_MASK)
        return self

    @_register_convertor(MaskFormat.C_MASK)
    def to_c_mask(self, copy=False):
        if self.format == MaskFormat.C_MASK:
            return self.copy() if copy else self
        elif self.format == MaskFormat.FORTRAN_MASK:
            fortran_mask = self.data.copy() if copy else self.data
            c_mask = np.ascontiguousarray(fortran_mask)
        else:
            fortran_mask = self.to_fortran_mask(copy=False).data
            c_mask = np.ascontiguousarray(fortran_mask)
        self = Mask(c_mask, MaskFormat.C_MASK)
        return self


class _MaskConstructorMixin(object):
    """
    Alternative ways to construct a masks object
    """

    @classmethod
    def from_polygons(Mask, polygons, shape):
        """
        Args:
            polygons (ndarray | List[ndarray]): one or more polygons that
                will be joined together. The ndarray may either be an
                Nx2 or a flat c-contiguous array or xy points.
            shape (Tuple): height / width of the source image

        Example:
            >>> polygons = [
            >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
            >>>     np.array([[0, 9],[4, 8],[2, 3]]),
            >>> ]
            >>> shape = (9, 5)
            >>> self = Mask.from_polygons(polygons, shape)
            >>> print(self)
            <Mask({'counts': b'724;MG2MN16', 'size': [9, 5]}, format=string_rle)>
            >>> polygon = polygons[0]
            >>> print(Mask.from_polygons(polygon, shape))
            <Mask({'counts': b'b04500N2', 'size': [9, 5]}, format=string_rle)>
        """
        h, w = shape
        # TODO: holes? geojson?
        if isinstance(polygons, np.ndarray):
            polygons = [polygons]
        flat_polys = [ps.ravel() for ps in polygons]
        encoded = cython_mask.frPoly(flat_polys, h, w)
        ccs = [Mask(e, MaskFormat.STRING_RLE) for e in encoded]
        self = Mask.union(*ccs)
        return self

    @classmethod
    def from_mask(Mask, mask, offset=None, shape=None):
        """
        Creates an RLE encoded mask from a raw binary mask, but you may
        optionally specify an offset if the mask is part of a larger image.
        """


class _MaskDrawMixin(object):
    """
    Non-core functions for mask visualization
    """

    def draw_on(self, image, color='blue', alpha=0.5):
        """
        Draws the mask on an image

        Example:
            >>> from kwimage.structs.mask import *  # NOQA
            >>> import kwimage
            >>> image = kwimage.grab_test_image()
            >>> self = Mask.random(shape=image.shape[0:2])
            >>> toshow = self.draw_on(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(toshow)
            >>> kwplot.show_if_requested()
        """
        import kwplot
        import kwimage

        mask = self.to_c_mask().data
        rgb01 = list(kwplot.Color(color).as01())
        rgba01 = np.array(rgb01 + [1])[None, None, :]
        alpha_mask = rgba01 * mask[:, :, None]
        alpha_mask[..., 3] = mask * alpha

        toshow = kwimage.overlay_alpha_images(alpha_mask, image)
        return toshow

    def draw(self, color='blue', alpha=0.5, ax=None):
        """
        Draw on the current matplotlib axis
        """
        import kwplot
        if ax is None:
            from matplotlib import pyplot as plt
            ax = plt.gca()

        mask = self.to_c_mask().data
        rgb01 = list(kwplot.Color(color).as01())
        rgba01 = np.array(rgb01 + [1])[None, None, :]
        alpha_mask = rgba01 * mask[:, :, None]
        alpha_mask[..., 3] = mask * alpha
        ax.imshow(alpha_mask)


class Mask(ub.NiceRepr, _MaskConversionMixin, _MaskConstructorMixin,
           _MaskDrawMixin):
    """
    Manages a single segmentation mask and can convert to and from
    multiple formats including:

        * string_rle
        * array_rle
        * c_mask
        * fortran_mask

    Example:
        >>> # a ms-coco style compressed string rle segmentation
        >>> segmentation = {'size': [5, 9], 'counts': ';?1B10O30O4'}
        >>> mask = Mask(segmentation, 'string_rle')
        >>> # convert to binary numpy representation
        >>> binary_mask = mask.to_c_mask().data
        >>> print(ub.repr2(binary_mask.tolist(), nl=1, nobr=1))
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0],

    """
    def __init__(self, data=None, format=None):
        self.data = data
        self.format = format

    def __nice__(self):
        return '{}, format={}'.format(ub.repr2(self.data, nl=0), self.format)

    @classmethod
    def random(Mask, rng=None, shape=(32, 32)):
        import kwarray
        import kwimage
        rng = kwarray.ensure_rng(rng)
        # Use random heatmap to make some blobs for the mask
        probs = kwimage.Heatmap.random(
            dims=shape, rng=rng, classes=2).data['class_probs'][1]
        c_mask = (probs > .5).astype(np.uint8)
        self = Mask(c_mask, MaskFormat.C_MASK)
        return self

    def copy(self):
        """
        Performs a deep copy of the mask data

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> other = self.copy()
            >>> assert other.data is not self.data
        """
        return Mask(copy.deepcopy(self.data), self.format)

    def union(self, *others):
        """
        This can be used as a staticmethod or an instancemethod

        Example:
            >>> masks = [Mask.random(shape=(8, 8), rng=i) for i in range(2)]
            >>> mask = Mask.union(*masks)
            >>> print(mask.area)
            33
        """
        cls = self.__class__ if isinstance(self, Mask) else Mask
        rle_datas = [item.to_string_rle().data for item in it.chain([self], others)]
        return cls(cython_mask.merge(rle_datas, intersect=0), MaskFormat.STRING_RLE)

    def intersection(self, *others):
        """
        This can be used as a staticmethod or an instancemethod

        Example:
            >>> masks = [Mask.random(shape=(8, 8), rng=i) for i in range(2)]
            >>> mask = Mask.intersection(*masks)
            >>> print(mask.area)
            9
        """
        cls = self.__class__ if isinstance(self, Mask) else Mask
        rle_datas = [item.to_string_rle().data for item in it.chain([self], others)]
        return cls(cython_mask.merge(rle_datas, intersect=1), MaskFormat.STRING_RLE)

    @property
    def shape(self):
        if self.format in {MaskFormat.STRING_RLE, MaskFormat.ARRAY_RLE}:
            if 'shape' in self.data:
                return self.data['shape']
            else:
                return self.data['size'][::-1]
        if self.format in {MaskFormat.C_MASK, MaskFormat.FORTRAN_MASK}:
            return self.data.shape

    @property
    def area(self):
        """
        Returns the number of non-zero pixels

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> self.area
            11
        """
        self = self.to_string_rle()
        return cython_mask.area([self.data])[0]

    def get_xywh(self):
        """
        Gets the bounding xywh box coordinates of this mask

        Returns:
            ndarray: x, y, w, h: Note we dont use a Boxes object because
                a general singular version does not yet exist.

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> self.get_xywh().tolist()
            [0.0, 1.0, 8.0, 4.0]
        """
        # import kwimage
        self = self.to_string_rle()
        xywh = cython_mask.toBbox([self.data])[0]
        # boxes = kwimage.Boxes(xywh, 'xywh')
        # return boxes
        return xywh

    def get_polygon(self):
        """
        Returns a list of (x,y)-coordinate lists. The length of the list is
        equal to the number of disjoint regions in the mask.

        Returns:
            List[ndarray]: polygon around each connected component of the
                mask. Each ndarray is an Nx2 array of xy points.

        NOTE:
            The returned polygon may not surround points that are only one
            pixel thick.

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> polygons = self.get_polygon()
            >>> print('polygons = ' + ub.repr2(polygons))
            polygons = [
                np.array([[6, 4],[7, 4]], dtype=np.int32),
                np.array([[0, 1],[0, 3],[2, 3],[2, 1]], dtype=np.int32),
            ]
            >>> other = Mask.from_polygons(polygons, self.shape)
            >>> self = self.to_string_rle()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> image = np.ones(self.shape)
            >>> image = self.draw_on(image, color='blue')
            >>> image = other.draw_on(image, color='red')
            >>> kwplot.imshow(image)
        """
        mask = self.to_c_mask().data
        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1,
                                         cv2.BORDER_CONSTANT, value=0)
        contours_, hierarchy_ = cv2.findContours(
            padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
            offset=(-1, -1))
        polygon = [c[:, 0, :] for c in contours_]

        # TODO: a kwimage structure for polygons

        return polygon

    def get_convex_hull(self):
        """
        Returns a list of xy points around the convex hull of this mask

        NOTE:
            The returned polygon may not surround points that are only one
            pixel thick.

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> polygons = self.get_convex_hull()
            >>> print('polygons = ' + ub.repr2(polygons))
            >>> other = Mask.from_polygons(polygons, self.shape)
        """
        mask = self.to_c_mask().data
        cc_y, cc_x = np.where(mask)
        points = np.vstack([cc_x, cc_y]).T
        hull = cv2.convexHull(points)[:, 0, :]
        return hull

    def iou(self, other):
        """
        The area of intersection over the area of union

        TODO:
            - [ ] Write plural Masks version of this class, which should
                  be able to perform this operation more efficiently.

        Example:
            >>> self = Mask.random(rng=0)
            >>> other = Mask.random(rng=1)
            >>> iou = self.iou(other)
            >>> print('iou = {:.4f}'.format(iou))
            iou = 0.0542
        """
        item1 = self.to_string_rle(copy=False).data
        item2 = other.to_string_rle(copy=False).data
        # I'm not sure what passing `pyiscrowd` actually does here
        # TODO: determine what `pyiscrowd` does, and document it.
        pyiscrowd = np.array([0], dtype=np.uint8)
        iou = cython_mask.iou([item1], [item2], pyiscrowd)[0, 0]
        return iou

if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.structs.mask all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
