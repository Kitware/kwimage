"""
Data structure for Binary Masks

Structure for efficient encoding of per-annotation segmentation masks
Based on efficient cython/C code in the cocoapi [CocoStuffPyx]_ [CocoStuffC]_
[CocoStuffH]_ [CocoStuffPy]_.

References:
    .. [CocoStuffPyx] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/_mask.pyx
    .. [CocoStuffC] https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.c
    .. [CocoStuffH] https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.h
    .. [CocoStuffPy] https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py

Goals:
    The goal of this file is to create a datastructure that lets the developer
    seemlessly convert between:
    (1) raw binary uint8 masks
    (2) memory-efficient compressed run-length-encodings of binary segmentation masks.
    (3) convex polygons
    (4) convex hull polygons
    (5) bounding box

    It is not there yet, and the API is subject to change in order to better
    accomplish these goals.

Note:
    IN THIS FILE ONLY: size corresponds to a h/w tuple to be compatible with
    the coco semantics. Everywhere else in this repo, size uses opencv
    semantics which are w/h.
"""
import sys
import copy
import numpy as np
import ubelt as ub
import itertools as it
import warnings
import numbers
from . import _generic

# try:
#     import torch
# except Exception:
#     torch = None

# try:
#     from line_profiler import profile  # NOQA
# except Exception:
#     from ubelt import identity as profile  # NOQA


class _Mask_Backends():
    # TODO: could make this prettier
    def __init__(self):
        self._funcs = None

    def _lazy_init(self):
        from kwimage import _internal
        _funcs = {}
        try:
            from pycocotools import _mask
            _funcs['pycoco'] = _mask
        except Exception as ex:
            if not _internal.KWIMAGE_DISABLE_IMPORT_WARNINGS:
                warnings.warn(
                    'optional module pycocotools is not available: {}'.format(
                        str(ex)))

        if not _internal.KWIMAGE_DISABLE_C_EXTENSIONS:
            try:
                from kwimage_ext.structs._mask_backend import cython_mask
                _funcs['kwimage'] = cython_mask
            except Exception as ex:
                if not _internal.KWIMAGE_DISABLE_IMPORT_WARNINGS:
                    warnings.warn(
                        'optional mask_backend is not available: {}'.format(str(ex)))

        self._funcs = _funcs
        self._valid = frozenset(self._funcs.keys())

    def get_backend(self, prefs):
        from kwimage import _internal
        if self._funcs is None:
            self._lazy_init()

        valid = ub.oset(prefs) & set(self._funcs)
        if not valid:
            if not _internal.KWIMAGE_DISABLE_IMPORT_WARNINGS:
                warnings.warn('no valid mask backend')
            return None, None
        key = ub.peek(valid)
        func = self._funcs[key]
        return key, func


_backends = _Mask_Backends()


@ub.memoize
def _lazy_mask_backend():
    backend_key, cython_mask = _backends.get_backend(['kwimage', 'pycoco'])
    return cython_mask


__all__ = ['Mask', 'MaskList']


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

    BYTES_RLE = _register('bytes_rle')  # cython compressed RLE
    ARRAY_RLE = _register('array_rle')  # numpy uncompreesed RLE
    C_MASK    = _register('c_mask')     # row-major raw binary mask
    F_MASK    = _register('f_mask')     # column-major raw binary mask

    aliases = {
    }
    for key in cannonical:
        aliases[key] = key


class _MaskConversionMixin(object):
    """
    Mixin class registering conversion functions

    For conversion speeds look into:
        ~/code/kwimage/dev/bench_rle.py
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

            copy (bool):
                if True, we always return copy of the data.
                if False, we try not to return a copy unless necessary.

        Returns:
            Mask: The Mask object with a new backend format

        Example:
            >>> # xdoctest: +REQUIRES(--mask)
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

    @_register_convertor(MaskFormat.BYTES_RLE)
    def to_bytes_rle(self, copy=False):
        """
        Converts the mask format to a bytes-based run-length encoding.

        Args:
            copy (bool):
                if True, we always return copy of the data.
                if False, we try not to return a copy unless necessary.

        Returns:
            Mask: The Mask object with a new backend format

        Example:
            >>> # xdoctest: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import MaskFormat  # NOQA
            >>> mask = Mask.demo()
            >>> print(mask.to_bytes_rle().data['counts'])
            ..._153L;4EL;1DO10;1DO10;1DO10;4EL;4ELW3b0jL^O60...
            >>> print(mask.to_array_rle().data['counts'].tolist())
            [47, 5, 3, 1, 14, 5, 3, 1, 14, 2, 2, 1, 3, 1, 14, ...
            >>> print(mask.to_array_rle().to_bytes_rle().data['counts'])
            ..._153L;4EL;1DO10;1DO10;1DO10;4EL;4ELW3b0jL^O60L0...
        """
        if self.format == MaskFormat.BYTES_RLE:
            return self.copy() if copy else self

        cython_mask = _lazy_mask_backend()

        if self.format == MaskFormat.ARRAY_RLE:
            h, w = self.data['size']
            if self.data.get('order', 'F') != 'F':
                raise ValueError('Expected column-major array RLE')
            if cython_mask is None:
                raise NotImplementedError('pure python version of array_rle to_bytes_rle')
            newdata = cython_mask.frUncompressedRLE([self.data], h, w)[0]
            self = Mask(newdata, MaskFormat.BYTES_RLE)

        elif self.format == MaskFormat.F_MASK:
            f_masks = self.data[:, :, None]
            if cython_mask is None:
                raise NotImplementedError('pure python version of f to to_bytes_rle')
            encoded = cython_mask.encode(f_masks)[0]
            if 'size' in encoded:
                encoded['size'] = list(map(int, encoded['size']))  # python2 fix
            self = Mask(encoded, format=MaskFormat.BYTES_RLE)
        elif self.format == MaskFormat.C_MASK:
            c_mask = self.data
            f_masks = np.asfortranarray(c_mask)[:, :, None]
            if cython_mask is None:
                raise NotImplementedError('pure python version of c to to_bytes_rle')
            encoded = cython_mask.encode(f_masks)[0]
            if 'size' in encoded:
                encoded['size'] = list(map(int, encoded['size']))  # python2 fix
            self = Mask(encoded, format=MaskFormat.BYTES_RLE)
        else:
            raise NotImplementedError(self.format)
        return self

    @_register_convertor(MaskFormat.ARRAY_RLE)
    def to_array_rle(self, copy=False):
        """
        Converts the mask format to an array-based run-length encoding.

        Args:
            copy (bool):
                if True, we always return copy of the data.
                if False, we try not to return a copy unless necessary.

        Returns:
            Mask: the underlying RLE data will be in F-contiguous order.
        """
        if self.format == MaskFormat.ARRAY_RLE:
            return self.copy() if copy else self
        elif self.format == MaskFormat.BYTES_RLE:
            from kwimage.im_runlen import _rle_bytes_to_array
            arr_counts = _rle_bytes_to_array(self.data['counts'])
            encoded = {
                'size': self.data['size'],
                'binary': self.data.get('binary', True),
                'counts': arr_counts,
                'order': self.data.get('order', 'F'),
            }
            encoded['shape'] = self.data.get('shape', encoded['size'])
            self = Mask(encoded, format=MaskFormat.ARRAY_RLE)
        else:
            import kwimage
            f_mask = self.to_fortran_mask().data
            encoded = kwimage.encode_run_length(f_mask, binary=True, order='F')
            # NOTE: Generally `size` means (width, height) and `shape` means
            # (height, width) but shape in this case is in F-order, which means
            # it is (width, hight), so it can be used directly as size
            encoded['size'] = encoded['shape']  # hack in size
            self = Mask(encoded, format=MaskFormat.ARRAY_RLE)
        return self

    @_register_convertor(MaskFormat.F_MASK)
    def to_fortran_mask(self, copy=False):
        """
        Convert the mask format to a dense mask array in columnwise (F) order

        Args:
            copy (bool):
                if True, we always return copy of the data.
                if False, we try not to return a copy unless necessary.

        Returns:
            Mask : the converted mask

        Example:
            >>> import kwimage
            >>> # This is modified version of a segmentation from COCO
            >>> # We have some hard-coded assumptions when handling rles
            >>> # that dont specify shape, order, and binary.
            >>> coco_sseg = {
            >>>     "size": (51, 50),
            >>>     "counts": [26, 2, 651, 3, 13, 1, 313, 12, 6, 3, 12, 322, 11, 323, 10, 325, 8, 93, 416]
            >>> }
            >>> rle = kwimage.Mask(coco_sseg, 'array_rle')
            >>> fmask = rle.to_fortran_mask()
            >>> fmask.data.sum()
            >>> # Note that the returned RLE is in our more explicit encoding
            >>> rle2 = fmask.to_array_rle()
            >>> assert rle2.data['counts'].tolist() == rle.data['counts']
            >>> assert rle2.data['order'] == 'F'
            >>> assert rle2.data['binary'] == True
            >>> assert rle2.data['shape'] == rle.data['size']
        """
        if self.format == MaskFormat.F_MASK:
            return self.copy() if copy else self
        elif self.format == MaskFormat.C_MASK:
            c_mask = self.data.copy() if copy else self.data
            f_mask = np.asfortranarray(c_mask)
        elif self.format == MaskFormat.ARRAY_RLE:
            import kwimage
            encoded = dict(self.data)
            # NOTE: Generally `size` means (width, height) and `shape` means
            # (height, width) but shape in this case is in F-order, which means
            # it is (width, hight), so it can be used directly as size
            # Ideally we are given "shape" instead of "size", but the original
            # COCO RLE's use "size", so we have to accept that here.

            # Handle RLE is in COCO format, thus the defaults passed to
            # decode_run_length should be specified with coco assumptions
            encoded = {
                'counts': self.data['counts'],
                'binary': self.data.get('binary', True),
                'order': self.data.get('order', 'F'),
            }
            if 'shape' in self.data:
                encoded['shape'] = self.data['shape']
            else:
                encoded['shape'] = self.data['size']

            f_mask = kwimage.decode_run_length(**encoded)
        else:
            # NOTE: inefficient, could be improved
            self = self.to_bytes_rle(copy=False)
            cython_mask = _lazy_mask_backend()
            if cython_mask is None:
                raise NotImplementedError('pure python version')
            f_mask = cython_mask.decode([self.data])[:, :, 0]
        self = Mask(f_mask, MaskFormat.F_MASK)
        return self

    @_register_convertor(MaskFormat.C_MASK)
    def to_c_mask(self, copy=False):
        """
        Convert the mask format to a dense mask array in rowwise (C) order

        Args:
            copy (bool):
                if True, we always return copy of the data.
                if False, we try not to return a copy unless necessary.

        Returns:
            Mask: The Mask object with a new backend format
        """
        if self.format == MaskFormat.C_MASK:
            return self.copy() if copy else self
        elif self.format == MaskFormat.F_MASK:
            f_mask = self.data.copy() if copy else self.data
            c_mask = np.ascontiguousarray(f_mask)
        else:
            f_mask = self.to_fortran_mask(copy=False).data
            c_mask = np.ascontiguousarray(f_mask)
        self = Mask(c_mask, MaskFormat.C_MASK)
        return self

    def numpy(self):
        """
        Ensure mask is in numpy format (if possible)

        Returns:
            Mask: The Mask object with a new backend format
        """
        data = self.data
        if self.format in {MaskFormat.C_MASK, MaskFormat.F_MASK}:
            torch = sys.modules.get('torch', None)
            if torch is not None and torch.is_tensor(data):
                data = data.data.cpu().numpy()
        newself = self.__class__(data, self.format)
        return newself

    def tensor(self, device=ub.NoParam):
        """
        Ensure mask is in tensor format (if possible)

        Returns:
            Mask: The Mask object with a new backend format
        """
        data = self.data
        if self.format in {MaskFormat.C_MASK, MaskFormat.F_MASK}:
            torch = sys.modules.get('torch', None)
            if torch is not None and not torch.is_tensor(data):
                data = torch.from_numpy(data)
            if device is not ub.NoParam:
                data = data.to(device)
        newself = self.__class__(data, self.format)
        return newself


class _MaskConstructorMixin(object):
    """
    Alternative ways to construct a masks object
    """

    @classmethod
    def from_polygons(Mask, polygons, dims):
        """
        DEPRICATE: use kwimage.Polygon.to_mask? or kwimage.Mask.coerce?

        Args:
            polygons (ndarray | List[ndarray]): one or more polygons that
                will be joined together. The ndarray may either be an
                Nx2 or a flat c-contiguous array or xy points.
            dims (Tuple): height / width of the source image

        Returns:
            Mask: the new Mask object

        Example:
            >>> # xdoctest: +REQUIRES(--mask)
            >>> polygons = [
            >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
            >>>     np.array([[0, 9],[4, 8],[2, 3]]),
            >>> ]
            >>> dims = (9, 5)
            >>> self = Mask.from_polygons(polygons, dims)
            >>> print(self)
            <Mask({'counts': ...'724;MG2MN16', 'size': [9, 5]}, format=bytes_rle)>
            >>> polygon = polygons[0]
            >>> print(Mask.from_polygons(polygon, dims))
            <Mask({'counts': ...'b04500N2', 'size': [9, 5]}, format=bytes_rle)>
        """
        h, w = dims
        # TODO: holes? geojson?
        if isinstance(polygons, np.ndarray):
            polygons = [polygons]
        flat_polys = [np.array(ps).ravel() for ps in polygons]
        cython_mask = _lazy_mask_backend()
        if cython_mask is None:
            raise NotImplementedError('pure python version from polygons')
        encoded = cython_mask.frPoly(flat_polys, h, w)
        if 'size' in encoded:
            encoded['size'] = list(map(int, encoded['size']))  # python2 fix
        ccs = [Mask(e, MaskFormat.BYTES_RLE) for e in encoded]
        self = Mask.union(*ccs)
        return self

    @classmethod
    def from_mask(Mask, mask, offset=None, shape=None, method='faster'):
        """
        Creates an RLE encoded mask from a raw binary mask.

        You may optionally specify an offset if the mask is part of a larger
        image.

        Args:
            mask (ndarray):
                a binary submask which belongs to a larger image

            offset (Tuple[int, int]):
                top-left xy location of the mask in the larger image

            shape (Tuple[int, int]): shape of the larger image

        SeeAlso:
            ../../test/test_rle.py

        Example:
            >>> mask = Mask.random(shape=(32, 32), rng=0).data
            >>> offset = (30, 100)
            >>> shape = (501, 502)
            >>> self = Mask.from_mask(mask, offset=offset, shape=shape, method='faster')
        """
        if shape is None:
            shape = mask.shape
        if offset is None:
            offset = (0, 0)
        if method == 'naive':
            # inefficent but used to test correctness of algorithms
            import kwimage
            rc_offset = offset[::-1]
            larger = kwimage.subpixel_translate(mask, rc_offset,
                                                output_shape=shape)
            # larger = np.zeros(shape, dtype=mask.dtype)
            # larger_rc = offset[::-1]
            # mask_dims = mask.shape[0:2]
            # index = tuple(slice(s, s + d) for s, d in zip(larger_rc, mask_dims))
            # larger[index] = mask
            self = Mask(larger, MaskFormat.C_MASK).to_array_rle()
        elif method == 'faster':
            import kwimage
            encoded = kwimage.encode_run_length(mask, binary=True, order='F')
            encoded['size'] = encoded['shape']
            self = Mask(encoded, MaskFormat.ARRAY_RLE)
            self = self.translate(offset, shape)
        else:
            raise KeyError(method)
        return self


class _MaskTransformMixin(object):
    """
    Mixin methods relating to geometric transformations of mask objects
    """

    # @profile
    def scale(self, factor, output_dims=None, inplace=False):
        """
        Perform a scale operation on the mask.

        Args:
            factor (float | Tuple[float, float]): the xy scale factor

            input_dims (Tuple[int, int]): unused

            output_dims (Tuple[int, int]): shape of the returned mask

        Returns:
            Mask: the transformed Mask object

        Note:
            * This function has not been optimized and may be inefficient

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Mask.random()
            >>> factor = 5
            >>> inplace = False
            >>> new = self.scale(factor)
            >>> print('new.shape = {!r}'.format(new.shape))
        """
        if not ub.iterable(factor):
            sx = sy = factor
        else:
            sx, sy = factor
        if output_dims is None:
            output_dims = (np.array(self.shape) * np.array((sy, sx))).astype(int)
        # FIXME: the warp breaks when the third row is left out
        transform = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0, 0, 1]])
        new = self.warp(transform, output_dims=output_dims, inplace=inplace)
        return new

    # @profile
    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Perform a matrix warp (e.g. affine or projective) on the underlying
        mask data.

        Args:
            transform (ndarray): the transform matrix

            input_dims (Tuple[int, int]): unused

            output_dims (Tuple[int, int]): shape of the returned mask

        Returns:
            Mask: the transformed Mask object

        Note:
            * This function has not been optimized and may be inefficient

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import kwimage
            >>> self = mask = kwimage.Mask.random()
            >>> transform = np.array([[5., 0, 0], [0, 5, 0], [0, 0, 1]])
            >>> output_dims = np.array(self.shape) * 6
            >>> new = self.warp(transform, output_dims=output_dims)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 1))
            >>> self.draw()
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
            >>> new.draw()

        Example:
            >>> # Verify that the warp transform does roughtly the same thing
            >>> # to a mask and an equivalent polygon
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import kwimage
            >>> input_dims = (100, 100)
            >>> output_dims = (200, 200)
            >>> rng = 92703548026074914707206344922748
            >>> transform = kwimage.Affine.random(shear=(0, 1.), rng=rng)
            >>> mask1 = kwimage.Mask.random(rng=rng, shape=input_dims)
            >>> poly1 = mask1.to_multi_polygon()
            >>> mask2 = mask1.warp(transform.matrix, output_dims=output_dims)
            >>> poly2 = poly1.warp(transform.matrix)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas1_m = np.zeros((*input_dims, 3))
            >>> canvas1_p = np.zeros((*input_dims, 3))
            >>> canvas2_m = np.zeros((*output_dims, 3))
            >>> canvas2_p = np.zeros((*output_dims, 3))
            >>> canvas1_m = mask1.draw_on(canvas1_m)
            >>> canvas1_p = poly1.draw_on(canvas1_p, color='red')
            >>> canvas2_m = mask2.draw_on(canvas2_m)
            >>> canvas2_p = poly2.draw_on(canvas2_p, color='red')
            >>> kwplot.imshow(canvas1_m, fnum=1, pnum=(2, 2, 1))
            >>> kwplot.imshow(canvas1_p, fnum=1, pnum=(2, 2, 2))
            >>> kwplot.imshow(canvas2_m, fnum=1, pnum=(2, 2, 3))
            >>> kwplot.imshow(canvas2_p, fnum=1, pnum=(2, 2, 4))
        """
        # HACK: use brute force just to get this implemented.
        # very inefficient
        import kwimage
        torch = sys.modules.get('torch', None)
        if torch is None:
            raise Exception('need torch to warp raster masks')

        if isinstance(transform, kwimage.Affine):
            transform = transform.matrix

        if transform is None:
            new = self if inplace else Mask(self.data.copy(), self.format)
            return new
        elif isinstance(transform, kwimage.Affine):
            transform = transform.matrix

        c_mask = self.to_c_mask(copy=False).data
        t_mask = torch.Tensor(c_mask)
        matrix = torch.Tensor(transform)
        output_dims = output_dims
        if output_dims is not None:
            if isinstance(output_dims, str):
                if output_dims == 'same':
                    output_dims = self.data.shape[0:2]
                else:
                    raise KeyError(output_dims)
        # TODO: could use kwimage.warp_image here instead if torch is not
        # available.
        w_mask = kwimage.warp_tensor(t_mask, matrix, output_dims=output_dims,
                                     mode='nearest')
        new = self if inplace else Mask(self.data, self.format)
        new.data = w_mask.numpy().astype(np.uint8)
        new.format = MaskFormat.C_MASK
        return new

    # @profile
    def translate(self, offset, output_dims=None, inplace=False):
        """
        Translate the pixel values in the mask.

        Works efficiently in rle or mask format when the offset is integral.

        Args:
            offset (Tuple | Number): x,y offset
            output_dims (Tuple[int, int]): h, w of transformed mask.
                If unspecified the parent shape is used.

            inplace (bool): for api compatability, currently ignored

        Returns:
            Mask: the transformed Mask object

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> shape = (10, 10)
            >>> offset = (1, 1)
            >>> data2 = self.translate(offset, shape).to_c_mask().data
            >>> assert np.all(data2[1:7, 1:7] == self.data[:6, :6])

        Example:
            >>> from kwimage.structs.mask import MaskFormat  # NOQA
            >>> shape = (10, 10)
            >>> offset = (2, -3)
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> # avoid pycocotools
            >>> test_formats = [MaskFormat.C_MASK, MaskFormat.F_MASK, MaskFormat.ARRAY_RLE]
            >>> input_formats = {f: self.toformat(f) for f in test_formats}
            >>> results = {}
            >>> for f, mask in input_formats.items():
            >>>     new_mask = mask.translate(offset, shape)
            >>>     results[f] = new_mask
            >>> assert ub.allsame(
            >>>     [r.toformat(MaskFormat.C_MASK).data
            >>>     for r in results.values()], eq=np.allclose)
        """
        import kwimage
        if output_dims is None:
            output_dims = self.shape
        if not ub.iterable(offset):
            offset = (offset, offset)

        integer_offset = all(isinstance(o, numbers.Integral) for o in offset)
        mask_format = self.format in {MaskFormat.C_MASK, MaskFormat.F_MASK}
        if mask_format or not integer_offset:
            integer_offset = None  # hack
            if integer_offset:
                # TODO: be more efficient
                offset_x, offset_y = offset
                new_data = np.zeros_like(self.data, shape=output_dims)
                new_self = Mask(new_data, self.format)
            else:
                c_data = self.toformat(MaskFormat.C_MASK, copy=False).data
                if c_data.dtype.kind == 'b':
                    c_data = c_data.astype(np.uint8)
                transform = kwimage.Affine.affine(offset=offset)
                dsize = output_dims[::-1]
                new_c_data = kwimage.warp_affine(
                    c_data, transform, dsize=dsize, interpolation='nearest')
                new_c_self = Mask(new_c_data, MaskFormat.C_MASK)
                new_self = new_c_self.toformat(self.format, copy=False)
        else:
            rle = self.to_array_rle(copy=False).data
            new_rle = kwimage.rle_translate(rle, offset, output_dims)
            new_rle['size'] = new_rle['shape']
            new_self = Mask(new_rle, MaskFormat.ARRAY_RLE)
        return new_self


class _MaskDrawMixin(object):
    """
    Mixin methods relating to visualizing mask objects via either
    matplotlib (the ``draw`` method) or opencv (the ``draw_on`` method).
    """

    def draw_on(self, image=None, color='blue', alpha=0.5,
                show_border=False, border_thick=1,
                border_color='white', copy=False):
        """
        Draws the mask on an image

        Args:
            image (ndarray): the image to draw on
            color (str | tuple): color code/rgb of the mask
            alpha (float): mask alpha value
            show_border (bool): draw border around the mask

        Returns:
            ndarray: the image with data drawn on it

        Example:
            >>> from kwimage.structs.mask import *  # NOQA
            >>> import kwimage
            >>> image = kwimage.grab_test_image()
            >>> self = Mask.random(shape=image.shape[0:2])
            >>> canvas = self.draw_on(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Test the case where the mask and image are different sizes
            >>> from kwimage.structs.mask import *  # NOQA
            >>> import kwimage
            >>> image = kwimage.grab_test_image()
            >>> self = Mask.random(shape=np.array(image.shape[0:2]) // 2)
            >>> canvas = self.draw_on(image)
            >>> self = Mask.random(shape=np.array(image.shape[0:2]) * 2)
            >>> canvas = self.draw_on(image)

        Example:
            >>> import kwimage
            >>> color = 'blue'
            >>> self = kwimage.Mask.random(shape=(128, 128))
            >>> # Test drawong on all channel + dtype combinations
            >>> im3 = np.random.rand(128, 128, 3).astype(np.float32)
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
        """
        import kwimage
        import cv2

        if image is None:
            image = np.zeros(self.shape[0:2] + (3,), dtype=np.float32)

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        if alpha is None:
            alpha = 1.0

        # Make an alpha mask with the requested color
        mask = self.to_c_mask().data
        rgb01 = list(kwimage.Color(color).as01())
        rgba01 = np.array(rgb01 + [1])[None, None, :]
        alpha_mask = rgba01 * mask[:, :, None]
        alpha_mask[..., 3] = mask * alpha

        mask_shape = tuple(alpha_mask.shape[0:2])
        canvas_shape = tuple(image.shape[0:2])

        if mask_shape != canvas_shape:
            # Overlay as much as is possible if the shapes dont match
            min_shape = list(map(min, zip(mask_shape, canvas_shape)))
            min_slice = tuple([slice(0, m) for m in min_shape])

            canvas = kwimage.ensure_alpha_channel(image, copy=True)
            alpha_part = alpha_mask[min_slice]
            image_part = image[min_slice]
            # TODO: could use add weighted to get a faster impl
            canvas_part = kwimage.overlay_alpha_images(alpha_part, image_part)
            canvas[min_slice] = canvas_part
        else:
            canvas = kwimage.overlay_alpha_images(alpha_mask, image)

        if show_border:
            # return shape of contours to openCV contours
            polys = self.to_multi_polygon()
            for poly in polys:
                contours = [np.expand_dims(c, axis=1) for c in poly.data['exterior']]
                canvas = cv2.drawContours((canvas * 255.).astype(np.uint8),
                                          contours, -1,
                                          kwimage.Color(border_color).as255(),
                                          border_thick, cv2.LINE_AA)

            canvas = canvas.astype(float) / 255.

        canvas = dtype_fixer(canvas, copy=False)
        return canvas

    def draw(self, color='blue', alpha=0.5, ax=None, show_border=False,
             border_thick=1, border_color='black'):
        """
        Draw on the current matplotlib axis

        Args:
            color (str | tuple): color code/rgb of the mask
            alpha (float): mask alpha value
        """
        import kwimage
        import cv2
        if ax is None:
            from matplotlib import pyplot as plt
            ax = plt.gca()

        mask = self.to_c_mask().numpy().data
        rgb01 = list(kwimage.Color(color).as01())
        rgba01 = np.array(rgb01 + [1])[None, None, :]
        alpha_mask = rgba01 * mask[:, :, None]
        alpha_mask[..., 3] = mask * alpha

        if show_border:
            # Add alpha channel to color
            border_color_tup = kwimage.Color(border_color).as255()
            border_color_tup = (border_color_tup[0], border_color_tup[1],
                                border_color_tup[2], 255 * alpha)

            # return shape of contours to openCV contours
            polys = self.to_multi_polygon()
            for poly in polys:
                contours = [np.expand_dims(c, axis=1) for c in poly.data['exterior']]
                alpha_mask = cv2.drawContours(
                    (alpha_mask * 255.).astype(np.uint8),
                    contours, -1, border_color_tup, border_thick, cv2.LINE_AA)

            alpha_mask = alpha_mask.astype(float) / 255.

        ax.imshow(alpha_mask)


class Mask(ub.NiceRepr, _MaskConversionMixin, _MaskConstructorMixin,
           _MaskTransformMixin, _MaskDrawMixin):
    """
    Manages a single segmentation mask and can convert to and from
    multiple formats including:

        * bytes_rle - byte encoded run length encoding
        * array_rle - raw run length encoding
        * c_mask - c-style binary mask
        * f_mask - fortran-style binary mask

    Example:
        >>> # xdoc: +REQUIRES(--mask)
        >>> # a ms-coco style compressed bytes rle segmentation
        >>> segmentation = {'size': [5, 9], 'counts': ';?1B10O30O4'}
        >>> mask = Mask(segmentation, 'bytes_rle')
        >>> # convert to binary numpy representation
        >>> binary_mask = mask.to_c_mask().data
        >>> print(ub.urepr(binary_mask.tolist(), nl=1, nobr=1))
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0],

    """

    def __init__(self, data=None, format=None):
        self.data = data
        self.format = format

    @property
    def dtype(self):
        try:
            return self.data.dtype
        except Exception:
            print('kwimage.mask: no dtype for ' + str(type(self.data)))
            raise

    def __nice__(self):
        return '{}, format={}'.format(ub.urepr(self.data, nl=0), self.format)

    @classmethod
    def random(Mask, rng=None, shape=(32, 32)):
        """
        Create a random binary mask object

        Args:
            rng (int | RandomState | None): the random seed

            shape (Tuple[int, int]): the height / width of the returned mask

        Returns:
            Mask: the random mask

        Example:
            >>> import kwimage
            >>> mask = kwimage.Mask.random()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> mask.draw()
            >>> kwplot.show_if_requested()
        """
        import kwarray
        import kwimage
        rng = kwarray.ensure_rng(rng)
        # Use random heatmap to make some blobs for the mask
        heatmap = kwimage.Heatmap.random(dims=shape, rng=rng, classes=2)
        probs = heatmap.data['class_probs'][1]
        c_mask = (probs > probs.mean()).astype(np.uint8)
        self = Mask(c_mask, MaskFormat.C_MASK)
        return self

    @classmethod
    def demo(cls):
        """
        Demo mask with holes and disjoint shapes

        Returns:
            Mask: the demo mask
        """
        text = ub.codeblock(
            '''
            ................................
            ..ooooooo....ooooooooooooo......
            ..ooooooo....o...........o......
            ..oo...oo....o.oooooooo..o......
            ..oo...oo....o.o......o..o......
            ..ooooooo....o.o..oo..o..o......
            .............o.o...o..o..o......
            .............o.o..oo..o..o......
            .............o.o......o..o......
            ..ooooooo....o.oooooooo..o......
            .............o...........o......
            .............o...........o......
            .............ooooooooooooo......
            .............o...........o......
            .............o...........o......
            .............o....ooooo..o......
            .............o....o...o..o......
            .............o....ooooo..o......
            .............o...........o......
            .............ooooooooooooo......
            ................................
            ................................
            ................................
            ''')
        self = cls.from_text(text, zero_chr='.')
        return self

    @classmethod
    def from_text(cls, text, zero_chr='.', shape=None, has_border=False):
        """
        Construct a mask from a text art representation

        Args:
            text (str):
                the text representing a mask

            zero_chr (str):
                the character that represents a zero

            shape (None | Tuple[int, int]):
                if specified force a specific height / width, otherwise
                the character extent determines this.

            has_border (bool):
                if True, assume the characters at the edge
                are representing a border and remove them.

        Example:
            >>> import kwimage
            >>> import ubelt as ub
            >>> text = ub.indent(ub.codeblock(
            >>>     '''
            >>>     ooo
            >>>     ooo
            >>>     ooooo
            >>>         o
            >>>     '''))
            >>> mask = kwimage.Mask.from_text(text, zero_chr=' ')
            >>> print(mask.data)
            [[0 0 0 0 1 1 1 0 0]
             [0 0 0 0 1 1 1 0 0]
             [0 0 0 0 1 1 1 1 1]
             [0 0 0 0 0 0 0 0 1]]

        Example:
            >>> import kwimage
            >>> import ubelt as ub
            >>> text = ub.codeblock(
            >>>     '''
            >>>     +------------+
            >>>     |            |
            >>>     |    ooo     |
            >>>     |    ooo     |
            >>>     |    ooooo   |
            >>>     |        o   |
            >>>     |            |
            >>>     +------------+
            >>>     ''')
            >>> mask = kwimage.Mask.from_text(text, has_border=True, zero_chr=' ')
            >>> print(mask.data)
            [[0 0 0 0 0 0 0 0 0 0 0 0]
             [0 0 0 0 1 1 1 0 0 0 0 0]
             [0 0 0 0 1 1 1 0 0 0 0 0]
             [0 0 0 0 1 1 1 1 1 0 0 0]
             [0 0 0 0 0 0 0 0 1 0 0 0]
             [0 0 0 0 0 0 0 0 0 0 0 0]]
        """
        lines = text.split('\n')
        data = [[0 if c == zero_chr else 1 for c in line] for line in lines]
        max_width = max(len(row) for row in data)
        max_height = len(data)
        if shape is not None:
            max_height, max_width = shape

        # Pad out (or shrink) the width of each row
        data = [
            row[0:max_width] if len(row) >= max_width else
            row + [0] * (max_width - len(row))
            for row in data
        ]

        # Pad out (or shrink) the height of the columns
        extra_rows = max_height - len(data)
        if extra_rows > 0:
            data = [[0] * max_width for _ in range(extra_rows)]
        else:
            data = data[0:max_height]

        data = np.array(data).astype(np.uint8)
        if has_border:
            data = data[1:-1, 1:-1]
        self = cls(data, format=MaskFormat.C_MASK)
        return self

    def copy(self):
        """
        Performs a deep copy of the mask data

        Returns:
            Mask: the copied mask

        Example:
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> other = self.copy()
            >>> assert other.data is not self.data
        """
        return Mask(copy.deepcopy(self.data), self.format)

    def union(self, *others):
        """
        This can be used as a staticmethod or an instancemethod

        Args:
            *others: multiple input masks to union

        Returns:
            Mask: the unioned mask

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import *  # NOQA
            >>> masks = [Mask.random(shape=(8, 8), rng=i) for i in range(2)]
            >>> mask = Mask.union(*masks)
            >>> print(mask.area)
            >>> masks = [m.to_c_mask() for m in masks]
            >>> mask = Mask.union(*masks)
            >>> print(mask.area)

            >>> masks = [m.to_bytes_rle() for m in masks]
            >>> mask = Mask.union(*masks)
            >>> print(mask.area)

        Ignore:
            import ubelt as ub
            ti = ub.Timerit(100, bestof=10, verbose=2)
            masks = [Mask.random(shape=(172, 172), rng=i) for i in range(2)]
            for timer in ti.reset('native rle union'):
                masks = [m.to_bytes_rle() for m in masks]
                with timer:
                    mask = Mask.union(*masks)
            for timer in ti.reset('native cmask union'):
                masks = [m.to_c_mask() for m in masks]
                with timer:
                    mask = Mask.union(*masks)
            for timer in ti.reset('cmask->rle union'):
                masks = [m.to_c_mask() for m in masks]
                with timer:
                    mask = Mask.union(*[m.to_bytes_rle() for m in masks])
        """
        if isinstance(self, Mask):
            cls = self.__class__
            items = list(it.chain([self], others))
        else:
            cls = Mask
            items = others

        if len(items) == 0:
            raise Exception('empty union')
        else:
            format = items[0].format
            if format == MaskFormat.C_MASK:
                datas = [item.to_c_mask().data for item in items]
                new_data = np.bitwise_or.reduce(datas)
                new = cls(new_data, MaskFormat.C_MASK)
            elif format == MaskFormat.BYTES_RLE:
                datas = [item.to_bytes_rle().data for item in items]
                cython_mask = _lazy_mask_backend()
                if cython_mask is None:
                    raise NotImplementedError('pure python version of bytes rle union')
                new_data = cython_mask.merge(datas, intersect=0)
                if 'size' in new_data:
                    new_data['size'] = list(map(int, new_data['size']))  # python2 fix
                new = cls(new_data, MaskFormat.BYTES_RLE)
            else:
                datas = [item.to_bytes_rle().data for item in items]
                if cython_mask is None:
                    raise NotImplementedError('pure python version of union')
                new_rle = cython_mask.merge(datas, intersect=0)
                if 'size' in new_rle:
                    new_rle['size'] = list(map(int, new_rle['size']))  # python2 fix
                new = cls(new_rle, MaskFormat.BYTES_RLE)
        return new

    def intersection(self, *others):
        """
        This can be used as a staticmethod or an instancemethod

        Args:
            *others: multiple input masks to intersect

        Returns:
            Mask: the intersection of the masks

        Example:
            >>> n = 3
            >>> masks = [Mask.random(shape=(8, 8), rng=i) for i in range(n)]
            >>> items = masks
            >>> mask = Mask.intersection(*masks)
            >>> areas = [item.area for item in items]
            >>> print('areas = {!r}'.format(areas))
            >>> print(mask.area)
            >>> print(Mask.intersection(*masks).area / Mask.union(*masks).area)
        """
        if isinstance(self, Mask):
            cls = self.__class__
            items = list(it.chain([self], others))
        else:
            cls = Mask
            items = others

        if len(items) == 0:
            raise Exception('empty intersection')
        else:
            format = items[0].format
            items2 = [item.toformat(format) for item in items]

            if format == MaskFormat.C_MASK or format == MaskFormat.F_MASK:
                bit_data = [item.data for item in items2]
                new_data = np.bitwise_and.reduce(bit_data)
                new = cls(new_data, format=format)
            else:
                rle_datas = [item.data for item in items]
                cython_mask = _lazy_mask_backend()
                if cython_mask is None:
                    raise NotImplementedError('pure python version of mask intersection')
                encoded = cython_mask.merge(rle_datas, intersect=1)
                if 'size' in encoded:
                    encoded['size'] = list(map(int, encoded['size']))  # python2 fix
                new = cls(encoded, MaskFormat.BYTES_RLE)
        return new

    @property
    def shape(self):
        if self.format in {MaskFormat.BYTES_RLE, MaskFormat.ARRAY_RLE}:
            if 'shape' in self.data:
                return self.data['shape']
            else:
                return self.data['size']
        if self.format in {MaskFormat.C_MASK, MaskFormat.F_MASK}:
            return self.data.shape

    @property
    def area(self):
        """
        Returns the number of non-zero pixels

        Returns:
            int: the number of non-zero pixels

        Example:
            >>> self = Mask.demo()
            >>> self.area
            150
        """
        if self.format == MaskFormat.C_MASK:
            return self.data.sum()
        elif self.format == MaskFormat.F_MASK:
            return self.data.sum()
        elif self.format == MaskFormat.BYTES_RLE:
            cython_mask = _lazy_mask_backend()
            if cython_mask is None:
                raise NotImplementedError('pure python version mask area')
            return cython_mask.area([self.data])[0]
        else:
            raise NotImplementedError('Mask.area for {}'.format(self.format))

    def get_patch(self):
        """
        Extract the patch with non-zero data

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import *  # NOQA
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> self.get_patch()
        """
        x, y, w, h = self.get_xywh().astype(int).tolist()
        output_dims = (h, w)
        xy_offset = (-x, -y)
        temp = self.translate(xy_offset, output_dims)
        patch = temp.to_c_mask().data
        return patch

    # @profile
    def get_xywh(self):
        """
        Gets the bounding xywh box coordinates of this mask

        Returns:
            ndarray: x, y, w, h: Note we dont use a Boxes object because
                a general singular version does not yet exist.

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> self.get_xywh().tolist()
            >>> self = Mask.random(rng=0).translate((10, 10))
            >>> self.get_xywh().tolist()

        Example:
            >>> # test empty case
            >>> import kwimage
            >>> self = kwimage.Mask(np.empty((0, 0), dtype=np.uint8), format='c_mask')
            >>> assert self.get_xywh().tolist() == [0, 0, 0, 0]

        Ignore:
            >>> import kwimage
            >>> self = kwimage.Mask(np.zeros((768, 768), dtype=np.uint8), format='c_mask')
            >>> x_coords = np.array([621, 752])
            >>> y_coords = np.array([366, 292])
            >>> self.data[y_coords, x_coords] = 1
            >>> self.get_xywh()

            >>> # References:
            >>> # https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
            >>> # https://answers.opencv.org/question/4183/what-is-the-best-way-to-find-bounding-box-for-binary-mask/
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('time'):
            >>>     with timer:
            >>>         y_coords, x_coords = np.where(self.data)
            >>> #
            >>> for timer in ti.reset('time'):
            >>>     with timer:
            >>>         cv2.findNonZero(data)

            self.data = np.random.rand(800, 700) > 0.5

            import timerit
            ti = timerit.Timerit(100, bestof=10, verbose=2)
            for timer in ti.reset('time'):
                with timer:
                    y_coords, x_coords = np.where(self.data)
            #
            for timer in ti.reset('time'):
                with timer:
                    data = np.ascontiguousarray(self.data).astype(np.uint8)
                    cv2_coords = cv2.findNonZero(data)

            >>> poly = self.to_multi_polygon()
        """
        import cv2
        if self.format == MaskFormat.C_MASK:
            # findNonZero seems much faster than np.where
            data = np.ascontiguousarray(self.data).astype(np.uint8)
            cv2_coords = cv2.findNonZero(data)
            if cv2_coords is None:
                xywh = np.array([0, 0, 0, 0])
            else:
                x_coords = cv2_coords[:, 0, 0]
                y_coords = cv2_coords[:, 0, 1]
                # # y_coords, x_coords = np.where(self.data)
                # if len(x_coords) == 0:
                #     xywh = np.array([0, 0, 0, 0])
                # else:
                tl_x = x_coords.min()
                br_x = x_coords.max()
                tl_y = y_coords.min()
                br_y = y_coords.max()
                w = br_x - tl_x
                h = br_y - tl_y
                xywh = np.array([tl_x, tl_y, w, h])
        elif self.format == MaskFormat.F_MASK:
            x_coords, y_coords = np.where(self.data)
            if len(x_coords) == 0:
                xywh = np.array([0, 0, 0, 0])
            else:
                tl_x = x_coords.min()
                br_x = x_coords.max()
                tl_y = y_coords.min()
                br_y = y_coords.max()
                w = br_x - tl_x
                h = br_y - tl_y
                xywh = np.array([tl_x, tl_y, w, h])
        else:
            try:
                self_rle = self.to_bytes_rle()
                cython_mask = _lazy_mask_backend()
                if cython_mask is None:
                    raise NotImplementedError('pure python version get_xywh')
                xywh = cython_mask.toBbox([self_rle.data])[0]
            except NotImplementedError:
                self_c = self.to_c_mask()  # alternate path
                xywh = self_c.get_xywh()
        return xywh

    def bounding_box(self):
        """
        Returns an axis-aligned bounding box for this mask

        Returns:
            kwimage.Boxes
        """
        import kwimage
        xywh = self.get_xywh()
        boxes = kwimage.Boxes([xywh], 'xywh')
        return boxes

    def get_polygon(self):
        """
        DEPRECATED: USE to_multi_polygon

        Returns a list of (x,y)-coordinate lists. The length of the list is
        equal to the number of disjoint regions in the mask.

        Returns:
            List[ndarray]: polygon around each connected component of the
                mask. Each ndarray is an Nx2 array of xy points.

        Note:
            The returned polygon may not surround points that are only one
            pixel thick.

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import *  # NOQA
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> polygons = self.get_polygon()
            >>> print('polygons = ' + ub.urepr(polygons))
            >>> polygons = self.get_polygon()
            >>> self = self.to_bytes_rle()
            >>> other = Mask.from_polygons(polygons, self.shape)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> image = np.ones(self.shape)
            >>> image = self.draw_on(image, color='blue')
            >>> image = other.draw_on(image, color='red')
            >>> kwplot.imshow(image)

        Ignore:
            polygons = [
                np.array([[6, 4],[7, 4]], dtype=np.int32),
                np.array([[0, 1],[0, 3],[2, 3],[2, 1]], dtype=np.int32),
            ]
        """
        import cv2
        ub.schedule_deprecation(
            'kwimage', 'Mask.get_polygon', 'method',
            migration='use Mask.to_multi_polygon instead',
            deprecate='0.9.5', error='1.0.0', remove='1.1.0',
        )
        p = 2

        if 0:
            mask = self.to_c_mask().data
            offset = (-p, -p)
        else:
            # It should be faster to only extract the patch of non-zero values
            x, y, w, h = self.get_xywh().astype(int).tolist()
            output_dims = (h, w)
            xy_offset = (-x, -y)
            temp = self.translate(xy_offset, output_dims)
            mask = temp.to_c_mask().data
            offset = (x - p, y - p)

        padded_mask = cv2.copyMakeBorder(mask, p, p, p, p,
                                         cv2.BORDER_CONSTANT, value=0)

        # print('src =\n{!r}'.format(padded_mask))
        kernel = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        padded_mask = cv2.dilate(padded_mask, kernel, dst=padded_mask)
        # print('dst =\n{!r}'.format(padded_mask))

        mode = cv2.RETR_LIST
        # mode = cv2.RETR_EXTERNAL

        # https://docs.opencv.org/3.1.0/d3/dc0/
        # group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff

        method = cv2.CHAIN_APPROX_SIMPLE
        # method = cv2.CHAIN_APPROX_NONE
        # method = cv2.CHAIN_APPROX_TC89_KCOS
        # Different versions of cv2 have different return types
        _ret = cv2.findContours(padded_mask, mode, method, offset=offset)
        if len(_ret) == 2:
            _contours, _hierarchy = _ret
        else:
            _img, _contours, _hierarchy = _ret

        polygon = [c[:, 0, :] for c in _contours]

        if False:
            import kwplot
            import kwimage
            kwplot.autompl()
            # Note that cv2 draw contours doesnt have the 1-pixel thick problem
            # it seems to just be the way the coco implementation is
            # interpreting polygons.
            image = kwimage.atleast_3channels(mask)
            canvas = np.zeros(image.shape, dtype="uint8")
            cv2.drawContours(canvas, _contours, -1, (255, 0, 0), 1)
            kwplot.imshow(canvas)

        return polygon

    def to_mask(self, dims=None, pixels_are='points'):
        """
        Converts to a mask object (which does nothing because this already is
        mask object!)

        Returns:
            kwimage.Mask
        """
        return self

    def to_boxes(self):
        """
        Returns the bounding box of the mask.

        Returns:
            kwimage.Boxes
        """
        import kwimage
        boxes = kwimage.Boxes([self.get_xywh()], 'xywh')
        return boxes

    # @profile
    def to_multi_polygon(self, pixels_are='points'):
        """
        Returns a MultiPolygon object fit around this raster including disjoint
        pieces and holes.

        Args:
            pixel_are (str):
                Can either be "points" or "areas".

                If pixels are "points", the we treat each pixel (i, j) as a
                single infinitely small point at (i, j). As such, some polygons
                may have zero area.

                If pixels are "areas", then each pixel (i, j) represents a
                square with coordinates ([i - 0.5, j - 0.5], [i + 0.5, j -
                0.5], [i + 0.5, j + 0.5], and [i - 0.5, j + 0.5]). Must have
                rasterio installed to use this method.

        Returns:
            kwimage.MultiPolygon: vectorized representation

        Note:
            The OpenCV (and thus this function) coordinate system places
            coordinates at the center of pixels, and the polygon is traced
            tightly around these coordinates. A single pixel is not considered
            to have any width, so polygon edges will directly trace through the
            centers of pixels, and in the case where an object is only 1 pixel
            thick, this will produce a polygon that is not a valid shapely
            polygon.

        TODO:
            - [x] add a flag where polygons consider pixels to have width and the resulting polygon is traced around the pixel edges, not the pixel centers.

            - [ ] Polygons and Masks should keep track of what "pixels_are"

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import *  # NOQA
            >>> self = Mask.demo()
            >>> self = self.scale(5)
            >>> multi_poly = self.to_multi_polygon()
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(color='red')
            >>> multi_poly.scale(1.1).draw(color='blue')

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> image = np.ones(self.shape)
            >>> image = self.draw_on(image, color='blue')
            >>> #image = other.draw_on(image, color='red')
            >>> kwplot.imshow(image)
            >>> multi_poly.draw()

        Example:
            >>> # Test empty cases
            >>> import kwimage
            >>> mask0 = kwimage.Mask(np.zeros((0, 0), dtype=np.uint8), format='c_mask')
            >>> mask1 = kwimage.Mask(np.zeros((1, 1), dtype=np.uint8), format='c_mask')
            >>> mask2 = kwimage.Mask(np.zeros((2, 2), dtype=np.uint8), format='c_mask')
            >>> mask3 = kwimage.Mask(np.zeros((3, 3), dtype=np.uint8), format='c_mask')
            >>> pixels_are = 'points'
            >>> poly0 = mask0.to_multi_polygon(pixels_are=pixels_are)
            >>> poly1 = mask1.to_multi_polygon(pixels_are=pixels_are)
            >>> poly2 = mask2.to_multi_polygon(pixels_are=pixels_are)
            >>> poly3 = mask3.to_multi_polygon(pixels_are=pixels_are)
            >>> assert len(poly0) == 0
            >>> assert len(poly1) == 0
            >>> assert len(poly2) == 0
            >>> assert len(poly3) == 0
            >>> # xdoctest: +REQUIRES(module:rasterio)
            >>> pixels_are = 'areas'
            >>> poly0 = mask0.to_multi_polygon(pixels_are=pixels_are)
            >>> poly1 = mask1.to_multi_polygon(pixels_are=pixels_are)
            >>> poly2 = mask2.to_multi_polygon(pixels_are=pixels_are)
            >>> poly3 = mask3.to_multi_polygon(pixels_are=pixels_are)
            >>> assert len(poly0) == 0
            >>> assert len(poly1) == 0
            >>> assert len(poly2) == 0
            >>> assert len(poly3) == 0

        Example:
            >>> # Test full ones cases
            >>> import kwimage
            >>> mask1 = kwimage.Mask(np.ones((1, 1), dtype=np.uint8), format='c_mask')
            >>> mask2 = kwimage.Mask(np.ones((2, 2), dtype=np.uint8), format='c_mask')
            >>> mask3 = kwimage.Mask(np.ones((3, 3), dtype=np.uint8), format='c_mask')
            >>> pixels_are = 'points'
            >>> poly1 = mask1.to_multi_polygon(pixels_are=pixels_are)
            >>> poly2 = mask2.to_multi_polygon(pixels_are=pixels_are)
            >>> poly3 = mask3.to_multi_polygon(pixels_are=pixels_are)
            >>> assert np.all(poly1.to_mask(mask1.shape).data == 1)
            >>> assert np.all(poly2.to_mask(mask2.shape).data == 1)
            >>> assert np.all(poly3.to_mask(mask3.shape).data == 1)
            >>> # xdoctest: +REQUIRES(module:rasterio)
            >>> pixels_are = 'areas'
            >>> poly1 = mask1.to_multi_polygon(pixels_are=pixels_are)
            >>> poly2 = mask2.to_multi_polygon(pixels_are=pixels_are)
            >>> poly3 = mask3.to_multi_polygon(pixels_are=pixels_are)
            >>> assert np.all(poly1.to_mask(mask1.shape).data == 1)
            >>> assert np.all(poly2.to_mask(mask2.shape).data == 1)
            >>> assert np.all(poly3.to_mask(mask3.shape).data == 1)

        Example:
            >>> # Corner case, only two pixels are on
            >>> import kwimage
            >>> self = kwimage.Mask(np.zeros((768, 768), dtype=np.uint8), format='c_mask')
            >>> x_coords = np.array([621, 752])
            >>> y_coords = np.array([366, 292])
            >>> self.data[y_coords, x_coords] = 1
            >>> poly = self.to_multi_polygon()

        Ignore:
            poly.to_mask(self.shape).data.sum()
            self.to_array_rle().to_c_mask().data.sum()
            temp.to_c_mask().data.sum()

        Example:
            >>> # xdoctest: +REQUIRES(module:rasterio)
            >>> import kwimage
            >>> dims = (10, 10)
            >>> data = np.zeros(dims, dtype=np.uint8)
            >>> data[0, 3:5] = 1
            >>> data[9, 1:3] = 1
            >>> data[3:5, 0:2] = 1
            >>> data[1, 1] = 1
            >>> # 1 pixel L shape
            >>> data[3, 5] = 1
            >>> data[4, 5] = 1
            >>> data[4, 6] = 1
            >>> data[1, 5] = 1
            >>> data[2, 6] = 1
            >>> data[3, 7] = 1
            >>> data[6, 1] = 1
            >>> data[7, 1] = 1
            >>> data[7, 2] = 1
            >>> data[6:10, 5] = 1
            >>> data[6:10, 8] = 1
            >>> data[9, 5:9] = 1
            >>> data[6, 5:9] = 1
            >>> #data = kwimage.imresize(data, scale=2.0, interpolation='nearest')
            >>> self = kwimage.Mask.coerce(data)
            >>> #self = self.translate((0, 0), output_dims=(10, 9))
            >>> self = self.translate((0, 1), output_dims=(11, 11))
            >>> dims = self.shape[0:2]
            >>> multi_poly1 = self.to_multi_polygon(pixels_are='points')
            >>> multi_poly2 = self.to_multi_polygon(pixels_are='areas')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pretty_data = kwplot.make_heatmask(self.data/1.0, cmap='magma')[..., 0:3]
            >>> def _pixel_grid_lines(self, ax):
            >>>     h, w = self.data.shape[0:2]
            >>>     ybasis = np.arange(0, h) + 0.5
            >>>     xbasis = np.arange(0, w) + 0.5
            >>>     xmin = 0 - 0.5
            >>>     xmax = w - 0.5
            >>>     ymin = 0 - 0.5
            >>>     ymax = h - 0.5
            >>>     ax.hlines(y=ybasis, xmin=xmin, xmax=xmax, color="gainsboro")
            >>>     ax.vlines(x=xbasis, ymin=ymin, ymax=ymax, color="gainsboro")
            >>> def _setup_grid(self, pnum):
            >>>     ax = kwplot.imshow(pretty_data, show_ticks=True, pnum=pnum)[1]
            >>>     # The gray ticks show the center of the pixels
            >>>     ax.grid(color='dimgray', linewidth=0.5)
            >>>     ax.set_xticks(np.arange(self.data.shape[1]))
            >>>     ax.set_yticks(np.arange(self.data.shape[0]))
            >>>     # Also draw black lines around the edges of the pixels
            >>>     _pixel_grid_lines(self, ax=ax)
            >>>     return ax
            >>> # Overlay the extracted polygons
            >>> ax = _setup_grid(self, pnum=(2, 3, 1))
            >>> ax.set_title('input binary mask data')
            >>> ax = _setup_grid(self, pnum=(2, 3, 2))
            >>> multi_poly1.draw(linewidth=5, alpha=0.5, radius=0.2, ax=ax, fill=False, vertex=0.2)
            >>> ax.set_title('opencv "point" polygons')
            >>> ax = _setup_grid(self, pnum=(2, 3, 3))
            >>> multi_poly2.draw(linewidth=5, alpha=0.5, radius=0.2, color='limegreen', ax=ax, fill=False, vertex=0.2)
            >>> ax.set_title('raterio "area" polygons')
            >>> ax.figure.suptitle(ub.codeblock(
            >>>     '''
            >>>     Gray lines are coordinates and pass through pixel centers (integer coords)
            >>>     White lines trace pixel boundaries (fractional coords)
            >>>     '''))
            >>> raster1 = multi_poly1.to_mask(dims, pixels_are='points')
            >>> raster2 = multi_poly2.to_mask(dims, pixels_are='areas')
            >>> kwplot.imshow(raster1.draw_on(), pnum=(2, 3, 5), title='rasterized')
            >>> kwplot.imshow(raster2.draw_on(), pnum=(2, 3, 6), title='rasterized')
        """
        from kwimage.structs.polygon import Polygon, MultiPolygon
        # Note: it is not necessarilly faster to to only exact the patch of
        # non-zero values
        temp_mask = self.to_c_mask(copy=False).data
        if temp_mask.dtype.kind == 'b':
            temp_mask = temp_mask.astype(np.uint8)
        # TODO: polygons and masks should keep track what "pixels_are"
        polys = _find_contours(temp_mask, pixels_are=pixels_are)
        poly_list = [Polygon(**data) for data in polys]
        multi_poly = MultiPolygon(poly_list)
        return multi_poly

    def get_convex_hull(self):
        """
        Returns a list of xy points around the convex hull of this mask

        Note:
            The returned polygon may not surround points that are only one
            pixel thick.

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> self = Mask.random(shape=(8, 8), rng=0)
            >>> polygons = self.get_convex_hull()
            >>> print('polygons = ' + ub.urepr(polygons))
            >>> other = Mask.from_polygons(polygons, self.shape)
        """
        import cv2
        mask = self.to_c_mask().data
        cc_y, cc_x = np.where(mask)
        points = np.vstack([cc_x, cc_y]).T
        hull = cv2.convexHull(points)[:, 0, :]
        return hull

    def iou(self, other):
        """
        The area of intersection over the area of union

        TODO:
            - [ ] Write plural Masks version of this class, which should be able to perform this operation more efficiently.

        CommandLine:
            xdoctest -m kwimage.structs.mask Mask.iou

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> self = Mask.demo()
            >>> other = self.translate(1)
            >>> iou = self.iou(other)
            >>> print('iou = {:.4f}'.format(iou))
            iou = 0.0830
            >>> iou2 = self.intersection(other).area / self.union(other).area
            >>> print('iou2 = {:.4f}'.format(iou2))
        """
        item1 = self.to_bytes_rle(copy=False).data
        item2 = other.to_bytes_rle(copy=False).data
        # I'm not sure what passing `pyiscrowd` actually does here
        # TODO: determine what `pyiscrowd` does, and document it.
        pyiscrowd = np.array([0], dtype=np.uint8)
        cython_mask = _lazy_mask_backend()
        if cython_mask is None:
            raise NotImplementedError('pure python version iou')
        iou = cython_mask.iou([item1], [item2], pyiscrowd)[0, 0]
        return iou

    @classmethod
    def coerce(Mask, data, dims=None):
        """
        Attempts to auto-inspect the format of the data and conver to Mask

        Args:
            data (Any) : the data to coerce

            dims (Tuple): required for certain formats like polygons
                height / width of the source image

        Returns:
            Mask: the constructed mask object

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> segmentation = {'size': [5, 9], 'counts': ';?1B10O30O4'}
            >>> polygon = [
            >>>     [np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]])],
            >>>     [np.array([[2, 1],[2, 2],[4, 2],[4, 1]])],
            >>> ]
            >>> dims = (9, 5)
            >>> mask = (np.random.rand(32, 32) > .5).astype(np.uint8)
            >>> Mask.coerce(polygon, dims).to_bytes_rle()
            >>> Mask.coerce(segmentation).to_bytes_rle()
            >>> Mask.coerce(mask).to_bytes_rle()
        """
        # TODO: this could be more explicitly written
        from kwimage.structs.segmentation import _coerce_coco_segmentation
        self = _coerce_coco_segmentation(data, dims)
        self = self.to_mask(dims)
        return self

    def _to_coco(self):
        """ use to_coco instead """
        return self.to_coco()

    def to_coco(self, style='orig'):
        """
        Convert the Mask to a COCO json representation based on the current
        format.

        A COCO mask is formatted as a run-length-encoding (RLE), of which there
        are two variants: (1) a array RLE, which is slightly more readable and
        extensible, and (2) a bytes RLE, which is slightly more concise. The
        returned format will depend on the current format of the Mask object.
        If it is in "bytes_rle" format, it will be returned in that format,
        otherwise it will be converted to the "array_rle" format and returned
        as such.

        Args:
            style (str): Does nothing for this particular method, exists for
                API compatibility and if alternate encoding styles are
                implemented in the future.

        Returns:
            dict: either a bytes-rle or array-rle encoding, depending
                on the current mask format. The keys in this dictionary
                are as follows:

                counts (List[int] | str): the array or bytes rle encoding

                size (Tuple[int]): the height and width of the encoded mask
                    *see note*.

                shape (Tuple[int]): only present in array-rle mode. This
                    is also the height/width of the underlying encoded array.
                    This exists for semantic consistency with other kwimage
                    conventions, and is not part of the original coco spec.

                order (str): only present in array-rle mode.
                    Either C or F, indicating if counts is aranged in row-major
                    or column-major order. For COCO-compatibility this is
                    always returned in F (column-major) order.

                binary (bool): only present in array-rle mode.
                    For COCO-compatibility this is always returned as False,
                    indicating the mask only contains binary 0 or 1 values.

        Note:
            The output dictionary will contain a key named "size", this is the
            only location in kwimage where "size" refers to a tuple in
            (height/width) order, in order to be backwards compatible with the
            original coco spec. In all other locations in kwimage a "size" will
            refer to a (width/height) ordered tuple.

        SeeAlso:
            :func: kwimage.im_runlen.encode_run_length - backend function that
                does array-style run length encoding.

        Example:
            >>> # xdoc: +REQUIRES(--mask)
            >>> from kwimage.structs.mask import *  # NOQA
            >>> self = Mask.demo()
            >>> coco_data1 = self.toformat('array_rle').to_coco()
            >>> coco_data2 = self.toformat('bytes_rle').to_coco()
            >>> print('coco_data1 = {}'.format(ub.urepr(coco_data1, nl=1)))
            >>> print('coco_data2 = {}'.format(ub.urepr(coco_data2, nl=1)))
            coco_data1 = {
                'binary': True,
                'counts': [47, 5, 3, 1, 14, ... 1, 4, 19, 141],
                'order': 'F',
                'shape': (23, 32),
                'size': (23, 32),
            }
            coco_data2 = {
                'counts': '_153L;4EL...ON3060L0N060L0Nb0Y4',
                'size': [23, 32],
            }
        """
        use_bytes = (self.format == MaskFormat.BYTES_RLE)
        if use_bytes:
            try:
                bytes_rle = self.to_bytes_rle()
            except NotImplementedError:
                use_bytes = False

        if use_bytes:
            # This is actually the original style, but it relies on
            # to_bytes_rle, which doesnt always work.
            data = bytes_rle.data.copy()
            text = data['counts']
            if isinstance(text, bytes):
                text = text.decode('utf8')
            data['counts'] = text
            return data
        else:
            data = self.to_array_rle().data.copy()
            data['counts'] = data['counts'].tolist()
        return data


class MaskList(_generic.ObjectList):
    """
    Store and manipulate multiple masks, usually within the same image
    """

    def to_polygon_list(self):
        """
        Converts all mask objects to multi-polygon objects

        Returns:
            kwimage.PolygonList
        """
        import kwimage
        new = kwimage.PolygonList([
            None if mask is None else mask.to_multi_polygon()
            for mask in self
        ])
        return new

    def to_segmentation_list(self):
        """
        Converts all items to segmentation objects

        Returns:
            kwimage.SegmentationList
        """
        import kwimage
        new = kwimage.SegmentationList([
            None if item is None else kwimage.Segmentation.coerce(item)
            for item in self
        ])
        return new

    def to_mask_list(self, dims=None, pixels_are='points'):
        """
        returns this object

        Returns:
            kwimage.MaskList
        """
        return self


def _find_contours(binary_mask, pixels_are='points'):
    """
    Finds the contours in a binary mask

    Args:
        binary_mask (ndarray): a binary valued numpy array

        pixel_are (str):
            Can either be "points" or "areas".

            If pixels are "points", the we treat each pixel (i, j) as a
            single infinitely small point at (i, j). As such, some polygons
            may have zero area.

            If pixels are "areas", then each pixel (i, j) represents a
            square with coordinates ([i - 0.5, j - 0.5], [i + 0.5, j -
            0.5], [i + 0.5, j + 0.5], and [i - 0.5, j + 0.5]). Must have
            rasterio installed to use this method.

    Returns:
        List[Dict]: list of polygon exteriors and interiors
    """
    if pixels_are == 'points':
        # Note this mask needs to be uint8 not bool
        polys = _opencv_find_contours(binary_mask)
    elif pixels_are == 'areas':
        polys = _rasterio_find_contours(binary_mask)
    else:
        raise KeyError(pixels_are)
    return polys


def _rasterio_find_contours(binary_mask):
    from rasterio import features
    if binary_mask.size == 0:
        return []
    shapes = list(features.shapes(binary_mask, connectivity=8))
    translate = np.array([-0.5, -0.5]).ravel()[None, :]
    polys = []
    for shape, value in shapes:
        if value > 0:
            coords = shape['coordinates']
            exterior = np.array(coords[0]) + translate
            interiors = [np.array(p) + translate for p in coords[1:]]
            polys.append({
                'exterior': exterior,
                'interiors': interiors,
            })
    return polys


def _opencv_find_contours(binary_mask):
    import cv2
    p = 2
    offset = (0 - p, 0 - p)
    padded_mask = cv2.copyMakeBorder(binary_mask, p, p, p, p,
                                     cv2.BORDER_CONSTANT, value=0)

    # https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
    mode = cv2.RETR_CCOMP
    method = cv2.CHAIN_APPROX_SIMPLE
    # method = cv2.CHAIN_APPROX_TC89_KCOS
    # Different versions of cv2 have different return types
    _ret = cv2.findContours(padded_mask, mode, method, offset=offset)
    if len(_ret) == 2:
        _contours, _hierarchy = _ret
    else:
        _img, _contours, _hierarchy = _ret

    if _hierarchy is None:
        if len(_contours) == 0:
            return []
        raise AssertionError('Contour extraction from binary mask failed')

    _hierarchy = _hierarchy[0]

    polys = {i: {'exterior': None, 'interiors': []}
             for i, row in enumerate(_hierarchy) if row[3] == -1}
    for i, row in enumerate(_hierarchy):
        # This only works in RETR_CCOMP mode
        nxt, prev, child, parent = row[0:4]
        if parent != -1:
            coords = _contours[i][:, 0, :]
            polys[parent]['interiors'].append(coords)
        else:
            coords = _contours[i][:, 0, :]
            # if len(coords) < 3:
            #     raise Exception
            polys[i]['exterior'] = coords
    polys = list(polys.values())
    return polys


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/structs/mask.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
