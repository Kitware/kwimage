"""
Coordinates the fundamental "point" datatype. They do not contain metadata,
only geometry. See the `Points` data type for a structure that maintains
metadata on top of coordinate data.
"""
import numpy as np
import ubelt as ub
import skimage
import kwarray
from distutils.version import LooseVersion
from . import _generic

try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile


try:
    import imgaug
    _HAS_IMGAUG_FLIP_BUG = LooseVersion(imgaug.__version__) <= LooseVersion('0.2.9') and not hasattr(imgaug.augmenters.size, '_crop_and_pad_kpsoi')
    _HAS_IMGAUG_XY_ARRAY = LooseVersion(imgaug.__version__) >= LooseVersion('0.2.9')
except ImportError:
    _HAS_IMGAUG_FLIP_BUG = None
    _HAS_IMGAUG_XY_ARRAY = None


class Coords(_generic.Spatial, ub.NiceRepr):
    """
    A data structure to store n-dimensional coordinate geometry.

    Currently it is up to the user to maintain what coordinate system this
    geometry belongs to.

    NOTE:
        This class was designed to hold coordinates in r/c format, but in
        general this class is anostic to dimension ordering as long as you are
        consistent. However, there are two places where this matters:
            (1) drawing and (2) gdal/imgaug-warping. In these places we will
            assume x/y for legacy reasons. This may change in the future.

        The term axes with resepct to ``Coords`` always refers to the final
        numpy axis. In other words the final numpy-axis represents ALL of the
        coordinate-axes.

    CommandLine:
        xdoctest -m kwimage.structs.coords Coords

    Example:
        >>> from kwimage.structs.coords import *  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> self = Coords.random(num=4, dim=3, rng=rng)
        >>> print('self = {}'.format(self))
        self = <Coords(data=
            array([[0.5488135 , 0.71518937, 0.60276338],
                   [0.54488318, 0.4236548 , 0.64589411],
                   [0.43758721, 0.891773  , 0.96366276],
                   [0.38344152, 0.79172504, 0.52889492]]))>
        >>> matrix = rng.rand(4, 4)
        >>> self.warp(matrix)
        <Coords(data=
            array([[0.71037426, 1.25229659, 1.39498435],
                   [0.60799503, 1.26483447, 1.42073131],
                   [0.72106004, 1.39057144, 1.38757508],
                   [0.68384299, 1.23914654, 1.29258196]]))>
        >>> self.translate(3, inplace=True)
        <Coords(data=
            array([[3.5488135 , 3.71518937, 3.60276338],
                   [3.54488318, 3.4236548 , 3.64589411],
                   [3.43758721, 3.891773  , 3.96366276],
                   [3.38344152, 3.79172504, 3.52889492]]))>
        >>> self.translate(3, inplace=True)
        <Coords(data=
            array([[6.5488135 , 6.71518937, 6.60276338],
                   [6.54488318, 6.4236548 , 6.64589411],
                   [6.43758721, 6.891773  , 6.96366276],
                   [6.38344152, 6.79172504, 6.52889492]]))>
        >>> self.scale(2)
        <Coords(data=
            array([[13.09762701, 13.43037873, 13.20552675],
                   [13.08976637, 12.8473096 , 13.29178823],
                   [12.87517442, 13.783546  , 13.92732552],
                   [12.76688304, 13.58345008, 13.05778984]]))>
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> self.tensor()
        >>> self.tensor().tensor().numpy().numpy()
        >>> self.numpy()
        >>> #self.draw_on()
    """
    # __slots__ = ('data', 'meta',)  # turn on when no longer developing

    def __init__(self, data=None, meta=None):
        if isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def __nice__(self):
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        return 'data={}'.format(data_repr)

    __repr__ = ub.NiceRepr.__str__

    def __len__(self):
        return len(self.data)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def dim(self):
        return self.data.shape[-1]

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        newdata = self._impl.copy(self.data)
        newmeta = self.meta.copy()
        new = self.__class__(newdata, newmeta)
        return new

    @classmethod
    def random(Coords, num=1, dim=2, rng=None, meta=None):
        """
        Makes random coordinates; typically for testing purposes
        """
        rng = kwarray.ensure_rng(rng)
        self = Coords(data=rng.rand(num, dim), meta=meta)
        return self

    def is_numpy(self):
        return self._impl.is_numpy

    def is_tensor(self):
        return self._impl.is_tensor

    def compress(self, flags, axis=0, inplace=False):
        """
        Filters items based on a boolean criterion

        Args:
            flags (ArrayLike[bool]): true for items to be kept
            axis (int): you usually want this to be 0
            inplace (bool, default=False): if True, modifies this object

        Returns:
            Coords: filtered coords

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> self.compress([True] * len(self))
            >>> self.compress([False] * len(self))
            <Coords(data=array([], shape=(0, 2), dtype=float64))>
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = self.tensor()
            >>> self.compress([True] * len(self))
            >>> self.compress([False] * len(self))
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data = self._impl.compress(new.data, flags, axis=axis)
        return new

    def take(self, indices, axis=0, inplace=False):
        """
        Takes a subset of items at specific indices

        Args:
            indices (ArrayLike[int]): indexes of items to take
            axis (int): you usually want this to be 0
            inplace (bool, default=False): if True, modifies this object

        Returns:
            Coords: filtered coords

        Example:
            >>> self = Coords(np.array([[25, 30, 15, 10]]))
            >>> self.take([0])
            <Coords(data=array([[25, 30, 15, 10]]))>
            >>> self.take([])
            <Coords(data=array([], shape=(0, 4), dtype=int64))>
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data = self._impl.take(new.data, indices, axis=axis)
        return new

    def astype(self, dtype, inplace=False):
        """
        Changes the data type

        Args:
            dtype : new type
            inplace (bool, default=False): if True, modifies this object

        Returns:
            Coords: modified coordinates
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data = self._impl.astype(new.data, dtype, copy=not inplace)
        return new

    def round(self, inplace=False):
        """
        Rounds data to the nearest integer

        Args:
            inplace (bool, default=False): if True, modifies this object

        Example:
            >>> import kwimage
            >>> self = kwimage.Coords.random(3).scale(10)
            >>> self.round()
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data = self._impl.round(new.data)
        return new

    def view(self, *shape):
        """
        Passthrough method to view or reshape

        Args:
            *shape : new shape of the data

        Returns:
            Coords: modified coordinates

        Example:
            >>> self = Coords.random(6, dim=4).numpy()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Coords.random(6, dim=4).tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
        """
        data_ = self._impl.view(self.data, *shape)
        return self.__class__(data_, self.meta)

    @classmethod
    def concatenate(cls, coords, axis=0):
        """
        Concatenates lists of coordinates together

        Args:
            coords (Sequence[Coords]): list of coords to concatenate
            axis (int, default=0): axis to stack on

        Returns:
            Coords: stacked coords

        CommandLine:
            xdoctest -m kwimage.structs.coords Coords.concatenate

        Example:
            >>> coords = [Coords.random(3) for _ in range(3)]
            >>> new = Coords.concatenate(coords)
            >>> assert len(new) == 9
            >>> assert np.all(new.data[3:6] == coords[1].data)
        """
        if len(coords) == 0:
            raise ValueError('need at least one box to concatenate')
        if axis != 0:
            raise ValueError('can only concatenate along axis=0')
        first = coords[0]
        impl = first._impl
        datas = [b.data for b in coords]
        newdata = impl.cat(datas, axis=axis)
        new = cls(newdata)
        return new

    @property
    def device(self):
        """
        If the backend is torch returns the data device, otherwise None
        """
        try:
            return self.data.device
        except AttributeError:
            return None

    # @ub.memoize_property
    @property
    def _impl(self):
        """
        Returns the internal tensor/numpy ArrayAPI implementation
        """
        return kwarray.ArrayAPI.coerce(self.data)

    def tensor(self, device=ub.NoParam):
        """
        Converts numpy to tensors. Does not change memory if possible.

        Returns:
            Coords: modified coordinates

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Coords.random(3).numpy()
            >>> newself = self.tensor()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        newdata = self._impl.tensor(self.data, device)
        new = self.__class__(newdata, self.meta)
        return new

    def numpy(self):
        """
        Converts tensors to numpy. Does not change memory if possible.

        Returns:
            Coords: modified coordinates

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Coords.random(3).tensor()
            >>> newself = self.numpy()
            >>> self.data[0, 0] = 0
            >>> assert newself.data[0, 0] == 0
            >>> self.data[0, 0] = 1
            >>> assert self.data[0, 0] == 1
        """
        newdata = self._impl.numpy(self.data)
        new = self.__class__(newdata, self.meta)
        return new

    def reorder_axes(self, new_order, inplace=False):
        """
        Change the ordering of the coordinate axes.

        Args:
            new_order (Tuple[int]): ``new_order[i]`` should specify
                which axes in the original coordinates should be mapped to the
                ``i-th`` position in the returned axes.

            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Coords: modified coordinates

        Note:
            This is the ordering of the "columns" in final numpy axis, not the
            numpy axes themselves.

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords(data=np.array([
            >>>     [7, 11],
            >>>     [13, 17],
            >>>     [21, 23],
            >>> ]))
            >>> new = self.reorder_axes((1, 0))
            >>> print('new = {!r}'.format(new))
            new = <Coords(data=
                array([[11,  7],
                       [17, 13],
                       [23, 21]]))>

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> new = self.reorder_axes((1, 0))
            >>> # Remapping using 1, 0 reverses the axes
            >>> assert np.all(new.data[:, 0] == self.data[:, 1])
            >>> assert np.all(new.data[:, 1] == self.data[:, 0])
            >>> # Remapping using 0, 1 does nothing
            >>> eye = self.reorder_axes((0, 1))
            >>> assert np.all(eye.data == self.data)
            >>> # Remapping using 0, 0, destroys the 1-th column
            >>> bad = self.reorder_axes((0, 0))
            >>> assert np.all(bad.data[:, 0] == self.data[:, 0])
            >>> assert np.all(bad.data[:, 1] == self.data[:, 0])
        """
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)

        if True:
            # --- Method 1 - Slicing ---
            # This will use slicing tricks to avoid a copy operation, but the
            # data.flags will be modified and contiguous-ness is not preserved
            new.data = new.data[..., new_order]

        if False:
            # --- Method 2 - Overwrite ---
            # This will cause a copy operation, but the data.flags will remain
            # the same, i.e. contiguous arrays will remain contiguous.
            new.data[..., :] = new.data[..., new_order]

        if False:
            # Benchmark different methods, using slicing tricks seems
            # to have the best default behavior
            import timerit
            ti = timerit.Timerit(100, bestof=10, verbose=2)
            for timer in ti.reset('method2-apply'):
                new = self.copy()
                with timer:
                    new.data[..., :] = new.data[..., new_order]

            for timer in ti.reset('method1-apply'):
                new = self.copy()
                with timer:
                    new.data = new.data[..., new_order]

            for timer in ti.reset('method2-use'):
                new = self.copy()
                new.data[..., :] = new.data[..., new_order]
                with timer:
                    new.data += 10

            for timer in ti.reset('method1-use'):
                new = self.copy()
                new.data = new.data[..., new_order]
                with timer:
                    new.data += 10
        return new

    # @profile
    def warp(self, transform, input_dims=None, output_dims=None,
             inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (GeometricTransform | ArrayLike | Augmenter | callable):
                scikit-image tranform, a 3x3 transformation matrix,
                an imgaug Augmenter, or generic callable which transforms
                an NxD ndarray.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused in non-raster structures, only exists
                for compatibility.

            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Coords: modified coordinates

        Notes:
            Let D = self.dims

            transformation matrices can be either:
                * (D + 1) x (D + 1)  # for homog
                * D x D  # for scale / rotate
                * D x (D + 1)  # for affine

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> transform = skimage.transform.AffineTransform(scale=(2, 2))
            >>> new = self.warp(transform)
            >>> assert np.all(new.data == self.scale(2).data)

        Doctest:
            >>> self = Coords.random(10, rng=0)
            >>> assert np.all(self.warp(np.eye(3)).data == self.data)
            >>> assert np.all(self.warp(np.eye(2)).data == self.data)

        Doctest:
            >>> # xdoctest: +REQUIRES(module:osr)
            >>> import osr
            >>> wgs84_crs = osr.SpatialReference()
            >>> wgs84_crs.ImportFromEPSG(4326)
            >>> dst_crs = osr.SpatialReference()
            >>> dst_crs.ImportFromEPSG(2927)
            >>> transform = osr.CoordinateTransformation(wgs84_crs, dst_crs)
            >>> self = Coords.random(10, rng=0)
            >>> new = self.warp(transform)
            >>> assert np.all(new.data != self.data)

            >>> # Alternative using generic func
            >>> def _gdal_coord_tranform(pts):
            ...     return np.array([transform.TransformPoint(x, y, 0)[0:2]
            ...                      for x, y in pts])
            >>> alt = self.warp(_gdal_coord_tranform)
            >>> assert np.all(alt.data != self.data)
            >>> assert np.all(alt.data == new.data)

        Doctest:
            >>> # can use a generic function
            >>> def func(xy):
            ...     return np.zeros_like(xy)
            >>> self = Coords.random(10, rng=0)
            >>> assert np.all(self.warp(func).data == 0)
        """
        import kwimage
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)
        if isinstance(transform, _generic.ARRAY_TYPES):
            matrix = transform
        elif isinstance(transform, skimage.transform._geometric.GeometricTransform):
            matrix = transform.params
        else:

            ### Try to accept imgaug tranforms ###
            try:
                import imgaug
            except ImportError:
                import warnings
                warnings.warn('imgaug is not installed')
            else:
                if isinstance(transform, imgaug.augmenters.Augmenter):
                    return new._warp_imgaug(transform, input_dims, inplace=True)

            ### Try to accept GDAL tranforms ###
            try:
                import osr
            except ImportError:
                import warnings
                warnings.warn('gdal/osr is not installed')
            else:
                if isinstance(transform, osr.CoordinateTransformation):
                    # NOTE: We are expecting lon/lat here for wgs84
                    new_pts = []
                    for x, y in new.data:
                        x, y, z = transform.TransformPoint(x, y, 0)
                        if z != 0:
                            raise AssertionError('z = {}'.format(z))
                        new_pts.append((x, y))
                    new.data = np.array(new_pts, dtype=new.data.dtype)
                    return new

            ### Try to accept generic callable transforms ###
            if callable(transform):
                new.data = transform(new.data)
                return new

            raise TypeError(type(transform))
        new.data = kwimage.warp_points(matrix, new.data)
        return new

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Warps by applying an augmenter from the imgaug library

        NOTE:
            We are assuming you are using X/Y coordinates here.

        Args:
            augmenter (imgaug.augmenters.Augmenter):
            input_dims (Tuple): h/w of the input image
            inplace (bool, default=False): if True, modifies data inplace

        CommandLine:
            xdoctest -m ~/code/kwimage/kwimage/structs/coords.py Coords._warp_imgaug

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> import imgaug
            >>> input_dims = (10, 10)
            >>> self = Coords.random(10).scale(input_dims)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self._warp_imgaug(augmenter, input_dims)
            >>> # y coordinate should not change
            >>> assert np.allclose(self.data[:, 1], new.data[:, 1])
            >>> assert np.allclose(input_dims[0] - self.data[:, 0], new.data[:, 0])

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> from matplotlib import pyplot as pl
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, input_dims[0])
            >>> ax.set_ylim(0, input_dims[1])
            >>> self.draw(color='red', alpha=.4, radius=0.1)
            >>> new.draw(color='blue', alpha=.4, radius=0.1)

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> import imgaug
            >>> input_dims = (32, 32)
            >>> inplace = 0
            >>> self = Coords.random(1000, rng=142).scale(input_dims).scale(.8)
            >>> self.data = self.data.astype(np.int32).astype(np.float32)
            >>> augmenter = imgaug.augmenters.CropAndPad(px=(-4, 4), keep_size=1).to_deterministic()
            >>> new = self._warp_imgaug(augmenter, input_dims)
            >>> # Change should be linear
            >>> norm1 = (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0))
            >>> norm2 = (new.data - new.data.min(axis=0)) / (new.data.max(axis=0) - new.data.min(axis=0))
            >>> diff = norm1 - norm2
            >>> assert np.allclose(diff, 0, atol=1e-6, rtol=1e-4)
            >>> #assert np.allclose(self.data[:, 1], new.data[:, 1])
            >>> #assert np.allclose(input_dims[0] - self.data[:, 0], new.data[:, 0])
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwimage
            >>> im = kwimage.imresize(kwimage.grab_test_image(), dsize=input_dims[::-1])
            >>> new_im = augmenter.augment_image(im)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(im, pnum=(1, 2, 1), fnum=1)
            >>> self.draw(color='red', alpha=.8, radius=0.5)
            >>> kwplot.imshow(new_im, pnum=(1, 2, 2), fnum=1)
            >>> new.draw(color='blue', alpha=.8, radius=0.5, coord_axes=[1, 0])
        """
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)
        kpoi = new.to_imgaug(input_dims=input_dims)
        # print('kpoi = {!r}'.format(kpoi))
        new_kpoi = augmenter.augment_keypoints(kpoi)
        # print('new_kpoi = {!r}'.format(new_kpoi))
        dtype = new.data.dtype
        if hasattr(new_kpoi, 'to_xy_array'):
            # imgaug.__version__ >= 0.2.9
            xy = new_kpoi.to_xy_array().astype(dtype)
        else:
            xy = np.array([[kp.x, kp.y] for kp in new_kpoi.keypoints], dtype=dtype)
        new.data = xy
        return new

    # @profile
    def to_imgaug(self, input_dims):
        """
        Translate to an imgaug object

        Returns:
            imgaug.KeypointsOnImage: imgaug data structure

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10)
            >>> input_dims = (10, 10)
            >>> kpoi = self.to_imgaug(input_dims)
            >>> new = Coords.from_imgaug(kpoi)
            >>> assert np.allclose(new.data, self.data)
        """
        import imgaug
        if _HAS_IMGAUG_FLIP_BUG:
            # Hack to fix imgaug bug
            h, w = input_dims
            input_dims = (int(h + 1.0), int(w + 1.0))

        input_dims = tuple(map(int, input_dims))
        if _HAS_IMGAUG_XY_ARRAY:
            if hasattr(imgaug, 'Keypoints'):
                # make use of new proposal when/if it lands
                kps = imgaug.Keypoints(self.data)
                kpoi = imgaug.KeypointsOnImage(kps, shape=input_dims)
            else:
                kpoi = imgaug.KeypointsOnImage.from_xy_array(
                    self.data, shape=input_dims)
        else:
            kps = [imgaug.Keypoint(x, y) for x, y in self.data]
            kpoi = imgaug.KeypointsOnImage(kps, shape=input_dims)
        return kpoi

    @classmethod
    def from_imgaug(cls, kpoi):
        if _HAS_IMGAUG_XY_ARRAY:
            xy = kpoi.to_xy_array()
        else:
            xy = np.array([[kp.x, kp.y] for kp in kpoi.keypoints])
        self = cls(xy)
        return self

    @profile
    def scale(self, factor, about=None, output_dims=None, inplace=False):
        """
        Scale coordinates by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or per-dimension tuple.
            about (Tuple | None):
                if unspecified scales about the origin (0, 0), otherwise the
                rotation is about this point.
            output_dims (Tuple): unused in non-raster spatial structures
            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Coords: modified coordinates

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> new = self.scale(10)
            >>> assert new.data.max() <= 10

            >>> self = Coords.random(10, rng=0)
            >>> self.data = (self.data * 10).astype(int)
            >>> new = self.scale(10)
            >>> assert new.data.dtype.kind == 'i'
            >>> new = self.scale(10.0)
            >>> assert new.data.dtype.kind == 'f'
        """
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)
        data = new.data

        # if not inplace:
        #     data = new.data = impl.copy(data)
        if impl.numel(data) > 0:
            dim = self.dim
            if not ub.iterable(factor):
                factor_ = impl.asarray([factor] * dim)
            elif isinstance(factor, (list, tuple)):
                factor_ = impl.asarray(factor)
            else:
                factor_ = factor

            if self._impl.dtype_kind(data) != 'f' and self._impl.dtype_kind(factor_) == 'f':
                data = self._impl.astype(data, factor_.dtype)

            assert factor_.shape == (dim,)

            if about is None:
                data *= factor_
            else:
                about_ = self._rectify_about(about)
                data -= about_
                data *= factor_
                data += about_
        new.data = data
        return new

    @profile
    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the coordinates

        Args:
            offset (float or Tuple[float]):
                transation offset as either a scalar or a per-dimension tuple.
            output_dims (Tuple): unused in non-raster spatial structures
            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Coords: modified coordinates

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, dim=3, rng=0)
            >>> new = self.translate(10)
            >>> assert new.data.min() >= 10
            >>> assert new.data.max() <= 11
            >>> Coords.random(3, dim=3, rng=0)
            >>> Coords.random(3, dim=3, rng=0).translate((1, 2, 3))
        """
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)
        data = new.data
        if not inplace:
            data = new.data = impl.copy(data)
        if impl.numel(data) > 0:
            dim = self.dim
            if not ub.iterable(offset):
                offset_ = impl.asarray([offset] * dim)
            elif isinstance(offset, (list, tuple)):
                offset_ = np.array(offset)
            else:
                offset_ = offset
            assert offset_.shape == (dim,)
            offset_ = impl.astype(offset_, data.dtype)
            data += offset_
        return new

    @profile
    def rotate(self, theta, about=None, output_dims=None, inplace=False):
        """
        Rotate the coordinates about a point.

        Args:
            theta (float):
                rotation angle in radians

            about (Tuple | None):
                if unspecified rotates about the origin (0, 0), otherwise the
                rotation is about this point.

            output_dims (Tuple): unused in non-raster spatial structures

            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Coords: modified coordinates

        TODO:
            - [ ] Generalized ND Rotations?

        References:
            https://math.stackexchange.com/questions/197772/gen-rot-matrix

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, dim=2, rng=0)
            >>> theta = np.pi / 2
            >>> new = self.rotate(theta)

            >>> # Test rotate agrees with warp
            >>> sin_ = np.sin(theta)
            >>> cos_ = np.cos(theta)
            >>> rot_ = np.array([[cos_, -sin_], [sin_,  cos_]])
            >>> new2 = self.warp(rot_)
            >>> assert np.allclose(new.data, new2.data)

            >>> #
            >>> # Rotate about a custom point
            >>> theta = np.pi / 2
            >>> new3 = self.rotate(theta, about=(0.5, 0.5))
            >>> #
            >>> # Rotate about the center of mass
            >>> about = self.data.mean(axis=0)
            >>> new4 = self.rotate(theta, about=about)
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> plt = kwplot.autoplt()
            >>> self.draw(radius=0.01, color='blue', alpha=.5, coord_axes=[1, 0], setlim='grow')
            >>> plt.gca().set_aspect('equal')
            >>> new3.draw(radius=0.01, color='red', alpha=.5, coord_axes=[1, 0], setlim='grow')
        """
        if self.dim != 2:
            raise NotImplementedError('only 2D rotations for now')

        dtype = self.dtype
        if isinstance(about, str):
            raise NotImplementedError(about)

        if about is None:
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            rot_ = np.array([[cos_, -sin_],
                             [sin_,  cos_]], dtype=dtype)
        else:
            about_ = self._rectify_about(about)
            """
            # Construct a general closed-form affine matrix about a point
            # Shows the symbolic construction of the code
            # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
            import sympy
            sx, sy, theta, shear_y, shear_x, tx, ty, x0, y0 = sympy.symbols(
                'sx, sy, theta, shear_y, shear_x, tx, ty, x0, y0')

            # Construct an general origin centered affine matrix
            sin_ = sympy.sin(theta)
            cos_ = sympy.cos(theta)
            R = np.array([[cos_, -sin_,  0],
                          [sin_,  cos_,  0],
                          [   0,     0,  1]])
            H = np.array([[      1, shear_x, 0],
                          [shear_y,       1, 0],
                          [      0,       0, 1]])
            S = np.array([[sx,  0, 0],
                          [ 0, sy, 0],
                          [ 0,  0, 1]])
            T = np.array([[1, 0, tx],
                          [0, 1, ty],
                          [0, 0,  1]])

            # combine simple transformations into an affine transform
            Aff_0 = sympy.Matrix(T @ S @ R @ H)
            Aff_0 = sympy.simplify(Aff_0)
            print(ub.hzcat(['Aff_0 = ', repr(Aff_0)]))

            # move to center xy0, apply affine transform, then move back
            tr1 = np.array([[1, 0, -x0],
                            [0, 1, -y0],
                            [0, 0,   1]])
            tr2 = np.array([[1, 0, x0],
                            [0, 1, y0],
                            [0, 0,  1]])
            AffAbout = tr2 @ Aff_0 @ tr1
            AffAbout = sympy.simplify(AffAbout)
            print(ub.hzcat(['AffAbout = ', repr(AffAbout)]))

            # Get the special case for rotation about
            print(repr(AffAbout.subs(dict(shear_x=0, shear_y=0, sx=1, sy=1, tx=0, ty=0))))
            """
            x0, y0 = about_
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            rot_ = np.array([
                [ cos_, -sin_, -x0 * cos_ + y0 * sin_ + x0],
                [ sin_,  cos_, -x0 * sin_ - y0 * cos_ + y0],
                [    0,     0,                           1]])
        return self.warp(rot_, output_dims=output_dims, inplace=inplace)

    def _rectify_about(self, about):
        """
        Ensures that about returns a specified point. Allows for special keys
        like center to be used.

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, dim=2, rng=0)
        """
        if about is None:
            about_ = None
        else:
            if isinstance(about, str):
                if about == 'origin':
                    about_ = (0, 0)
                else:
                    raise KeyError(about)
            else:
                about_ = about if ub.iterable(about) else [about] * self.dim
        return about_

    def fill(self, image, value, coord_axes=None, interp='bilinear'):
        """
        Sets sub-coordinate locations in a grid to a particular value

        Args:
            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images, if you are storing r/c data,
                set to [0,1], if you are storing x/y data, set to [1,0].

        Returns:
            ndarray: image with coordinates rasterized on it
        """
        import kwimage
        index = self.data
        image = kwimage.subpixel_setvalue(image, index, value,
                                          coord_axes=coord_axes, interp=interp)
        return image

    def soft_fill(self, image, coord_axes=None, radius=5):
        """
        Used for drawing keypoint truth in heatmaps

        Args:
            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images, if you are storing r/c data,
                set to [0,1], if you are storing x/y data, set to [1,0].

                In other words the i-th entry in coord_axes specifies which
                row-major spatial dimension the i-th column of a coordinate
                corresponds to. The index is the coordinate dimension and the
                value is the axes dimension.

        Returns:
            ndarray: image with coordinates rasterized on it

        References:
            https://stackoverflow.com/questions/54726703/generating-keypoint-heatmaps-in-tensorflow

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> s = 64
            >>> self = Coords.random(10, meta={'shape': (s, s)}).scale(s)
            >>> # Put points on edges to to verify "edge cases"
            >>> self.data[1] = [0, 0]       # top left
            >>> self.data[2] = [s, s]       # bottom right
            >>> self.data[3] = [0, s + 10]  # bottom left
            >>> self.data[4] = [-3, s // 2] # middle left
            >>> self.data[5] = [s + 1, -1]  # top right
            >>> # Put points in the middle to verify overlap blending
            >>> self.data[6] = [32.5, 32.5] # middle
            >>> self.data[7] = [34.5, 34.5] # middle
            >>> fill_value = 1
            >>> coord_axes = [1, 0]
            >>> radius = 10
            >>> image1 = np.zeros((s, s))
            >>> self.soft_fill(image1, coord_axes=coord_axes, radius=radius)
            >>> radius = 3.0
            >>> image2 = np.zeros((s, s))
            >>> self.soft_fill(image2, coord_axes=coord_axes, radius=radius)
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(image1, pnum=(1, 2, 1))
            >>> kwplot.imshow(image2, pnum=(1, 2, 2))
        """
        import scipy.stats

        if radius <= 0:
            raise ValueError('radius must be positive')

        # OH! How I HATE the squeeze function!
        SCIPY_STILL_USING_SQUEEZE_FUNC = True

        blend_mode = 'maximum'

        image_ndims = len(image.shape)

        for pt in self.data:

            # Find a grid of coordinates on the image to fill for this point
            low = np.floor(pt - radius).astype(int)
            high = np.ceil(pt + radius).astype(int)
            grid = np.dstack(np.mgrid[tuple(
                slice(s, t) for s, t in zip(low, high))])

            # Flatten the grid into a list of coordinates to be filled
            rows_of_coords = grid.reshape(-1, grid.shape[-1])

            # Remove grid coordinates that are out of bounds
            lower_bound = np.array([0, 0])
            upper_bound = np.array([
                image.shape[i] for i in coord_axes
            ])[None, :]
            in_bounds_flags1 = (rows_of_coords >= lower_bound).all(axis=1)
            rows_of_coords = rows_of_coords[in_bounds_flags1]
            in_bounds_flags2 = (rows_of_coords < upper_bound).all(axis=1)
            rows_of_coords = rows_of_coords[in_bounds_flags2]

            if len(rows_of_coords) > 0:
                # Create a index into the image and insert the columns of
                # coordinates to fill into the appropirate dimensions
                img_index = [slice(None)] * image_ndims
                for axes_idx, coord_col in zip(coord_axes, rows_of_coords.T):
                    img_index[axes_idx] = coord_col
                img_index = tuple(img_index)

                # Note: Do we just use kwimage.gaussian_patch for the 2D case
                # instead?
                # TODO: is there a better method for making a "brush stroke"?
                # cov = 0.3 * ((extent - 1) * 0.5 - 1) + 0.8
                cov = radius
                rv = scipy.stats.multivariate_normal(mean=pt, cov=cov)
                new_values = rv.pdf(rows_of_coords)

                # the mean will be the maximum values of the normal
                # distribution, normalize by that.
                max_val = float(rv.pdf(pt))

                if SCIPY_STILL_USING_SQUEEZE_FUNC:
                    # If multivariate_normal was implemented right we would not
                    # need to check for scalar values
                    # See: https://github.com/scipy/scipy/issues/7689
                    if len(rows_of_coords) == 1:
                        if len(new_values.shape) != 0:
                            import warnings
                            warnings.warn(ub.paragraph(
                                '''
                                Scipy fixed the bug in multivariate_normal!
                                We can remove this stupid hack!
                                '''))
                        else:
                            # Ensure new_values is always a list of scalars
                            new_values = new_values[None]

                new_values = new_values / max_val

                # Blend the sampled values onto the existing pixels
                prev_values = image[img_index]

                # HACK: wont generalize?
                if len(prev_values.shape) != len(new_values.shape):
                    new_values = new_values[:, None]

                if blend_mode == 'maximum':
                    blended = np.maximum(prev_values, new_values)
                else:
                    raise KeyError(blend_mode)

                # Draw the blended pixels inplace
                image[img_index] = blended

        return image

    def draw_on(self, image=None, fill_value=1, coord_axes=[1, 0],
                interp='bilinear'):
        """
        Note:
            unlike other methods, the defaults assume x/y internal data

        Args:
            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images, if you are storing r/c data,
                set to [0,1], if you are storing x/y data, set to [1,0].

                In other words the i-th entry in coord_axes specifies which
                row-major spatial dimension the i-th column of a coordinate
                corresponds to. The index is the coordinate dimension and the
                value is the axes dimension.

        Returns:
            ndarray: image with coordinates drawn on it

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> s = 256
            >>> self = Coords.random(10, meta={'shape': (s, s)}).scale(s)
            >>> self.data[0] = [10, 10]
            >>> self.data[1] = [20, 40]
            >>> image = np.zeros((s, s))
            >>> fill_value = 1
            >>> image = self.draw_on(image, fill_value, coord_axes=[1, 0], interp='bilinear')
            >>> # image = self.draw_on(image, fill_value, coord_axes=[0, 1], interp='nearest')
            >>> # image = self.draw_on(image, fill_value, coord_axes=[1, 0], interp='bilinear')
            >>> # image = self.draw_on(image, fill_value, coord_axes=[1, 0], interp='nearest')
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(image)
            >>> self.draw(radius=3, alpha=.5, coord_axes=[1, 0])
        """
        # import kwimage
        if image is None:
            shape_ = self._impl.max(self.data, axis=0).astype(int)
            shape = tuple((shape_ + 1).tolist())
            image = self._impl.zeros(self.meta.get('shape', shape))
        image = self.fill(image, fill_value, coord_axes=coord_axes,
                          interp=interp)
        return image

    def draw(self, color='blue', ax=None, alpha=None, coord_axes=[1, 0],
             radius=1, setlim=False):
        """
        Note:
            unlike other methods, the defaults assume x/y internal data

        Args:
            setlim (bool): if True ensures the limits of the axes contains the
                polygon

            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images,
                    if you are storing r/c data, set to [0,1],
                    if you are storing x/y data, set to [1,0].

        Returns:
            List[mpl.collections.PatchCollection]: drawn matplotlib objects

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(radius=3.0, setlim=True)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.draw(radius=3.0)
        """
        import matplotlib as mpl
        import kwimage
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()
        data = self.data

        if self.dim != 2:
            raise NotImplementedError('need 2d for mpl')

        # More grouped patches == more efficient runtime
        if alpha is None:
            alpha = [1.0] * len(data)
        elif not ub.iterable(alpha):
            alpha = [alpha] * len(data)

        ptcolors = [kwimage.Color(color, alpha=a).as01('rgba') for a in alpha]
        color_groups = ub.group_items(range(len(ptcolors)), ptcolors)

        default_centerkw = {
            'radius': radius,
            'fill': True
        }
        centerkw = default_centerkw.copy()
        collections = []
        for pcolor, idxs in color_groups.items():
            yx_list = [row[coord_axes] for row in data[idxs]]
            patches = [
                mpl.patches.Circle((x, y), ec=None, fc=pcolor, **centerkw)
                for y, x in yx_list
            ]
            col = mpl.collections.PatchCollection(patches, match_original=True)
            collections.append(col)
            ax.add_collection(col)

        if setlim:
            x1, y1 = self.data.min(axis=0)
            x2, y2 = self.data.max(axis=0)

            if setlim == 'grow':
                # only allow growth
                x1_, x2_ = ax.get_xlim()
                y1_, y2_ = ax.get_ylim()
                x1 = min(x1_, x1)
                x2 = max(x2_, x2)
                y1 = min(y1_, y1)
                y2 = max(y2_, y2)

            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
        return collections

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwimage.structs.coords all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
