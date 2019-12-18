"""
Coordinates the fundamental "point" datatype. They do not contain metadata,
only geometry. See the `Points` data type for a structure that maintains
metadata on top of coordinate data.
"""
import numpy as np
import ubelt as ub
import skimage
import torch
import kwarray
from distutils.version import LooseVersion
from . import _generic


try:
    import imgaug
    _HAS_IMGAUG_FLIP_BUG = LooseVersion(imgaug.__version__) <= LooseVersion('0.2.9') and not hasattr(imgaug.augmenters.size, '_crop_and_pad_kpsoi')
    _HAS_IMGAUG_XY_ARRAY = LooseVersion(imgaug.__version__) >= LooseVersion('0.2.9')
except ImportError:
    _HAS_IMGAUG_FLIP_BUG = None
    _HAS_IMGAUG_XY_ARRAY = None


class Coords(_generic.Spatial, ub.NiceRepr):
    """
    This stores arbitrary sparse n-dimensional coordinate geometry.

    You can specify data, but you don't have to.
    We dont care what it is, we just warp it.

    NOTE:
        This class was designed to hold coordinates in r/c format, but in
        general this class is anostic to dimension ordering as long as you are
        consistent. However, there are two places where this matters:
            (1) drawing and (2) gdal/imgaug-warping. In these places we will
            assume x/y for legacy reasons. This may change in the future.

    CommandLine:
        xdoctest -m kwimage.structs.coords Coords

    Example:
        >>> from kwimage.structs.coords import *  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> self = Coords.random(num=4, dim=3, rng=rng)
        >>> matrix = rng.rand(4, 4)
        >>> self.warp(matrix)
        >>> self.translate(3, inplace=True)
        >>> self.translate(3, inplace=True)
        >>> self.scale(2)
        >>> self.tensor()
        >>> # self.tensor(device=0)
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
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data = self._impl.astype(new.data, dtype, copy=not inplace)
        return new

    def view(self, *shape):
        """
        Passthrough method to view or reshape

        Args:
            *shape : new shape of the data

        Example:
            >>> self = Coords.random(6, dim=4).tensor()
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
            >>> self = Coords.random(6, dim=4).numpy()
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

        Example:
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

        Example:
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

    # @profile
    def warp(self, transform, input_dims=None, output_dims=None,
             inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (GeometricTransform | ArrayLike | Augmenter):
                scikit-image tranform, a transformation matrix, or
                an imgaug Augmenter.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused in non-raster structures, only exists
                for compatibility.

            inplace (bool, default=False): if True, modifies data inplace

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

        Ignore:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(module:osr)
            >>> wgs84_crs = osr.SpatialReference()
            >>> wgs84_crs.ImportFromEPSG(4326)
            >>> transform = osr.CoordinateTransformation(wgs84_crs, wgs84_crs)
            >>> self = Coords.random(10, rng=0)
            >>> new = self.warp(transform)
            >>> assert np.all(new.data == self.data)
        """
        import kwimage
        impl = self._impl
        new = self if inplace else self.__class__(impl.copy(self.data), self.meta)
        if isinstance(transform, (np.ndarray, torch.Tensor)):
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
                raise TypeError(type(transform))
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
                    new_pts = []
                    for x, y in new.data:
                        x, y, z = transform.TransformPoint(x, y, 0)
                        assert z == 0
                        new_pts.append((x, y))
                    new.data = np.array(new_pts, dtype=new.data.dtype)
                    return new
            raise TypeError(type(transform))
        new.data = kwimage.warp_points(matrix, new.data)
        return new

    # @profile
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

    def scale(self, factor, output_dims=None, inplace=False):
        """
        Scale coordinates by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or per-dimension tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> new = self.scale(10)
            >>> assert new.data.max() <= 10

            >>> self = Coords.random(10, rng=0)
            >>> self.data = (self.data * 10).astype(np.int)
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
            data *= factor_
        new.data = data
        return new

    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the coordinates

        Args:
            offset (float or Tuple[float]):
                transation offset as either a scalar or a per-dimension tuple.
            output_dims (Tuple): unused in non-raster spatial structures

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

    def fill(self, image, value, coord_axes=None, interp='bilinear'):
        """
        Sets sub-coordinate locations in a grid to a particular value

        Args:
            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images, if you are storing r/c data,
                set to [0,1], if you are storing x/y data, set to [1,0].
        """
        import kwimage
        index = self.data
        image = kwimage.subpixel_setvalue(image, index, value,
                                          coord_axes=coord_axes, interp=interp)
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
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
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
             radius=1):
        """
        Note:
            unlike other methods, the defaults assume x/y internal data

        Args:
            coord_axes (Tuple): specify which image axes each coordinate dim
                corresponds to.  For 2D images,
                    if you are storing r/c data, set to [0,1],
                    if you are storing x/y data, set to [1,0].

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(radius=3.0)
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
        return collections
