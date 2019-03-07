import numpy as np
import ubelt as ub
import skimage
import kwarray
from . import _generic


class _WarpMixin:

    def warp(self, transform, input_shape=None, output_shape=None, inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (GeometricTransform | ArrayLike | Augmenter):
                scikit-image tranform, a 3x3 transformation matrix, or
                an imgaug Augmenter.

            input_shape (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_shape (Tuple): unused, only exists for compatibility

            inplace (bool, default=False): if True, modifies data inplace

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
        """
        import kwimage
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        if isinstance(transform, np.ndarray):
            matrix = transform
        elif isinstance(transform, skimage.transform._geometric.GeometricTransform):
            matrix = transform.params
        else:
            raise TypeError(type(transform))
        new.data = kwimage.warp_points(matrix, new.data)
        return new

    def scale(self, factor, output_shape=None, inplace=False):
        """
        Scale coordinates by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or per-dimension tuple.
            output_shape (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, rng=0)
            >>> new = self.scale(10)
            >>> assert new.data.max() <= 10
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        data = new.data
        impl = kwarray.ArrayAPI.coerce(data)
        if not inplace:
            data = new.data = impl.copy(data)
        if impl.numel(data) > 0:
            ndims = self.ndims
            if not ub.iterable(factor):
                factor_ = impl.asarray([factor] * ndims)
            elif isinstance(factor, (list, tuple)):
                factor_ = np.array(factor)
            else:
                factor_ = factor
            assert factor_.shape == (ndims,)
            data *= factor_
        return new

    def translate(self, offset, output_shape=None, inplace=False):
        """
        Shift the coordinates up/down left/right

        Args:
            offset (float or Tuple[float]):
                transation offset as either a scalar or a (t_x, t_y) tuple.
            output_shape (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10, dim=3, rng=0)
            >>> new = self.translate(10)
            >>> assert new.data.min() >= 10
            >>> assert new.data.max() <= 11
            >>> Coords.random(3, dim=3, rng=0)
            >>> Coords.random(3, dim=3, rng=0).translate((1, 2, 3))
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        data = new.data
        impl = kwarray.ArrayAPI.coerce(data)
        if not inplace:
            data = new.data = impl.copy(data)
        if impl.numel(data) > 0:
            ndims = self.ndims
            if not ub.iterable(offset):
                offset_ = impl.asarray([offset] * ndims)
            elif isinstance(offset, (list, tuple)):
                offset_ = np.array(offset)
            else:
                offset_ = offset
            assert offset_.shape == (ndims,)
            data += offset_
        return new


class Coords(ub.NiceRepr, _WarpMixin):
    """
    This stores arbitrary sparse coordinate geometry.

    You can specify data, but you don't have to.
    We dont care what it is, we just warp it.
    The `fill` call lets you

    CommandLine:
        xdoctest -m kwimage.structs.coords Coords

    Example:
        >>> from kwimage.structs.coords import *  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> self = Coords.random(num=4, dim=3, rng=0)
        >>> matrix = rng.rand(4, 4)
        >>> self.warp(matrix)
        >>> self.translate(3, inplace=True)
        >>> self.translate(3, inplace=True)
        >>> self.scale(2)
        >>> self.tensor()
        >>> # self.tensor(device=0)
        >>> self.tensor().tensor().numpy().numpy()
        >>> self.numpy()
        >>> self.draw_on()
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
    def ndims(self):
        return self.data.shape[-1]

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

    @_generic.memoize_property
    def _impl(self):
        return kwarray.ArrayAPI.coerce(self.data)

    def tensor(self, device=ub.NoParam):
        newdata = self._impl.tensor(self.data, device)
        new = self.__class__(newdata, self.meta)
        return new

    def numpy(self):
        newdata = self._impl.numpy(self.data)
        new = self.__class__(newdata, self.meta)
        return new

    def draw_on(self, image=None, fill_value=1):
        """
        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> s = 256
            >>> self = Coords.random(10, meta={'shape': (s, s)}).scale(s)
            >>> image = np.zeros((s, s))
            >>> fill_value = 1
            >>> image = self.draw_on(image, fill_value)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> self.draw(radius=3)
        """
        # import kwimage
        if image is None:
            shape = tuple((self._impl.max(self.data, axis=0).astype(int) + 1).tolist())
            image = self._impl.zeros(self.meta.get('shape', shape))

        # TODO: do a better fill job
        maxdim = min(image.shape)
        index = tuple(self.data.astype(int).clip(0, maxdim - 1).T)
        flat_index = np.ravel_multi_index(index, image.shape)
        image.ravel()[flat_index] = fill_value

        # kwimage.subpixel_set(image, fill_value, index, interp_axes=None)
        # image[self.data] = fill_value
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1):
        """
        Example:
            >>> from kwimage.structs.coords import *  # NOQA
            >>> self = Coords.random(10)
            >>> self.draw(radius=3.0)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.draw(radius=3.0)
        """
        import kwplot
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()
        data = self.data

        if self.ndims != 2:
            raise NotImplementedError('need 2d for mpl')

        # More grouped patches == more efficient runtime
        if alpha is None:
            alpha = [1.0] * len(data)
        elif not ub.iterable(alpha):
            alpha = [alpha] * len(data)

        ptcolors = [kwplot.Color(color, alpha=a).as01('rgba') for a in alpha]
        color_groups = ub.group_items(range(len(ptcolors)), ptcolors)

        default_centerkw = {
            'radius': radius,
            'fill': True
        }
        centerkw = default_centerkw.copy()
        for pcolor, idxs in color_groups.items():
            patches = [
                mpl.patches.Circle((x, y), ec=None, fc=pcolor, **centerkw)
                for x, y in data[idxs]
            ]
            col = mpl.collections.PatchCollection(patches, match_original=True)
            ax.add_collection(col)
