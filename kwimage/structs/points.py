import numpy as np
import ubelt as ub
import skimage
import kwarray
import torch
from . import _generic


class _PointsWarpMixin:

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Warps by applying an augmenter from the imgaug library

        Args:
            augmenter (imgaug.augmenters.Augmenter):
            input_dims (Tuple): h/w of the input image
            inplace (bool, default=False): if True, modifies data inplace

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.points import *  # NOQA
            >>> import imgaug
            >>> input_dims = (10, 10)
            >>> self = Points.random(10).scale(input_dims)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self._warp_imgaug(augmenter, input_dims)

            >>> # xdoc: +REQUIRES(--show)
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> from matplotlib import pyplot as pl
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 10)
            >>> ax.set_ylim(0, 10)
            >>> self.draw(color='red', alpha=.4, radius=0.1)
            >>> new.draw(color='blue', alpha=.4, radius=0.1)
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        kpoi = new.to_imgaug(input_dims=input_dims)
        new_kpoi = augmenter.augment_keypoints(kpoi)
        dtype = new.data['xy'].data.dtype
        xy = np.array([[kp.x, kp.y] for kp in new_kpoi.keypoints], dtype=dtype)
        import kwimage
        new.data['xy'] = kwimage.Coords(xy)
        return new

    def to_imgaug(self, input_dims):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.points import *  # NOQA
            >>> pts = Points.random(10)
            >>> input_dims = (10, 10)
            >>> kpoi = pts.to_imgaug(input_dims)
        """
        import imgaug
        from distutils.version import LooseVersion
        if LooseVersion(imgaug.__version__) <= LooseVersion('0.2.9'):
            # Hack to fix imgaug bug
            h, w = input_dims
            input_dims = (h + 1.0, w + 1.0)
        else:
            # Note: the bug was in FlipLR._augment_keypoints, denoted by a todo
            # comment: "is this still correct with float keypoints?  Seems like
            # the -1 should be dropped"
            raise Exception('WAS THE BUG FIXED IN A NEW VERSION? '
                            'imgaug.__version__={}'.format(imgaug.__version__))
        kps = [imgaug.Keypoint(x, y) for x, y in self.data['xy'].data]
        kpoi = imgaug.KeypointsOnImage(kps, shape=input_dims)
        return kpoi

    @classmethod
    def from_imgaug(cls, kpoi):
        import numpy as np
        xy = np.array([[kp.x, kp.y] for kp in kpoi.keypoints])
        self = cls(xy=xy)
        return self

    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (GeometricTransform | ArrayLike | Augmenter):
                scikit-image tranform, a 3x3 transformation matrix, or
                an imgaug Augmenter.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused, only exists for compatibility

            inplace (bool, default=False): if True, modifies data inplace

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10, rng=0)
            >>> transform = skimage.transform.AffineTransform(scale=(2, 2))
            >>> new = self.warp(transform)
            >>> assert np.all(new.xy == self.scale(2).xy)

        Doctest:
            >>> self = Points.random(10, rng=0)
            >>> assert np.all(self.warp(np.eye(3)).xy == self.xy)
            >>> assert np.all(self.warp(np.eye(2)).xy == self.xy)
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        if not isinstance(transform, (np.ndarray, skimage.transform._geometric.GeometricTransform)):
            try:
                import imgaug
            except ImportError:
                import warnings
                warnings.warn('imgaug is not installed')
                raise TypeError(type(transform))
            if isinstance(transform, imgaug.augmenters.Augmenter):
                return new._warp_imgaug(transform, input_dims, inplace=True)
            else:
                raise TypeError(type(transform))
        new.data['xy'] = new.data['xy'].warp(transform, input_dims,
                                             output_dims, inplace)
        return new

    def scale(self, factor, output_dims=None, inplace=False):
        """
        Scale a points by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10, rng=0)
            >>> new = self.scale(10)
            >>> assert new.xy.max() <= 10
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['xy'] = new.data['xy'].scale(factor, output_dims=output_dims,
                                              inplace=inplace)
        return new

    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the points up/down left/right

        Args:
            factor (float or Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10, rng=0)
            >>> new = self.translate(10)
            >>> assert new.xy.min() >= 10
            >>> assert new.xy.max() <= 11
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['xy'] = new.data['xy'].translate(offset, output_dims, inplace)
        return new


class Points(_generic.Spatial, _PointsWarpMixin):
    """
    Stores multiple keypoints for a single object.

    This stores both the geometry and the class metadata if available

    Ignore:
        meta = {
         "names" = ['head', 'nose', 'tail'],
         "skeleton" = [(0, 1), (0, 2)],
        }

    Example:
        >>> from kwimage.structs.points import *  # NOQA
        >>> xy = np.random.rand(10, 2)
        >>> pts = Points(xy=xy)
        >>> print('pts = {!r}'.format(pts))
    """
    # __slots__ = ('data', 'meta',)

    # Pre-registered keys for the data dictionary
    __datakeys__ = ['xy', 'class_idxs', 'visible']
    # Pre-registered keys for the meta dictionary
    __metakeys__ = ['classes']

    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None,
                 **kwargs):
        if kwargs:
            if data or meta:
                raise ValueError('Cannot specify kwargs AND data/meta dicts')
            _datakeys = self.__datakeys__
            _metakeys = self.__metakeys__
            # Allow the user to specify custom data and meta keys
            if datakeys is not None:
                _datakeys = _datakeys + list(datakeys)
            if metakeys is not None:
                _metakeys = _metakeys + list(metakeys)
            # Perform input checks whenever kwargs is given
            data = {key: kwargs.pop(key) for key in _datakeys if key in kwargs}
            meta = {key: kwargs.pop(key) for key in _metakeys if key in kwargs}
            if kwargs:
                raise ValueError(
                    'Unknown kwargs: {}'.format(sorted(kwargs.keys())))

            if 'xy' in data:
                if isinstance(data['xy'], (np.ndarray, torch.Tensor)):
                    import kwimage
                    data['xy'] = kwimage.Coords(data['xy'])

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def __nice__(self):
        data_repr = repr(self.xy)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        return 'xy={}'.format(data_repr)

    __repr__ = ub.NiceRepr.__str__

    def __len__(self):
        return len(self.data['xy'])

    @property
    def shape(self):
        return self.data['xy'].shape

    @property
    def xy(self):
        return self.data['xy'].data

    @classmethod
    def random(Points, num=1, classes=None, rng=None):
        """
        Makes random points; typically for testing purposes

        Example:
            >>> import kwimage
            >>> self = kwimage.Points.random(classes=[1, 2, 3])
            >>> self.data
        """
        rng = kwarray.ensure_rng(rng)
        if ub.iterable(num):
            shape = tuple(num) + (2,)
        else:
            shape = (num, 2)
        self = Points(xy=rng.rand(*shape))
        if classes is not None:
            class_idxs = (rng.rand(len(self)) * len(classes)).astype(np.int)
            self.data['class_idxs'] = class_idxs
            self.meta['classes'] = classes
        return self

    def is_numpy(self):
        return self.data['xy'].is_numpy()

    def is_tensor(self):
        return self.data['xy'].is_tensor()

    @ub.memoize_property
    def _impl(self):
        return self.data['xy']._impl

    def tensor(self, device=ub.NoParam):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10)
            >>> self.tensor()
        """
        impl = self._impl
        newdata = {k: v.tensor(device) if hasattr(v, 'tensor')
                   else impl.tensor(v, device)
                   for k, v in self.data.items()}
        new = self.__class__(newdata, self.meta)
        return new

    def numpy(self):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10)
            >>> self.tensor().numpy().tensor().numpy()
        """
        impl = self._impl
        newdata = {k: v.numpy() if hasattr(v, 'numpy') else impl.numpy(v)
                   for k, v in self.data.items()}
        new = self.__class__(newdata, self.meta)
        return new

    def draw_on(self, image, color='white', radius=None):
        """
        CommandLine:
            xdoctest -m ~/code/kwimage/kwimage/structs/points.py Points.draw_on --show

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.points import *  # NOQA
            >>> s = 128
            >>> image = np.zeros((s, s))
            >>> self = Points.random(10).scale(s)
            >>> image = self.draw_on(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> self.draw(radius=3, alpha=.5)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.points import *  # NOQA
            >>> s = 128
            >>> image = np.zeros((s, s))
            >>> self = Points.random(10).scale(s)
            >>> image = self.draw_on(image, radius=3)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> self.draw(radius=3, alpha=.5)
            >>> kwplot.show_if_requested()
        """
        import kwplot
        import kwimage

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        if radius is None:
            image = kwimage.atleast_3channels(image)
            image = kwimage.ensure_float01(image)
            value = kwplot.Color(color).as01()
            image = self.data['xy'].fill(
                image, value, coord_axes=[1, 0], interp='bilinear')
        else:
            import cv2
            image = kwimage.atleast_3channels(image)

            if image.dtype.kind == 'f':
                value = kwplot.Color(color).as01()
            else:
                value = kwplot.Color(color).as255()
            # image = kwimage.ensure_float01(image)

            for xy in self.data['xy'].data.reshape(-1, 2):
                # center = tuple(map(int, xy.tolist()))
                center = tuple(xy.tolist())
                axes = (radius / 2, radius / 2)
                center = tuple(map(int, center))
                axes = tuple(map(int, axes))
                # print('center = {!r}'.format(center))
                # print('axes = {!r}'.format(axes))
                image = cv2.ellipse(image, center, axes, angle=0.0,
                                    startAngle=0.0, endAngle=360.0,
                                    color=value, thickness=-1)

        image = dtype_fixer(image)
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1):
        """
        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.points import *  # NOQA
            >>> pts = Points.random(10)
            >>> pts.draw(radius=0.01)
        """
        import kwplot
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()
        xy = self.data['xy'].data.reshape(-1, 2)

        # More grouped patches == more efficient runtime
        if alpha is None:
            alpha = [1.0] * len(xy)
        elif not ub.iterable(alpha):
            alpha = [alpha] * len(xy)

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
                for x, y in xy[idxs]
            ]
            col = mpl.collections.PatchCollection(patches, match_original=True)
            ax.add_collection(col)

    def compress(self, flags, axis=0, inplace=False):
        """
        Filters items based on a boolean criterion

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(4)
            >>> flags = [1, 0, 1, 1]
            >>> other = self.compress(flags)
            >>> assert len(self) == 4
            >>> assert len(other) == 3

            >>> other = self.tensor().compress(flags)
            >>> assert len(other) == 3
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        for k, v in self.data.items():
            try:
                new.data[k] = _generic._safe_compress(v, flags, axis=axis)
            except Exception:
                print('FAILED TO COMPRESS k={!r}, v={!r}'.format(k, v))
                raise
        return new

    def take(self, indices, axis=0, inplace=False):
        """
        Takes a subset of items at specific indices

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(4)
            >>> indices = [1, 3]
            >>> other = self.take(indices)
            >>> assert len(self) == 4
            >>> assert len(other) == 2

            >>> other = self.tensor().take(indices)
            >>> assert len(other) == 2
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        for k, v in self.data.items():
            new.data[k] = _generic._safe_take(v, indices, axis=axis)
        return new

    @classmethod
    def concatenate(cls, points, axis=0):
        if len(points) == 0:
            raise ValueError('need at least one box to concatenate')
        if axis != 0:
            raise ValueError('can only concatenate along axis=0')
        import kwimage
        first = points[0]
        datas = [p.data['xy'] for p in points]
        newxy = kwimage.Coords.concatenate(datas)
        new = cls({'xy': newxy}, first.meta)
        return new

    def _to_coco(self):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(4)
        """
        visible = self.data.get('visible', None)
        assert len(self.xy.shape) == 2
        if visible is None:
            visible = np.full((len(self), 1), fill_value=2)
        else:
            raise NotImplementedError

        # TODO: ensure these are in the right order for the classes
        flat_pts = np.hstack([self.xy, visible]).reshape(-1)
        return flat_pts

    def to_coco(self):
        if len(self.xy.shape) == 2:
            return self._to_coco()
        else:
            raise NotImplementedError('dim > 2, dense case todo')

    @classmethod
    def _from_coco(cls, coco_kpts):
        """
        """
        if coco_kpts is None:
            return None
        kp = np.array(coco_kpts).reshape(-1, 3)
        xy = kp[:, 0:2]
        visible = kp[:, 2]
        return cls(xy=xy, visible=visible)


class PointsList(_generic.ObjectList):
    """
    Stores a list of Points, each item usually corresponds to a different object.

    Notes:
        # TODO: when the data is homogenous we can use a more efficient
        # representation, otherwise we have to use heterogenous storage.
    """
