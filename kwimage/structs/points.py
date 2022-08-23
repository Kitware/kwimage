"""
Data structures to represent and manipulate 2D Points
"""
import numpy as np
import ubelt as ub
import skimage
import kwarray
import numbers
import warnings
from kwimage.structs import _generic


class _PointsWarpMixin:

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Warps by applying an augmenter from the imgaug library

        Args:
            augmenter (imgaug.augmenters.Augmenter):
            input_dims (Tuple): h/w of the input image
            inplace (bool): if True, modifies data inplace

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from kwimage.structs.points import *  # NOQA
            >>> import imgaug
            >>> input_dims = (10, 10)
            >>> self = Points.random(10).scale(input_dims)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self._warp_imgaug(augmenter, input_dims)

            >>> self = Points(xy=(np.random.rand(10, 2) * 10).astype(int))
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self._warp_imgaug(augmenter, input_dims)

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 10)
            >>> ax.set_ylim(0, 10)
            >>> self.draw(color='red', alpha=.4, radius=0.1)
            >>> new.draw(color='blue', alpha=.4, radius=0.1)
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['xy'] = new.data['xy']._warp_imgaug(augmenter, input_dims,
                                                     inplace=inplace)
        if 'tf_data_to_img' in self.meta:
            # warping via imgaug invalidates the tf_data_to_img transform
            self.meta = self.meta.copy()
            self.meta.pop('tf_data_to_img')
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
        return self.data['xy'].to_imgaug(input_dims)

    @classmethod
    def from_imgaug(cls, kpoi):
        import kwimage
        data = kwimage.Coords.from_imgaug(kpoi)
        self = cls(data)
        return self

    @property
    def dtype(self):
        try:
            return self.data.dtype
        except Exception:
            print('kwimage.mask: no dtype for ' + str(type(self.data)))
            raise

    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (ArrayLike | Callable | kwimage.Affine | GeometricTransform | Augmenter):
                scikit-image tranform, a 3x3 transformation matrix,
                an imgaug Augmenter, or generic callable which transforms
                an NxD ndarray.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused, only exists for compatibility

            inplace (bool): if True, modifies data inplace

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
        import kwimage
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        if transform is None:
            return new

        if not isinstance(transform, (np.ndarray,
                                      skimage.transform._geometric.GeometricTransform,
                                      kwimage.Affine)):
            try:
                import imgaug
            except ImportError:
                pass
                # warnings.warn('imgaug is not installed')
                # raise TypeError(type(transform))
            else:
                if isinstance(transform, imgaug.augmenters.Augmenter):
                    return new._warp_imgaug(transform, input_dims, inplace=True)
            # else:
            #     raise TypeError(type(transform))
        new.data['xy'] = new.data['xy'].warp(transform, input_dims,
                                             output_dims, inplace)
        if 'tf_data_to_img' in new.meta:
            # if we are maintaining a transform to img space, we need to update it
            new.meta = new.meta.copy()
            tf = transform
            if isinstance(tf, np.ndarray):
                tf = skimage.transform.AffineTransform(matrix=transform)
            elif callable(tf):
                raise NotImplementedError(
                    'callables cant transform linear data_to_img yet')
            inv_tf = skimage.transform.AffineTransform(matrix=tf._inv_matrix)
            # new.meta['tf_data_to_img'] = new.meta['tf_data_to_img'] + inv_tf
            new.meta['tf_data_to_img'] = inv_tf + new.meta['tf_data_to_img']
        return new

    def scale(self, factor, output_dims=None, inplace=False):
        """
        Scale a points by a factor

        Args:
            factor (float | Tuple[float, float]):
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
        if 'tf_data_to_img' in new.meta:
            # if we are maintaining a transform to img space, we need to update it
            new.meta = new.meta.copy()
            tf = skimage.transform.AffineTransform(scale=factor)
            inv_tf = skimage.transform.AffineTransform(matrix=tf._inv_matrix)
            new.meta['tf_data_to_img'] = (inv_tf + new.meta['tf_data_to_img'])
        return new

    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the points

        Args:
            factor (float | Tuple[float, float]):
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
        if 'tf_data_to_img' in new.meta:
            # if we are maintaining a transform to img space, we need to update it
            new.meta = new.meta.copy()
            tf = skimage.transform.AffineTransform(translation=offset)
            inv_tf = skimage.transform.AffineTransform(matrix=tf._inv_matrix)
            new.meta['tf_data_to_img'] = (inv_tf + new.meta['tf_data_to_img'])
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
                if isinstance(data['xy'], _generic.ARRAY_TYPES):
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
            >>> print('self.data = {!r}'.format(self.data))
        """
        rng = kwarray.ensure_rng(rng)
        if ub.iterable(num):
            shape = tuple(num) + (2,)
        else:
            shape = (num, 2)
        self = Points(xy=rng.rand(*shape))
        self.data['visible'] = np.full(len(self), fill_value=2)
        if classes is not None:
            class_idxs = (rng.rand(len(self)) * len(classes)).astype(int)
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
            >>> # xdoctest: +REQUIRES(module:torch)
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

    def round(self, inplace=False):
        """
        Rounds data to the nearest integer

        Args:
            inplace (bool): if True, modifies this object

        Example:
            >>> import kwimage
            >>> self = kwimage.Points.random(3).scale(10)
            >>> self.round()
        """
        new = self if inplace else self.__class__(self.data, self.meta)
        new.data['xy'] = self.data['xy'].round()
        return new

    def numpy(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10)
            >>> self.tensor().numpy().tensor().numpy()
        """
        impl = self._impl
        newdata = {k: v.numpy() if hasattr(v, 'numpy') else impl.numpy(v)
                   for k, v in self.data.items()}
        new = self.__class__(newdata, self.meta)
        return new

    def draw_on(self, image=None, color='white', radius=None, copy=False):
        """

        Args:
            image (ndarray): image to draw points on.

            color (str | Any | List[Any]):
                one color for all boxes or a list of colors for each box
                Can be any type accepted by kwimage.Color.coerce.
                Extended types: str | ColorLike | List[ColorLike]

            radius (None | int):
                if an integer, an circle is drawn at each xy point with this
                radius.
                if None, attempts to fill a single point with subpixel accuracy,
                which generally means 4 pixels will be given some weight.
                Note: color can only be a single value for all points in this
                case.

            copy (bool): if True, force a copy of the image, otherwise
                try to draw inplace (may not work depending on dtype).

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
            >>> image = self.draw_on(image, radius=3, color='distinct')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> #self.draw(radius=3, alpha=.5, color='classes')
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwimage
            >>> s = 32
            >>> self = kwimage.Points.random(10).scale(s)
            >>> color = 'kitware_green'
            >>> # Test drawing on all channel + dtype combinations
            >>> im3 = np.zeros((s, s, 3), dtype=np.float32)
            >>> im_chans = {
            >>>     'im3': im3,
            >>>     'im1': kwimage.convert_colorspace(im3, 'rgb', 'gray'),
            >>>     'im4': kwimage.convert_colorspace(im3, 'rgb', 'rgba'),
            >>> }
            >>> inputs = {}
            >>> for k, im in im_chans.items():
            >>>     inputs[k + '_01'] = (kwimage.ensure_float01(im.copy()), {'radius': None})
            >>>     inputs[k + '_255'] = (kwimage.ensure_uint255(im.copy()), {'radius': None})
            >>> outputs = {}
            >>> for k, v in inputs.items():
            >>>     im, kw = v
            >>>     outputs[k] = self.draw_on(im, color=color, **kw)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=2, doclf=True)
            >>> plt = kwplot.autoplt()
            >>> pnum_ = kwplot.PlotNums(nRows=2, nSubplots=len(inputs))
            >>> for k in inputs.keys():
            >>>     kwplot.imshow(outputs[k], fnum=2, pnum=pnum_(), title=k)
            >>> plt.gcf().suptitle('Test draw points on channel + dtype combos')
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10).scale(32)
            >>> image = self.draw_on(radius=3, color='distinct')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(image)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # Test cases where single and multiple colors are given
            >>> # with radius=None and radius=scalar
            >>> from kwimage.structs.points import *  # NOQA
            >>> import kwimage
            >>> self = kwimage.Points.random(10).scale(32)
            >>> image1 = self.draw_on(radius=2, color='blue')
            >>> image2 = self.draw_on(radius=None, color='blue')
            >>> image3 = self.draw_on(radius=2, color='distinct')
            >>> image4 = self.draw_on(radius=None, color='distinct')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> canvas = kwimage.stack_images_grid(
            >>>     [image1, image2, image3, image4],
            >>>     pad=3, bg_value=(1, 1, 1))
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        if image is None:
            maxx, maxy = self.xy.max(axis=0)
            maxx = int(np.ceil(maxx) + 1)
            maxy = int(np.ceil(maxy) + 1)
            image = np.zeros((maxy, maxx, 3), dtype=np.float32)
        elif isinstance(image, tuple):
            # I forgot what the standard is that we use here...
            maxy, maxx = image
            image = np.zeros((maxx, maxy, 3), dtype=np.float32)

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        single_color = False

        if color == 'distinct':
            colors = [kwimage.Color(c) for c in kwimage.Color.distinct(len(self))]
        elif color == 'classes':
            # TODO: read colors from categories if they exist
            class_idxs = self.data['class_idxs']
            _keys, _vals = kwarray.group_indices(class_idxs)
            cls_colors = kwimage.Color.distinct(len(self.meta['classes']))
            colors = list(ub.take(cls_colors, class_idxs))
            colors = [kwimage.Color(c) for c in colors]
        else:
            num = len(self)
            if isinstance(color, list) and not isinstance(color, numbers.Number):
                # Passed list of color for each point
                colors = [kwimage.Color(c) for c in color]
            else:
                # Passed a single color
                single_color = True
                colors = [kwimage.Color(color)] * num

        if radius is None:
            image = kwimage.atleast_3channels(image)
            image = kwimage.ensure_float01(image, copy=copy)
            # value = kwimage.Color(color).as01()
            if single_color:
                color_value = np.array(colors[0]._forimage(image))
                image = self.data['xy'].fill(
                    image, color_value, coord_axes=[1, 0], interp='bilinear')
            else:
                # Need to loop when thare are multiple colors
                color_values = [np.array(kwimage.Color(c)._forimage(image))
                                for c in colors]
                xy_pts = self.data['xy'].data.reshape(-1, 2)
                for xy, color_ in zip(xy_pts, color_values):
                    image = kwimage.subpixel_setvalue(
                        image, xy[None, :], color_, coord_axes=[1, 0],
                        interp='bilinear')
        else:
            import cv2
            image = kwimage.atleast_3channels(image, copy=copy)
            # note: ellipse has a different return type (UMat) and does not
            # work inplace if the input is not contiguous.
            image = np.ascontiguousarray(image)

            xy_pts = self.data['xy'].data.reshape(-1, 2)
            color_values = [kwimage.Color(c)._forimage(image) for c in colors]

            for xy, color_ in zip(xy_pts, color_values):
                # center = tuple(map(int, xy.tolist()))
                center = tuple(xy.tolist())
                axes = (radius / 2, radius / 2)
                center = tuple(map(int, center))
                axes = tuple(map(int, axes))
                # print('center = {!r}'.format(center))
                # print('axes = {!r}'.format(axes))

                cv2.ellipse(image, center, axes, angle=0.0, startAngle=0.0,
                            endAngle=360.0, color=color_, thickness=-1)

        image = dtype_fixer(image, copy=False)
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1, setlim=False, **kwargs):
        """
        TODO: can use kwplot.draw_points

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.points import *  # NOQA
            >>> pts = Points.random(10)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=1)
            >>> pts.draw(radius=0.01)
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10, classes=['a', 'b', 'c'])
            >>> self.draw(radius=0.01, color='classes')
        """
        import kwimage
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()
        xy = self.data['xy'].data.reshape(-1, 2)
        # kwplot.draw_points(color=color, class_idxs)

        # More grouped patches == more efficient runtime
        if alpha is None:
            alpha = [1.0] * len(xy)
        elif not ub.iterable(alpha):
            alpha = [alpha] * len(xy)

        if color == 'distinct':
            colors = kwimage.Color.distinct(len(alpha))
        elif color == 'classes':
            # TODO: read colors from categories if they exist
            try:
                class_idxs = self.data['class_idxs']
                cls_colors = kwimage.Color.distinct(len(self.meta['classes']))
            except KeyError:
                raise Exception('cannot draw class colors without class_idxs and classes')
            _keys, _vals = kwarray.group_indices(class_idxs)
            colors = list(ub.take(cls_colors, class_idxs))
        else:
            colors = [color] * len(alpha)

        ptcolors = [kwimage.Color(c, alpha=a).as01('rgba')
                    for c, a in zip(colors, alpha)]
        color_groups = ub.group_items(range(len(ptcolors)), ptcolors)

        circlekw = {
            'radius': radius,
            'fill': True,
            'ec': None,
        }
        if 'fc' in kwargs:
            warnings.warning(
                'Warning: specifying fc to Points.draw overrides '
                'the color argument. Use color instead')
        circlekw.update(kwargs)
        fc = circlekw.pop('fc', None)  # hack

        collections = []
        for pcolor, idxs in color_groups.items():

            # hack for fc
            if fc is not None:
                pcolor = fc

            print(f'circlekw={circlekw}')
            print(f'pcolor={pcolor}')
            patches = [
                mpl.patches.Circle((x, y), fc=pcolor, **circlekw)
                for x, y in xy[idxs]
            ]
            col = mpl.collections.PatchCollection(patches, match_original=True)
            collections.append(col)
            ax.add_collection(col)

        if setlim:
            xmin = xy[:, 0].min()
            xmax = xy[:, 0].max()
            ymin = xy[:, 1].min()
            ymax = xy[:, 1].max()
            _generic._setlim(xmin, ymin, xmax, ymax, setlim=setlim, ax=ax)

        return collections

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

            >>> # xdoctest: +REQUIRES(module:torch)
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

            >>> # xdoctest: +REQUIRES(module:torch)
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

    def to_coco(self, style='orig'):
        """
        Converts to an mscoco-like representation

        Note:
            items that are usually id-references to other objects may need to
            be rectified.

        Args:
            style (str): either orig, new, new-id, or new-name

        Returns:
            Dict: mscoco-like representation

        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(4, classes=['a', 'b'])
            >>> orig = self._to_coco(style='orig')
            >>> print('orig = {!r}'.format(orig))
            >>> new_name = self._to_coco(style='new-name')
            >>> print('new_name = {}'.format(ub.repr2(new_name, nl=-1)))
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> import kwcoco
            >>> self.meta['classes'] = kwcoco.CategoryTree.coerce(self.meta['classes'])
            >>> new_id = self._to_coco(style='new-id')
            >>> print('new_id = {}'.format(ub.repr2(new_id, nl=-1)))
        """
        if self.xy.size == 0:
            return []
        if len(self.xy.shape) == 2:
            return self._to_coco(style=style)
        else:
            raise NotImplementedError('dim > 2, dense case todo')

    def _to_coco(self, style='orig'):
        """
        See to_coco
        """
        if style == 'orig':
            visible = self.data.get('visible', None)
            assert len(self.xy.shape) == 2
            if visible is None:
                visible = np.full((len(self), 1), fill_value=2)
            else:
                visible = visible.reshape(-1, 1)

            # TODO: ensure these are in the right order for the classes
            flat_pts = np.hstack([self.xy, visible]).reshape(-1)
            return flat_pts.tolist()
        elif style.startswith('new'):

            if style == 'new-id':
                use_id = True
            elif style == 'new-name':
                use_id = False
            elif style == 'new':
                use_id = False
            else:
                raise KeyError(style)

            new_kpts = []
            for i, xy in enumerate(self.data['xy'].data.tolist()):
                kpdict = {'xy': xy}
                if 'visible' in self.data:
                    kpdict['visible'] = int(self.data['visible'][i])
                if 'class_idxs' in self.data:
                    cidx = self.data['class_idxs'][i]
                    if use_id:
                        cid = self.meta['classes'].idx_to_id[cidx]
                        kpdict['keypoint_category_id'] = int(cid)
                    else:
                        cname = self.meta['classes'][cidx]
                        kpdict['keypoint_category'] = cname
                new_kpts.append(kpdict)
            return new_kpts
        else:
            raise KeyError(style)

    @classmethod
    def coerce(cls, data):
        """
        Attempt to coerce data into a Points object
        """
        if isinstance(data, cls):
            return data
        elif isinstance(data, (list, dict)):
            # TODO: determine if coco or geojson
            return cls.from_coco(data)
        elif isinstance(data, _generic.ARRAY_TYPES):
            return cls(data)
        else:
            raise TypeError(type(data))

    @classmethod
    def _from_coco(cls, coco_kpts, class_idxs=None, classes=None):
        # backwards compatibility
        return cls.from_coco(coco_kpts, class_idxs=class_idxs, classes=classes)

    @classmethod
    def from_coco(cls, coco_kpts, class_idxs=None, classes=None, warn=False):
        """
        Args:
            coco_kpts (list | dict): either the original list keypoint encoding
                or the new dict keypoint encoding.

            class_idxs (list): only needed if using old style

            classes (list | kwcoco.CategoryTree):
                list of all keypoint category names

            warn (bool): if True raise warnings

        Example:
            >>> ##
            >>> classes = ['mouth', 'left-hand', 'right-hand']
            >>> coco_kpts = [
            >>>     {'xy': (0, 0), 'visible': 2, 'keypoint_category': 'left-hand'},
            >>>     {'xy': (1, 2), 'visible': 2, 'keypoint_category': 'mouth'},
            >>> ]
            >>> Points.from_coco(coco_kpts, classes=classes)
            >>> # Test without classes
            >>> Points.from_coco(coco_kpts)
            >>> # Test without any category info
            >>> coco_kpts2 = [ub.dict_diff(d, {'keypoint_category'}) for d in coco_kpts]
            >>> Points.from_coco(coco_kpts2)
            >>> # Test without category instead of keypoint_category
            >>> coco_kpts3 = [ub.map_keys(lambda x: x.replace('keypoint_', ''), d) for d in coco_kpts]
            >>> Points.from_coco(coco_kpts3)
            >>> #
            >>> # Old style
            >>> coco_kpts = [0, 0, 2, 0, 1, 2]
            >>> Points.from_coco(coco_kpts)
            >>> # Fail case
            >>> coco_kpts4 = [{'xy': [4686.5, 1341.5], 'category': 'dot'}]
            >>> Points.from_coco(coco_kpts4, classes=[])

        Example:
            >>> # xdoctest: +REQUIRES(module:kwcoco)
            >>> import kwcoco
            >>> classes = kwcoco.CategoryTree.from_coco([
            >>>     {'name': 'mouth', 'id': 2}, {'name': 'left-hand', 'id': 3}, {'name': 'right-hand', 'id': 5}
            >>> ])
            >>> coco_kpts = [
            >>>     {'xy': (0, 0), 'visible': 2, 'keypoint_category_id': 5},
            >>>     {'xy': (1, 2), 'visible': 2, 'keypoint_category_id': 2},
            >>> ]
            >>> pts = Points.from_coco(coco_kpts, classes=classes)
            >>> assert pts.data['class_idxs'].tolist() == [2, 0]
        """
        if coco_kpts is None:
            return None

        if len(coco_kpts) and isinstance(ub.peek(coco_kpts), dict):
            # new style
            xy = []
            visible = []
            cidx_list = []

            if class_idxs is not None:
                if warn:
                    warnings.warn('class_idxs should not be specified for new-style')
                class_idxs = None

            # raise NotImplementedError(
            #     '''
            #     Needs to have extra information available to map
            #     between keypoint category ids and idxs.
            #     ''')

            if classes is None or not bool(classes):
                # See if we can infer the classes.
                # This may cause compatiblity issues.
                inferred_classes = [kpdict.get('keypoint_category',
                                               kpdict.get('category', None))
                                    for kpdict in coco_kpts]
                if all(inferred_classes):
                    if warn:
                        warnings.warn(
                            'Inferring keypoint classes in Points.from_coco. '
                            'It would be better to specify them explicitly')
                    classes = sorted(set(inferred_classes))

            for kpdict in coco_kpts:
                if classes is not None:
                    if 'keypoint_category_id' in kpdict:
                        cid = kpdict['keypoint_category_id']
                        try:
                            cidx = classes.id_to_idx[cid]
                        except AttributeError:
                            raise TypeError('classes needs to be a kwcoco.CategoryTree to parse keypoint_category_id')
                    elif 'keypoint_category' in kpdict:
                        assert classes is not None
                        cname = kpdict['keypoint_category']
                        cidx = classes.index(cname)
                    elif 'category_name' in kpdict:
                        assert classes is not None
                        cname = kpdict['category_name']
                        cidx = classes.index(cname)
                    ### Legacy support, these are not prefered names ###
                    elif 'category_id' in kpdict:
                        if warn:
                            warnings.warn('Keypoints got category_id, but we would prefer keypoint_category_id')
                        cid = kpdict['category_id']
                        try:
                            cidx = classes.id_to_idx[cid]
                        except AttributeError:
                            raise TypeError('classes needs to be a kwcoco.CategoryTree to parse keypoint_category_id')
                    elif 'category' in kpdict:
                        if warn:
                            warnings.warn('Keypoints got category, but we would prefer keypoint_category')
                        assert classes is not None
                        cname = kpdict['category']
                        cidx = classes.index(cname)
                    # else:
                    #     raise Exception('Keypoint category was not specified')
                    cidx_list.append(cidx)
                else:
                    if 'keypoint_category_id' in kpdict or 'keypoint_category' in kpdict:
                        # warnings.warn('classes should be specified for new-style')
                        raise Exception('classes should be specified for new-style')

                xy.append(kpdict['xy'])
                visible.append(kpdict.get('visible', 2))

            if cidx_list:
                assert len(cidx_list) == len(xy), 'missing category indices'
            else:
                cidx_list = None

            cidx_list = np.array(cidx_list)

            xy = np.array(xy)
            visible = np.array(visible)
            self = cls(xy=xy, visible=visible, class_idxs=cidx_list,
                       classes=classes)
        else:
            # original style
            kp = np.array(coco_kpts).reshape(-1, 3)
            xy = kp[:, 0:2]
            visible = kp[:, 2]
            if class_idxs is not None:
                if len(class_idxs) == 0:
                    if len(kp) > 0:
                        if warn:
                            warnings.warn('Creating keypoints with unknown class information')
                        # raise Exception('Creating keypoints with unknown class information')
                        class_idxs = [-1] * len(xy)
                    else:
                        class_idxs = []
                else:
                    assert len(class_idxs) == len(xy), '{} {}'.format(
                        len(class_idxs), len(xy))
            self = cls(xy=xy, visible=visible, class_idxs=class_idxs,
                       classes=classes)
        return self


class PointsList(_generic.ObjectList):
    """
    Stores a list of Points, each item usually corresponds to a different object.

    Note:
        # TODO: when the data is homogenous we can use a more efficient
        # representation, otherwise we have to use heterogenous storage.
    """
