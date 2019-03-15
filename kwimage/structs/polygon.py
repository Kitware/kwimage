import ubelt as ub
import cv2
import numpy as np
from . import _generic
import torch


class _PolyArrayBackend:
    def is_numpy(self):
        return self._impl.is_numpy

    def is_tensor(self):
        return self._impl.is_tensor

    def tensor(self, device=ub.NoParam):
        """
        Example:
            >>> from kwimage.structs.polygon import *
            >>> self = Polygon.random()
            >>> self.tensor()
        """
        impl = self._impl
        newdata = {k: v.tensor(device) if hasattr(v, 'tensor')
                   else impl.tensor(v, device)
                   for k, v in self.data.items()}
        new = self.__class__(newdata)
        return new

    def numpy(self):
        """
        Example:
            >>> from kwimage.structs.polygon import *
            >>> self = Polygon.random()
            >>> self.tensor().numpy().tensor().numpy()
        """
        impl = self._impl
        newdata = {k: v.numpy() if hasattr(v, 'numpy') else impl.numpy(v)
                   for k, v in self.data.items()}
        new = self.__class__(newdata)
        return new


class _PolyWarpMixin:

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Warps by applying an augmenter from the imgaug library

        Args:
            augmenter (imgaug.augmenters.Augmenter):
            input_dims (Tuple): h/w of the input image
            inplace (bool, default=False): if True, modifies data inplace

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> import imgaug
            >>> input_dims = np.array((10, 10))
            >>> self = Polygon.random(10, n_holes=1, rng=0).scale(input_dims)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self._warp_imgaug(augmenter, input_dims)

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> from matplotlib import pyplot as pl
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 10)
            >>> ax.set_ylim(0, 10)
            >>> self.draw(color='red', alpha=.4)
            >>> new.draw(color='blue', alpha=.4)
        """
        new = self if inplace else self.__class__(self.data.copy())

        # current version of imgaug doesnt fully support polygons
        # coerce to and from points instead
        dtype = self.data['exterior'].data.dtype

        import imgaug
        parts = []
        kps = [imgaug.Keypoint(x, y) for x, y in self.data['exterior'].data]
        parts.append(kps)
        for hole in self.data.get('interiors', []):
            kps = [imgaug.Keypoint(x, y) for x, y in hole.data]
            parts.append(kps)

        cs = [0] + np.cumsum(np.array(list(map(len, parts)))).tolist()
        flat_kps = list(ub.flatten(parts))

        if imgaug.__version__ == '0.2.8':
            # Hack to fix imgaug bug
            h, w = input_dims
            input_dims = (h + 1.0, w + 1.0)
        else:
            raise Exception('WAS THE BUG FIXED IN A NEW VERSION?')
        kpoi = imgaug.KeypointsOnImage(flat_kps, shape=tuple(input_dims))

        kpoi = augmenter.augment_keypoints(kpoi)

        _new_parts = []
        for a, b in ub.iter_window(cs, 2):
            unpacked = [[kp.x, kp.y] for kp in kpoi.keypoints[a:b]]
            new_part = np.array(unpacked, dtype=dtype)
            _new_parts.append(new_part)
        new_parts = _new_parts[::-1]

        import kwimage
        new_exterior = kwimage.Coords(new_parts[0])
        new_interiors = [kwimage.Coords(p) for p in new_parts[1:]]
        new.data['exterior'] = new_exterior
        new.data['interiors'] = new_interiors
        return new

    def to_imgaug(self, shape):
        import imgaug
        ia_exterior = imgaug.Polygon(self.data['exterior'])
        ia_interiors = [imgaug.Polygon(p) for p in self.data.get('interiors', [])]
        iamp = imgaug.MultiPolygon([ia_exterior] + ia_interiors)
        return iamp

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
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random()
            >>> transform = skimage.transform.AffineTransform(scale=(2, 2))
            >>> new = self.warp(transform)
            >>> assert np.all(new.xy == self.scale(2).xy)

        Doctest:
            >>> self = Polygon.random()
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self.warp(augmenter, input_dims=(1, 1))
            >>> print('new = {!r}'.format(new.data))
            >>> print('self = {!r}'.format(self.data))
            >>> #assert np.all(self.warp(np.eye(3)).exterior == self.exterior)
            >>> #assert np.all(self.warp(np.eye(2)).exterior == self.exterior)
        """
        new = self if inplace else self.__class__(self.data.copy())
        import skimage
        if not isinstance(transform, (np.ndarray, skimage.transform._geometric.GeometricTransform)):
            import imgaug
            if isinstance(transform, imgaug.augmenters.Augmenter):
                return new._warp_imgaug(transform, input_dims, inplace=True)
            else:
                raise TypeError(type(transform))
        new.data['exterior'] = new.data['exterior'].warp(transform, input_dims,
                                                         output_dims, inplace)
        new.data['interiors'] = [
            p.warp(transform, input_dims, output_dims, inplace)
            for p in new.data['interiors']
        ]
        return new

    def scale(self, factor, output_dims=None, inplace=False):
        """
        Scale a polygon by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> new = self.scale(10)
            >>> assert new.xy.max() <= 10
        """
        new = self if inplace else self.__class__(self.data.copy())
        new.data['exterior'] = new.data['exterior'].scale(factor, output_dims, inplace)
        new.data['interiors'] = [p.scale(factor, output_dims, inplace) for p in new.data['interiors']]
        return new

    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the polygon up/down left/right

        Args:
            factor (float or Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> new = self.translate(10)
            >>> assert new.xy.min() >= 10
            >>> assert new.xy.max() <= 11
        """
        new = self if inplace else self.__class__(self.data.copy())
        new.data['exterior'] = new.data['exterior'].translate(offset, output_dims, inplace)
        new.data['interiors'] = [p.translate(offset, output_dims, inplace) for p in new.data['interiors']]
        return new


class Polygon(_PolyArrayBackend, _PolyWarpMixin):
    """
    Represents a single polygon as set of exterior boundary points and a list
    of internal polygons representing holes.

    Example:
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [
        >>>         np.array([[13, 13], [14, 12], [24, 12], [25, 13], [25, 18], [24, 19], [14, 19], [13, 18]]),
        >>>         np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11], [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = Polygon(**data)
    """
    __datakeys__ = ['exterior', 'interiors']

    def __init__(self, data=None, datakeys=None, **kwargs):
        if kwargs:
            if data:
                raise ValueError('Cannot specify kwargs AND data dict')
            _datakeys = self.__datakeys__
            # Allow the user to specify custom data keys
            if datakeys is not None:
                _datakeys = _datakeys + list(datakeys)
            # Perform input checks whenever kwargs is given
            data = {key: kwargs.pop(key) for key in _datakeys if key in kwargs}
            if kwargs:
                raise ValueError(
                    'Unknown kwargs: {}'.format(sorted(kwargs.keys())))

            import kwimage
            if 'exterior' in data:
                if isinstance(data['exterior'], (np.ndarray, torch.Tensor)):
                    data['exterior'] = kwimage.Coords(data['exterior'])
            if 'interiors' in data:
                holes = []
                for hole in data['interiors']:
                    if isinstance(hole, (np.ndarray, torch.Tensor)):
                        hole = kwimage.Coords(hole)
                    holes.append(hole)
                data['interiors'] = holes
            else:
                data['interiors'] = []

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data is explicitly specified
            data = data.data
        self.data = data

    @classmethod
    def random(cls, n=6, n_holes=0, rng=None):
        """
        Args:
            n (int): number of points in the polygon (must be more than 4)
            n_holes (int): number of holes

        Example:
            >>> rng = 0
            >>> n = 20
            >>> n_holes = 2
            >>> cls = Polygon
            >>> self = Polygon.random(n=n, rng=rng, n_holes=n_holes)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> self.draw()

        References:
            https://gis.stackexchange.com/questions/207731/random-multipolygon
            https://stackoverflow.com/questions/8997099/random-polygon
            https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons
        """
        import kwarray
        import shapely
        from shapely.geometry import Point
        import scipy
        rng = kwarray.ensure_rng(rng)
        points = rng.rand(n, 2)

        # exterior = _order_vertices(points)
        hull = scipy.spatial.ConvexHull(points)
        exterior = hull.points[hull.vertices]
        exterior = _order_vertices(exterior)

        polygon = shapely.geometry.Polygon(shell=exterior)

        def generate_random(number, polygon, rng):
            # FIXME: needs to be inside a convex portion of the polygon
            list_of_points = []
            minx, miny, maxx, maxy = polygon.bounds
            counter = 0
            while counter < number:
                xy = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                pnt = Point(*xy)
                if polygon.contains(pnt):
                    list_of_points.append(xy)
                    counter += 1
            return list_of_points

        interiors = []
        for _ in range(n_holes):
            polygon = shapely.geometry.Polygon(shell=exterior, holes=interiors)
            in_pts = generate_random(4, polygon, rng)
            interior = _order_vertices(np.array(in_pts))[::-1]
            interiors.append(interior)

        self = cls(exterior=exterior, interiors=interiors)
        return self

    @ub.memoize_property
    def _impl(self):
        return self.data['exterior']._impl

    def draw_on(self, image, color='blue', fill=True, border=False):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1).scale(128)
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> image = self.draw_on(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(image, fnum=1)
        """
        import kwplot
        import kwimage
        # return shape of contours to openCV contours

        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8
        image = kwimage.ensure_uint255(image)
        image = kwimage.atleast_3channels(image)

        data = self.data
        coords = [data['exterior']] + data['interiors']
        contours = [np.expand_dims(c.data.astype(np.int), axis=1) for c in coords]

        rgb = kwplot.Color(color).as255()

        if fill:
            image = cv2.fillPoly(image, contours, rgb, line_type, shift=0)

        if border:
            thickness = 1
            contour_idx = -1
            image = cv2.drawContours(image, contours, contour_idx, rgb,
                                     thickness, line_type)
        image = kwimage.ensure_float01(image)
        return image

    def to_mask(self, dims=None):
        """
        Convert this polygon to a mask

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1).scale(128)
            >>> mask = self.to_mask((128, 128))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> mask.draw(color='blue')
            >>> mask.to_multi_polygon().draw(color='red', alpha=.5)
        """
        import kwimage
        c_mask = np.zeros(dims, dtype=np.uint8)
        # return shape of contours to openCV contours

        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8

        data = self.data
        coords = [data['exterior']] + data['interiors']
        contours = [np.expand_dims(c.data.astype(np.int), axis=1) for c in coords]

        value = 1

        c_mask = cv2.fillPoly(c_mask, contours, value, line_type, shift=0)

        mask = kwimage.Mask(c_mask, 'c_mask')
        return mask

    def draw(self, color='blue', ax=None, alpha=1.0, radius=1):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1)
            >>> self = self.scale(100)
            >>> self.draw()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw()
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 120)
            >>> ax.set_ylim(0, 120)
            >>> ax.invert_yaxis()
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        data = self.data

        exterior = data['exterior'].data.tolist()
        exterior.append(exterior[0])
        n = len(exterior)
        verts = []
        verts.extend(exterior)
        codes = [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        interiors = data.get('interiors', [])
        for hole in interiors:
            hole = hole.data.tolist()
            hole.append(hole[0])
            n = len(hole)
            verts.extend(hole)
            codes += [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        verts = np.array(verts)
        path = Path(verts, codes)
        patch = mpl.patches.PathPatch(path, alpha=alpha, color=color)
        ax.add_patch(patch)


def _order_vertices(verts):
    """
    References:
        https://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
    """
    mlat = verts.T[0].sum() / len(verts)
    mlng = verts.T[1].sum() / len(verts)

    tau = np.pi * 2
    angle = (np.arctan2(mlat - verts.T[0], verts.T[1] - mlng) + tau) % tau
    sortx = angle.argsort()
    verts = verts.take(sortx, axis=0)
    return verts


class MultiPolygon(_generic.ObjectList):

    @classmethod
    def random(self, n=3, rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)
        data = [Polygon.random(rng=rng) for _ in range(n)]
        self = MultiPolygon(data)
        return self

    def to_mask(self, dims=None):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> s = 100
            >>> self = MultiPolygon.random(rng=0).scale(s)
            >>> dims = (s, s)
            >>> mask = self.to_mask(dims)

            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> from matplotlib import pyplot as pl
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, s)
            >>> ax.set_ylim(0, s)
            >>> self.draw(color='red', alpha=.4)
            >>> mask.draw(color='blue', alpha=.4)
        """
        import kwimage
        masks = [poly.to_mask(dims) for poly in self.data]
        mask = kwimage.Mask.union(*masks)
        return mask


class PolygonList(_generic.ObjectList):
    pass
