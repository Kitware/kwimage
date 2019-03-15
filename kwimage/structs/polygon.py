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
            >>> self = Polygon.random(10)
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> input_dims = (10, 10)
            >>> new = self._warp_imgaug(augmenter, input_dims)
            >>> new2 = self.warp(augmenter, input_dims)
        """
        raise NotImplementedError
        new = self if inplace else self.__class__(self.data.copy())
        kpoi = new.to_imgaug(shape=input_dims)
        kpoi = augmenter.augment_keypoints(kpoi)
        xy = np.array([[kp.x, kp.y] for kp in kpoi.keypoints],
                      dtype=new.data['xy'].data.dtype)
        new.data['xy'].data = xy
        return new

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

    @_generic.memoize_property
    def _impl(self):
        return self.data['exterior']._impl

    def draw_on(self, image, color='blue', fill=True):
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
        thickness = 1

        contour_idx = -1
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

        if 1:
            image = cv2.drawContours(image, contours, contour_idx, rgb,
                                     thickness, line_type)
        image = kwimage.ensure_float01(image)
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1):
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
        patch = mpl.patches.PathPatch(path, alpha=.4)
        ax.add_patch(patch)

    def to_imgaug(self, shape):
        import imgaug
        ia_exterior = imgaug.Polygon(self.data['exterior'])
        ia_interiors = [imgaug.Polygon(p) for p in self.data.get('interiors', [])]
        iamp = imgaug.MultiPolygon([ia_exterior] + ia_interiors)
        return iamp


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
    pass


class PolygonList(_generic.ObjectList):
    pass
