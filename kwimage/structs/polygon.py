import ubelt as ub
import cv2
import numpy as np
import torch
import skimage
from . import _generic


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

        if True:
            newdata = {}
            for k, v in self.data.items():
                if hasattr(v, 'tensor'):
                    v2 = v.tensor(device)
                elif isinstance(v, list):
                    v2 = [x.tensor(device) for x in v]
                else:
                    v2 = impl.tensor(v, device)
                newdata[k] = v2
        else:
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
        if True:
            newdata = {}
            for k, v in self.data.items():
                if hasattr(v, 'numpy'):
                    v2 = v.numpy()
                elif isinstance(v, list):
                    v2 = [x.numpy() for x in v]
                else:
                    v2 = impl.numpy(v)
                newdata[k] = v2
        else:
            # newdata = {k: v.tensor(device) if hasattr(v, 'tensor')
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
            >>> # xdoctest: +REQUIRES(module:imgaug)
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

        import imgaug
        from distutils.version import LooseVersion
        if LooseVersion(imgaug.__version__) <= LooseVersion('0.2.9'):
            # Hack to fix imgaug bug
            h, w = input_dims
            input_dims = (int(h + 1.0), int(w + 1.0))
        else:
            # Note: the bug was in FlipLR._augment_keypoints, denoted by a todo
            # comment: "is this still correct with float keypoints?  Seems like
            # the -1 should be dropped"
            # raise Exception('WAS THE BUG FIXED IN A NEW VERSION? '
            #                 'imgaug.__version__={}'.format(imgaug.__version__))
            # Yes, the bug was fixed. I fixed it.
            pass
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

        Doctest:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> self = Polygon.random()
            >>> import imgaug
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> new = self.warp(augmenter, input_dims=(1, 1))
            >>> print('new = {!r}'.format(new.data))
            >>> print('self = {!r}'.format(self.data))
            >>> #assert np.all(self.warp(np.eye(3)).exterior == self.exterior)
            >>> #assert np.all(self.warp(np.eye(2)).exterior == self.exterior)
        """
        new = self if inplace else self.__class__(self.data.copy())
        # print('WARP new = {!r}'.format(new))
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
        """
        new = self if inplace else self.__class__(self.data.copy())
        new.data['exterior'] = new.data['exterior'].translate(
            offset, output_dims, inplace)
        new.data['interiors'] = [p.translate(offset, output_dims, inplace)
                                 for p in new.data['interiors']]
        return new


class Polygon(_generic.Spatial, _PolyArrayBackend, _PolyWarpMixin, ub.NiceRepr):
    """
    Represents a single polygon as set of exterior boundary points and a list
    of internal polygons representing holes.

    By convention exterior boundaries should be counterclockwise and interior
    holes should be clockwise.

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
                if isinstance(data['exterior'], (list, tuple)):
                    data['exterior'] = kwimage.Coords(np.array(data['exterior']))
                elif isinstance(data['exterior'], (np.ndarray, torch.Tensor)):
                    data['exterior'] = kwimage.Coords(data['exterior'])
            if 'interiors' in data:
                holes = []
                for hole in data['interiors']:
                    if isinstance(hole, (list, tuple)):
                        hole = kwimage.Coords(np.array(hole))
                    elif isinstance(hole, (np.ndarray, torch.Tensor)):
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

    def __nice__(self):
        return ub.repr2(self.data, nl=1)

    @classmethod
    def random(cls, n=6, n_holes=0, convex=True, tight=False, rng=None):
        """
        Args:
            n (int): number of points in the polygon (must be more than 4)
            n_holes (int): number of holes
            tight (bool, default=False): fits the minimum and maximum points
                between 0 and 1

        CommandLine:
            xdoctest -m kwimage.structs.polygon Polygon.random

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

        import scipy
        rng = kwarray.ensure_rng(rng)
        points = rng.rand(n, 2)

        if convex:
            hull = scipy.spatial.ConvexHull(points)
            exterior = hull.points[hull.vertices]
        else:
            exterior = points
        exterior = _order_vertices(exterior)

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
        if n_holes:
            try:
                import shapely
                from shapely.geometry import Point
            except Exception:
                print('FAILED TO IMPORT SHAPELY')
                raise
            polygon = shapely.geometry.Polygon(shell=exterior)
            for _ in range(n_holes):
                polygon = shapely.geometry.Polygon(shell=exterior, holes=interiors)
                in_pts = generate_random(4, polygon, rng)
                interior = _order_vertices(np.array(in_pts))[::-1]
                interiors.append(interior)

        self = cls(exterior=exterior, interiors=interiors)

        if tight:
            min_xy = self.data['exterior'].data.min(axis=0)
            max_xy = self.data['exterior'].data.max(axis=0)
            extent = max_xy - min_xy
            self = self.translate(-min_xy).scale(1 / extent)
        return self

    @ub.memoize_property
    def _impl(self):
        return self.data['exterior']._impl

    def draw_on(self, image, color='blue', fill=True, border=False, alpha=1.0):
        """
        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
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

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8

        data = self.data
        coords = [data['exterior']] + data['interiors']
        contours = [np.expand_dims(c.data.astype(np.int), axis=1) for c in coords]

        # alpha = 1.0
        if alpha == 1.0:
            image = kwimage.ensure_uint255(image)
            image = kwimage.atleast_3channels(image)
            rgba = kwplot.Color(color).as255()
        else:
            # fill = False
            image = kwimage.ensure_float01(image)
            image = kwimage.ensure_alpha_channel(image)
            rgba = kwplot.Color(color, alpha=alpha).as01()
            # print('rgba = {!r}'.format(rgba))
        # print('rgba = {!r}'.format(rgba))
        # print('image = {!r}'.format(image.shape))
        # alpha # TODO

        if fill:
            if alpha == 1.0:
                image = cv2.fillPoly(image, contours, rgba, line_type, shift=0)
            else:
                orig = image.copy()
                mask = np.zeros_like(orig)
                mask = cv2.fillPoly(mask, contours, rgba, line_type, shift=0)
                image = kwimage.overlay_alpha_images(mask, orig)

        if border or True:
            thickness = 4
            contour_idx = -1
            image = cv2.drawContours(image, contours, contour_idx, rgba,
                                     thickness, line_type)
        image = kwimage.ensure_float01(image)[..., 0:3]

        image = dtype_fixer(image)
        return image

    def to_mask(self, dims=None):
        """
        Convert this polygon to a mask

        TODO:
            - [ ] currently not efficient

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
        if dims is None:
            raise Exception('REQUIRES DIMS')

        self.to_boxes()

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

    def to_shapely(self):
        """
        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoc: +REQUIRES(module:shapely)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1)
            >>> self = self.scale(100)
            >>> geom = self.to_shapely()
            >>> print('geom = {!r}'.format(geom))
        """
        import shapely
        import shapely.geometry
        geom = shapely.geometry.Polygon(
            shell=self.data['exterior'].data,
            holes=[c.data for c in self.data['interiors']]
        )
        return geom

    def draw(self, color='blue', ax=None, alpha=1.0, radius=1, setlim=False):
        """
        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1)
            >>> self = self.scale(100)
            >>> self.draw()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw(setlim=True)
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        import kwplot
        if ax is None:
            ax = plt.gca()

        color = list(kwplot.Color(color).as01())

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

        if setlim:
            x1, y1, x2, y2 = self.to_boxes().to_tlbr().data[0]
            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)

    def _to_coco(self):
        interiors = self.data.get('interiors', [])
        if interiors:
            raise NotImplementedError('no holes yet')
            _new = {
                'exterior': self.data['exterior'].data.ravel().tolist(),
                'interiors': [item.data.ravel().tolist() for item in interiors]
            }
            return _new
        else:
            return self.data['exterior'].data.ravel().tolist()

    def to_coco(self):
        return self._to_coco()

    def to_multi_polygon(self):
        return MultiPolygon([self])

    def to_boxes(self):
        import kwimage
        xys = self.data['exterior'].data
        tl = xys.min(axis=0)
        br = xys.max(axis=0)
        tlbr = np.hstack([tl, br])[None, :]
        boxes = kwimage.Boxes(tlbr, 'tlbr')
        return boxes
        # return MultiPolygon([self])

    def copy(self):
        self2 = Polygon(self.data)
        self2.data['exterior'] = self2.data['exterior'].copy()
        self2.data['interiors'] = [x.copy() for x in self2.data['interiors']]
        return self2

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip polygon to image boundaries.

        Example:
            >>> from kwimage.structs.polygon import *
            >>> self = Polygon.random().scale(10).translate(-1)
            >>> self2 = self.clip(1, 1, 3, 3)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self2.draw(setlim=True)
        """
        if inplace:
            self2 = self
        else:
            self2 = self.copy()
        impl = self._impl
        xs, ys = impl.T(self2.data['exterior'].data)
        np.clip(xs, x_min, x_max, out=xs)
        np.clip(ys, y_min, y_max, out=ys)
        return self2


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
    def random(self, n=3, rng=None, tight=False):
        import kwarray
        rng = kwarray.ensure_rng(rng)
        data = [Polygon.random(rng=rng, tight=tight) for _ in range(n)]
        self = MultiPolygon(data)
        return self

    def to_multi_polygon(self):
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

    @classmethod
    def coerce(cls, data, dims=None):
        """
        See Mask.coerce
        """
        if data is None:
            return None
        from kwimage.structs.mask import _coerce_coco_segmentation
        self = _coerce_coco_segmentation(data, dims=dims)
        self = self.to_multi_polygon()
        return self

    def to_shapely(self):
        """
        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoc: +REQUIRES(module:shapely)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = MultiPolygon.random(rng=0)
            >>> geom = self.to_shapely()
            >>> print('geom = {!r}'.format(geom))
        """
        import shapely
        import shapely.geometry
        polys = [p.to_shapely() for p in self.data]
        geom = shapely.geometry.MultiPolygon(polys)
        return geom

    def _to_coco(self):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = MultiPolygon.random(1, rng=0)
            >>> self._to_coco()
        """
        return [item._to_coco() for item in self.data]

    def to_coco(self):
        return [item.to_coco() for item in self.data]


class PolygonList(_generic.ObjectList):
    pass
