import ubelt as ub
import cv2
import numpy as np
import skimage
import numbers
from . import _generic

try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile


class _PolyArrayBackend:
    def is_numpy(self):
        return self._impl.is_numpy

    def is_tensor(self):
        return self._impl.is_tensor

    def tensor(self, device=ub.NoParam):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
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
        new = self.__class__(newdata, self.meta)
        return new

    def numpy(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
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
        new = self.__class__(newdata, self.meta)
        return new


class _PolyWarpMixin:

    # @profile
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
            >>> assert np.allclose(self.data['exterior'].data[:, 1], new.data['exterior'].data[:, 1])
            >>> assert np.allclose(input_dims[0] - self.data['exterior'].data[:, 0], new.data['exterior'].data[:, 0])

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
        import kwimage
        new = self if inplace else self.__class__(self.data.copy())

        # current version of imgaug doesnt fully support polygons
        # coerce to and from points instead
        dtype = self.data['exterior'].data.dtype

        parts = [self.data['exterior']] + self.data.get('interiors', [])
        parts = [p.data for p in parts]
        cs = [0] + np.cumsum(np.array(list(map(len, parts)))).tolist()
        flat_kps = np.concatenate(parts, axis=0)

        flat_coords = kwimage.Coords(flat_kps)
        flat_coords = flat_coords._warp_imgaug(augmenter, input_dims, inplace=True)
        flat_parts = flat_coords.data
        new_parts = []
        for a, b in ub.iter_window(cs, 2):
            new_part = np.array(flat_parts[a:b], dtype=dtype)
            new_parts.append(new_part)

        new_exterior = kwimage.Coords(new_parts[0])
        new_interiors = [kwimage.Coords(p) for p in new_parts[1:]]
        new.data['exterior'] = new_exterior
        new.data['interiors'] = new_interiors
        return new

    # @profile
    def to_imgaug(self, shape):
        import imgaug
        ia_exterior = imgaug.Polygon(self.data['exterior'])
        ia_interiors = [imgaug.Polygon(p) for p in self.data.get('interiors', [])]
        iamp = imgaug.MultiPolygon([ia_exterior] + ia_interiors)
        return iamp

    # @profile
    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Generalized coordinate transform.

        Args:
            transform (GeometricTransform | ArrayLike | Augmenter | callable):
                scikit-image tranform, a 3x3 transformation matrix,
                an imgaug Augmenter, or generic callable which transforms
                an NxD ndarray.

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
            # else:
            #     raise TypeError(type(transform))
        new.data['exterior'] = new.data['exterior'].warp(transform, input_dims,
                                                         output_dims, inplace)
        new.data['interiors'] = [
            p.warp(transform, input_dims, output_dims, inplace)
            for p in new.data['interiors']
        ]
        return new

    @profile
    def scale(self, factor, about=None, output_dims=None, inplace=False):
        """
        Scale a polygon by a factor

        Args:
            factor (float or Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.
            about (Tuple | None):
                if unspecified scales about the origin (0, 0), otherwise the
                rotation is about this point.
            output_dims (Tuple): unused in non-raster spatial structures
            inplace (bool, default=False): if True, modifies data inplace

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> new = self.scale(10)

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0).translate((0.5))
            >>> new = self.scale(1.5, about='center')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> self.draw(color='red', alpha=0.5)
            >>> new.draw(color='blue', alpha=0.5, setlim=True)
        """
        new = self if inplace else self.__class__(self.data.copy())
        about = self._rectify_about(about)
        new.data['exterior'] = new.data['exterior'].scale(
            factor, about=about, output_dims=output_dims, inplace=inplace)
        new.data['interiors'] = [
            p.scale(factor, about=about, output_dims=output_dims,
                    inplace=inplace)
            for p in new.data['interiors']]
        return new

    @profile
    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the polygon up/down left/right

        Args:
            factor (float or Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures
            inplace (bool, default=False): if True, modifies data inplace

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

    @profile
    def rotate(self, theta, about=None, output_dims=None, inplace=False):
        """
        Rotate the polygon

        Args:
            theta (float):
                rotation angle in radians

            about (Tuple | None | str):
                if unspecified rotates about the origin (0, 0). If "center"
                then rotate around the center of this polygon. Otherwise the
                rotation is about a custom specified point.

            output_dims (Tuple): unused in non-raster spatial structures

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> new = self.rotate(np.pi / 2, about='center')
            >>> new2 = self.rotate(np.pi / 2)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> self.draw(color='red', alpha=0.5)
            >>> new.draw(color='blue', alpha=0.5)
        """
        new = self if inplace else self.__class__(self.data.copy())
        about = self._rectify_about(about)
        new.data['exterior'] = new.data['exterior'].rotate(
            theta, about, output_dims, inplace)
        new.data['interiors'] = [p.rotate(theta, about, output_dims, inplace)
                                 for p in new.data['interiors']]
        return new

    def _rectify_about(self, about):
        """
        Ensures that about returns a specified point. Allows for special keys
        like center to be used.

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> self._rectify_about('center')
        """
        if about is None:
            about_ = None
        else:
            if isinstance(about, str):
                if about == 'origin':
                    about_ = (0., 0.)
                elif about == 'center':
                    centroid = self.to_shapely().centroid
                    about_ = (centroid.x, centroid.y)
                else:
                    raise KeyError(about)
            else:
                about_ = about if ub.iterable(about) else [about] * 2
        return about_

    def swap_axes(self, inplace=False):
        """
        Swap the x and y coordinate axes

        Args:
            inplace (bool, default=False): if True, modifies data inplace

        Returns:
            Polygon: modified polygon
        """
        new = self if inplace else self.__class__(self.data.copy())
        new.data['exterior'] = new.data['exterior'].reorder_axes(
            (1, 0), inplace=inplace)
        new.data['interiors'] = [
            p.reorder_axes((1, 0), inplace=inplace)
            for p in new.data['interiors']]
        return new


class Polygon(_generic.Spatial, _PolyArrayBackend, _PolyWarpMixin, ub.NiceRepr):
    """
    Represents a single polygon as set of exterior boundary points and a list
    of internal polygons representing holes.

    By convention exterior boundaries should be counterclockwise and interior
    holes should be clockwise.

    Example:
        >>> import kwimage
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [
        >>>         np.array([[13, 13], [14, 12], [24, 12], [25, 13], [25, 18],
        >>>                   [24, 19], [14, 19], [13, 18]]),
        >>>         np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11],
        >>>                   [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = kwimage.Polygon(**data)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw(setlim=True)

    Example:
        >>> import kwimage
        >>> self = kwimage.Polygon.random(
        >>>     n=5, n_holes=1, convex=False, rng=0)
        >>> print('self = {}'.format(self))
        self = <Polygon({
            'exterior': <Coords(data=
                            array([[0.30371392, 0.97195856],
                                   [0.24372304, 0.60568445],
                                   [0.21408694, 0.34884262],
                                   [0.5799477 , 0.44020379],
                                   [0.83720288, 0.78367234]]))>,
            'interiors': [<Coords(data=
                             array([[0.50164209, 0.83520279],
                                    [0.25835064, 0.40313428],
                                    [0.28778562, 0.74758761],
                                    [0.30341266, 0.93748088]]))>],
        })>
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw(setlim=True)
    """
    __datakeys__ = ['exterior', 'interiors']
    __metakeys__ = ['classes']

    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None, **kwargs):
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

            import kwimage
            if 'exterior' in data:
                if isinstance(data['exterior'], (list, tuple)):
                    data['exterior'] = kwimage.Coords(np.array(data['exterior']))
                elif isinstance(data['exterior'], _generic.ARRAY_TYPES):
                    data['exterior'] = kwimage.Coords(data['exterior'])
            if 'interiors' in data:
                holes = []
                for hole in data['interiors']:
                    if isinstance(hole, (list, tuple)):
                        hole = kwimage.Coords(np.array(hole))
                    elif isinstance(hole, _generic.ARRAY_TYPES):
                        hole = kwimage.Coords(hole)
                    holes.append(hole)
                data['interiors'] = holes
            else:
                data['interiors'] = []

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data is explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    @property
    def exterior(self):
        return self.data['exterior']

    @property
    def interiors(self):
        return self.data['interiors']

    def __nice__(self):
        return ub.repr2(self.data, nl=1)

    @classmethod
    def circle(cls, xy, r, resolution=64):
        """
        Create a circular polygon

        Example:
            >>> xy = (0.5, 0.5)
            >>> r = .3
            >>> poly = Polygon.circle(xy, r)
        """
        tau = 2 * np.pi
        theta = np.linspace(0, tau, resolution)
        y_offset = np.sin(theta) * r
        x_offset = np.cos(theta) * r

        center = np.array(xy)
        xcoords = center[0] + x_offset
        ycoords = center[1] + y_offset

        exterior = np.hstack([
            xcoords.ravel()[:, None],
            ycoords.ravel()[:, None],
        ])

        self = cls(exterior=exterior)
        return self

    @classmethod
    def random(cls, n=6, n_holes=0, convex=True, tight=False, rng=None):
        """
        Args:
            n (int): number of points in the polygon (must be 3 or more)
            n_holes (int): number of holes
            tight (bool, default=False): fits the minimum and maximum points
                between 0 and 1
            convex (bool, default=True): force resulting polygon will be convex
               (may remove exterior points)

        CommandLine:
            xdoctest -m kwimage.structs.polygon Polygon.random

        Example:
            >>> rng = None
            >>> n = 4
            >>> n_holes = 1
            >>> cls = Polygon
            >>> self = Polygon.random(n=n, rng=rng, n_holes=n_holes, convex=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.autompl()
            >>> self.draw()

        References:
            https://gis.stackexchange.com/questions/207731/random-multipolygon
            https://stackoverflow.com/questions/8997099/random-polygon
            https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons
            https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
        """
        import kwarray
        import scipy
        rng = kwarray.ensure_rng(rng)

        def _gen_polygon2(n, irregularity, spikeyness):
            """
            Creates the polygon by sampling points on a circle around the centre.
            Random noise is added by varying the angular spacing between sequential points,
            and by varying the radial distance of each point from the centre.

            Based on original code by Mike Ounsworth

            Args:
                n (int): number of vertices
                irregularity (float): [0,1] indicating how much variance there
                    is in the angular spacing of vertices. [0,1] will map to
                    [0, 2pi/numberOfVerts]
                spikeyness (float): [0,1] indicating how much variance there is
                    in each vertex from the circle of radius aveRadius. [0,1] will
                    map to [0, aveRadius]

            Returns:
                a list of vertices, in CCW order.

            Example:
                n = 4
                irregularity = 0
                spikeyness = 0
            """
            # Generate around the unit circle
            cx, cy = (0.0, 0.0)
            radius = 1

            tau = np.pi * 2

            irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
            spikeyness = np.clip(spikeyness, 1e-9, 1)

            # generate n angle steps
            lower = (tau / n) - irregularity
            upper = (tau / n) + irregularity
            angle_steps = rng.uniform(lower, upper, n)

            # normalize the steps so that point 0 and point n+1 are the same
            k = angle_steps.sum() / (2 * np.pi)
            angles = (angle_steps / k).cumsum() + rng.uniform(0, tau)

            from kwarray import distributions
            tnorm = distributions.TruncNormal(radius, spikeyness,
                                              low=0, high=2 * radius, rng=rng)

            # now generate the points
            radii = tnorm.sample(n)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)

            points = np.hstack([x_pts[:, None], y_pts[:, None]])

            # Scale to 0-1 space
            points = points - points.min(axis=0)
            points = points / points.max(axis=0)

            # Randomly place within 0-1 space
            points = points * (rng.rand() * .8 + .2)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            high = (1 - max_pt)
            low = (0 - min_pt)
            offset = (rng.rand(2) * (high - low)) + low
            points = points + offset
            return points

        # points = rng.rand(n, 2)
        points = _gen_polygon2(n, 0.9, 0.1 if convex else 0.9)

        if convex:
            points = _order_vertices(points)
            hull = scipy.spatial.ConvexHull(points)
            exterior = hull.points[hull.vertices]

            # hack
            if len(exterior) != n:
                points = _gen_polygon2(n, 1.0, 0)
                points = _order_vertices(points)
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

    def to_mask(self, dims=None):
        """
        Convert this polygon to a mask

        TODO:
            - [ ] currently not efficient

        Args:
            dims (Tuple): height and width of the output mask

        Returns:
            kwimage.Mask

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
            raise ValueError('Must specify output raster dimensions')
        c_mask = np.zeros(dims, dtype=np.uint8)
        value = 1
        self.fill(c_mask, value)
        mask = kwimage.Mask(c_mask, 'c_mask')
        return mask

    def to_relative_mask(self):
        """
        Returns a translated mask such the mask dimensions are minimal.

        In other words, we move the polygon all the way to the top-left and
        return a mask just big enough to fit the polygon.

        Returns:
            kwimage.Mask

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random().scale(8).translate(100, 100)
            >>> mask = self.to_relative_mask()
            >>> assert mask.shape <= (8, 8)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> mask.draw(color='blue')
            >>> mask.to_multi_polygon().draw(color='red', alpha=.5)
        """
        x, y, w, h = self.to_boxes().quantize().to_xywh().data[0]
        mask = self.translate((-x, -y)).to_mask(dims=(h, w))
        return mask

    def fill(self, image, value=1):
        """
        Inplace fill in an image based on this polyon.

        Args:
            image (ndarray): image to draw on
            value (int | Tuple[int], default=1): value fill in with

        Returns:
            ndarray: the image that has been modified in place
        """
        # line_type = cv2.LINE_AA
        cv_contours = self._to_cv_countours()
        line_type = cv2.LINE_8
        # Modification happens inplace
        cv2.fillPoly(image, cv_contours, value, line_type, shift=0)
        return image

    def _to_cv_countours(self):
        """
        OpenCV polygon representation, which is a list of points.  Holes are
        implicitly represented. When another polygon is drawn over an existing
        polyon via cv2.fillPoly

        Returns:
            List[ndarray]: where each ndarray is of shape [N, 1, 2],
                where N is the number of points on the boundary, the middle
                dimension is always 1, and the trailing dimension represents
                x and y coordinates respectively.
        """
        data = self.data
        coords = [data['exterior']] + data['interiors']
        cv_contours = [np.expand_dims(c.data.astype(int), axis=1)
                       for c in coords]
        return cv_contours

    @classmethod
    def coerce(Polygon, data):
        """
        Try to autodetermine format of input polygon and coerce it into a
        kwimage.Polygon.

        Args:
            data (object): some type of data that can be interpreted as a
                polygon.

        Returns:
            kwimage.Polygon

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random()
            >>> self.coerce(self)
            >>> self.coerce(self.exterior)
            >>> self.coerce(self.exterior.data)
            >>> self.coerce(self.data)
            >>> self.coerce(self.to_geojson())
        """
        import kwimage
        if isinstance(data, Polygon):
            return data
        if isinstance(data, (np.ndarray, kwimage.Coords)):
            return Polygon(exterior=data)  # TODO accept torch
        if isinstance(data, str):
            return Polygon.from_wkt(data)
        if isinstance(data, dict):
            if 'coordinates' in data:
                return Polygon.from_geojson(data)
            if 'exterior' in data:
                return Polygon(data)

        import shapely
        import shapely.geometry
        if isinstance(data, shapely.geometry.Polygon):
            return Polygon.from_shapely(data)
        raise TypeError(
            'coerce data into a polygon not implemented for this case: {}'.format(
                type(data)))

    @classmethod
    def from_shapely(Polygon, geom):
        """
        Convert a shapely polygon to a kwimage.Polygon

        Args:
            geom (shapely.geometry.polygon.Polygon): a shapely polygon

        Returns:
            kwimage.Polygon
        """
        exterior = np.array(geom.exterior.coords.xy).T
        interiors = [np.array(g.coords.xy).T for g in geom.interiors]
        self = Polygon(exterior=exterior, interiors=interiors)
        return self

    @classmethod
    def from_wkt(Polygon, data):
        """
        Convert a WKT string to a kwimage.Polygon

        Args:
            data (str): a WKT polygon string

        Returns:
            kwimage.Polygon

        Example:
            >>> import kwimage
            >>> data = 'POLYGON ((0.11 0.61, 0.07 0.588, 0.015 0.50, 0.11 0.61))'
            >>> self = kwimage.Polygon.from_wkt(data)
            >>> assert len(self.exterior) == 4
        """
        from shapely import wkt
        geom = wkt.loads(data)
        self = Polygon.from_shapely(geom)
        return self

    @classmethod
    def from_geojson(Polygon, data_geojson):
        """
        Convert a geojson polygon to a kwimage.Polygon

        Args:
            data_geojson (dict): geojson data

        References:
            https://geojson.org/geojson-spec.html

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=2)
            >>> data_geojson = self.to_geojson()
            >>> new = Polygon.from_geojson(data_geojson)
        """
        # By the spec a Polygon should have 3 levels of nesting, but lets
        # handle the common case (mistake?) where there are only two
        geojson_type = data_geojson.get('type', 'Polygon').lower()
        if geojson_type != 'polygon':
            raise ValueError('Type is {}, not Polygon'.format(geojson_type))

        # TODO: better method for checking nest depth
        coords = data_geojson['coordinates']
        def check_leftmost_depth(data):
            # quick check leftmost depth of a nested struct
            item = data
            depth = 0
            while isinstance(item, list):
                if len(item) == 0:
                    raise Exception('no child node')
                item = item[0]
                depth += 1
            return depth
        depth = check_leftmost_depth(coords)
        if depth == 2:
            raise Exception(ub.codeblock(
                '''
                The GEOJSON spec has a depth of 3!

                coodinates should be:
                    'coordinates': [
                       [ [x_1, y_1], ... , [x_n, y_n] ],  # exterior
                       [ [x_1, y_1], ... , [x_n, y_n] ],  # hole 1
                       [ [x_1, y_1], ... , [x_n, y_n] ],  # hole 2
                    ]

                '''))
            # exterior = np.array(coords)
            # interiors = []
        elif depth == 3:
            exterior = np.array(coords[0])
            interiors = [np.array(h) for h in coords[1:]]
        else:
            raise Exception('Unknown geojson format')
        self = Polygon(exterior=exterior, interiors=interiors)
        return self

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

    def to_geojson(self):
        """
        Converts polygon to a geojson structure

        Returns:
            Dict[str, object]

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random()
            >>> print(self.to_geojson())
        """
        coords = [self.data['exterior'].data.tolist()]
        holes = [interior.data.tolist() for interior in self.data['interiors']]
        if holes:
            coords.extend(holes)
        geojson = {
            'type': 'Polygon',
            'coordinates': coords,
        }
        return geojson

    def to_wkt(self):
        """
        Convert a kwimage.Polygon to WKT string

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random()
            >>> print(self.to_wkt())
        """
        return self.to_shapely().to_wkt()

    @classmethod
    def from_coco(cls, data, dims=None):
        """
        Accepts either new-style or old-style coco polygons
        """
        if isinstance(data, list):
            if len(data) > 0:
                assert isinstance(ub.peek(data), numbers.Number)
                exterior = np.array(data).reshape(-1, 2)
                self = cls(exterior=exterior)
            else:
                self = cls(exterior=[])
        elif isinstance(data, dict):
            if 'exterior' not in data:
                raise ValueError('dict requires exterior key')
            self = cls(**data)
        else:
            raise TypeError(type(data))
        return self

    def _to_coco(self, style='orig'):
        return self.to_coco(style=style)

    def to_coco(self, style='orig'):
        """
        Returns:
            List | Dict : coco-style polygons
        """
        interiors = self.data.get('interiors', [])
        if style == 'orig':
            if interiors:
                raise ValueError('Original coco does not support holes')
            return self.data['exterior'].data.ravel().tolist()
        elif style == 'new':
            _new = {
                'exterior': self.data['exterior'].data.tolist(),
                'interiors': [item.data.tolist() for item in interiors]
            }
            return _new
        else:
            raise KeyError(style)

    def to_multi_polygon(self):
        return MultiPolygon([self])

    def to_boxes(self):
        """
        Deprecated: lossy conversion use 'bounding_box' instead
        """
        return self.bounding_box()

    def bounding_box(self):
        """
        Returns an axis-aligned bounding box for the segmentation

        Returns:
            kwimage.Boxes
        """
        import kwimage
        xys = self.data['exterior'].data
        lt = xys.min(axis=0)
        rb = xys.max(axis=0)
        ltrb = np.hstack([lt, rb])[None, :]
        boxes = kwimage.Boxes(ltrb, 'ltrb')
        return boxes

    def bounding_box_polygon(self):
        """
        Returns an axis-aligned bounding polygon for the segmentation.

        Notes:
            This Polygon will be a Box, not a convex hull! Use shapely for
            convex hulls.

        Returns:
            kwimage.Polygon
        """
        new = self.bounding_box().to_polygons()[0]
        return new

    def copy(self):
        self2 = Polygon(self.data, self.meta)
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

    def draw_on(self, image, color='blue', fill=True, border=False, alpha=1.0,
                copy=False):
        """
        Rasterizes a polygon on an image. See `draw` for a vectorized
        matplotlib version.

        Args:
            image (ndarray): image to raster polygon on.
            color (str | tuple): data coercable to a color
            fill (bool, default=True): draw the center mass of the polygon
            border (bool, default=False): draw the border of the polygon
            alpha (float, default=1.0): polygon transparency (setting alpha < 1
                makes this function much slower).
            copy (bool, default=False): if False only copies if necessary

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

        Example:
            >>> import kwimage
            >>> color = 'blue'
            >>> self = kwimage.Polygon.random(n_holes=1).scale(128)
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> # Test drawong on all channel + dtype combinations
            >>> im3 = np.random.rand(128, 128, 3)
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
        # return shape of contours to openCV contours

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        # print('--- A')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))

        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8

        cv_contours = self._to_cv_countours()

        if alpha is None or alpha == 1.0:
            # image = kwimage.ensure_uint255(image)
            image = kwimage.atleast_3channels(image, copy=copy)
            rgba = kwimage.Color(color)._forimage(image)
        else:
            image = kwimage.ensure_float01(image)
            image = kwimage.ensure_alpha_channel(image)
            rgba = kwimage.Color(color, alpha=alpha)._forimage(image)

        # print('--- B')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))
        # print('rgba = {!r}'.format(rgba))

        if fill:
            if alpha is None or alpha == 1.0:
                # Modification happens inplace
                image = cv2.fillPoly(image, cv_contours, rgba, line_type, shift=0)
            else:
                orig = image.copy()
                mask = np.zeros_like(orig)
                mask = cv2.fillPoly(mask, cv_contours, rgba, line_type, shift=0)
                # TODO: could use add weighted
                image = kwimage.overlay_alpha_images(mask, orig)
                rgba = kwimage.Color(rgba)._forimage(image)

        # print('--- C')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))
        # print('rgba = {!r}'.format(rgba))

        if border or True:
            thickness = 4
            contour_idx = -1
            image = cv2.drawContours(image, cv_contours, contour_idx, rgba,
                                     thickness, line_type)
        # image = kwimage.ensure_float01(image)[..., 0:3]

        # print('--- D')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))

        image = dtype_fixer(image, copy=False)
        return image

    def draw(self, color='blue', ax=None, alpha=1.0, radius=1, setlim=False,
             border=False, linewidth=2):
        """
        Draws polygon in a matplotlib axes. See `draw_on` for in-memory image
        modification.

        Args:
            setlim (bool): if True ensures the limits of the axes contains the
                polygon

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1)
            >>> self = self.scale(100)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw()
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw(setlim=True)
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        import kwimage
        if ax is None:
            ax = plt.gca()

        color = list(kwimage.Color(color).as01())

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

        kw = {}
        # TODO:
        # depricate border kwarg in favor of standard matplotlib args
        if border:
            kw['linewidth'] = linewidth
            try:
                edgecolor = list(kwimage.Color(border).as01())
            except Exception:
                edgecolor = list(color)
                # hack to darken
                edgecolor[0] -= .1
                edgecolor[1] -= .1
                edgecolor[2] -= .1
                edgecolor = [min(1, max(0, c)) for c in edgecolor]
            kw['edgecolor'] = edgecolor
        else:
            kw['linewidth'] = 0

        patch = mpl.patches.PathPatch(path, alpha=alpha, facecolor=color, **kw)
        ax.add_patch(patch)

        if setlim:
            x1, y1, x2, y2 = self.to_boxes().to_ltrb().data[0]

            if setlim == 'grow':
                # only allow growth
                x1_, x2_ = ax.get_xlim()
                x1 = min(x1_, x1)
                x2 = max(x2_, x2)

                y1_, y2_ = ax.get_ylim()
                y1 = min(y1_, y1)
                y2 = max(y2_, y2)

            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
        return patch

    def _ensure_vertex_order(self, inplace=False):
        """
        Fixes vertex ordering so the exterior ring is CCW and the interior rings
        are CW.

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random(n=3, n_holes=2, rng=0)
            >>> print('self = {!r}'.format(self))
            >>> new = self._ensure_vertex_order()
            >>> print('new = {!r}'.format(new))

            >>> self = kwimage.Polygon.random(n=3, n_holes=2, rng=0).swap_axes()
            >>> print('self = {!r}'.format(self))
            >>> new = self._ensure_vertex_order()
            >>> print('new = {!r}'.format(new))
        """
        new = self if inplace else self.__class__(self.data.copy())

        exterior = new.data['exterior']

        if _is_clockwise(exterior.data):
            # ensure exterior is CCW
            exterior.data = exterior.data[::-1]
            pass

        for interior in new.data['interiors']:
            if not _is_clockwise(interior.data):
                interior.data = interior.data[::-1]
        return new


def _is_clockwise(verts):
    """
    References:
        https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order#:~:text=If%20the%20determinant%20is%20negative,q%20and%20r%20are%20collinear.

    Ignore:
        verts = poly.data['exterior'].data[::-1]
    """
    cross_product = np.cross(verts[:-1], verts[1:])
    is_clockwise = cross_product.sum() > 0
    return is_clockwise


def _order_vertices(verts):
    """
    References:
        https://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise

    Ignore:
        verts = poly.data['exterior'].data[::-1]
    """
    mean_x = verts.T[0].sum() / len(verts)
    mean_y = verts.T[1].sum() / len(verts)

    delta_x = mean_x - verts.T[0]
    delta_y = verts.T[1] - mean_y

    tau = np.pi * 2
    angle = (np.arctan2(delta_x, delta_y) + tau) % tau
    sortx = angle.argsort()
    verts = verts.take(sortx, axis=0)
    return verts


class MultiPolygon(_generic.ObjectList):
    """
    Data structure for storing multiple polygons (typically related to the same
    underlying but potentitally disjoing object)

    Attributes:
        data (List[Polygon])
    """

    @classmethod
    def random(self, n=3, n_holes=0, rng=None, tight=False):
        """
        Create a random MultiPolygon

        Returns:
            MultiPolygon
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        data = [Polygon.random(rng=rng, n_holes=n_holes, tight=tight)
                for _ in range(n)]
        self = MultiPolygon(data)
        return self

    def fill(self, image, value=1):
        """
        Inplace fill in an image based on this multi-polyon.

        Args:
            image (ndarray): image to draw on (inplace)
            value (int | Tuple[int], default=1): value fill in with

        Returns:
            ndarray: the image that has been modified in place
        """
        for p in self.data:
            p.fill(image, value=value)
        return image

    def to_multi_polygon(self):
        return self

    def to_boxes(self):
        """
        Deprecated: lossy conversion use 'bounding_box' instead
        """
        return self.bounding_box()

    def bounding_box(self):
        """
        Return the bounding box of the multi polygon

        Returns:
            kwimage.Boxes: a Boxes object with one box that encloses all
                polygons

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = MultiPolygon.random(rng=0, n=10)
            >>> boxes = self.to_boxes()
            >>> sub_boxes = [d.to_boxes() for d in self.data]
            >>> areas1 = np.array([s.intersection(boxes).area[0] for s in sub_boxes])
            >>> areas2 = np.array([s.area[0] for s in sub_boxes])
            >>> assert np.allclose(areas1, areas2)
        """
        import kwimage
        lt = np.array([np.inf, np.inf])
        rb = np.array([-np.inf, -np.inf])
        for data in self.data:
            xys = data.data['exterior'].data
            lt = np.minimum(lt, xys.min(axis=0))
            rb = np.maximum(rb, xys.max(axis=0))
        ltrb = np.hstack([lt, rb])[None, :]
        boxes = kwimage.Boxes(ltrb, 'ltrb')
        return boxes

    def to_mask(self, dims=None):
        """
        Returns a mask object indication regions occupied by this multipolygon

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
        if dims is None:
            raise ValueError('Must specify output raster dimensions')
        c_mask = np.zeros(dims, dtype=np.uint8)
        for p in self.data:
            if p is not None:
                p.fill(c_mask, value=1)
        mask = kwimage.Mask(c_mask, 'c_mask')
        return mask

    def to_relative_mask(self):
        """
        Returns a translated mask such the mask dimensions are minimal.

        In other words, we move the polygon all the way to the top-left and
        return a mask just big enough to fit the polygon.

        Returns:
            Mask
        """
        x, y, w, h = self.to_boxes().quantize().to_xywh().data[0]
        mask = self.translate((-x, -y)).to_mask(dims=(h, w))
        return mask

    @classmethod
    def coerce(cls, data, dims=None):
        """
        Attempts to construct a MultiPolygon instance from the input data

        See Mask.coerce
        """
        from kwimage.structs.segmentation import _coerce_coco_segmentation
        self = _coerce_coco_segmentation(data, dims=dims)
        if self is not None:
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

    @classmethod
    def from_shapely(MultiPolygon, geom):
        """
        Convert a shapely polygon or multipolygon to a kwimage.MultiPolygon
        """
        if geom.type == 'Polygon':
            polys = [Polygon.from_shapely(geom)]
        else:
            polys = [Polygon.from_shapely(g) for g in geom.geoms]
        self = MultiPolygon(polys)
        return self

    @classmethod
    def from_geojson(MultiPolygon, data_geojson):
        """
        Convert a geojson polygon or multipolygon to a kwimage.MultiPolygon

        Example:
            >>> import kwimage
            >>> orig = kwimage.MultiPolygon.random()
            >>> data_geojson = orig.to_geojson()
            >>> self = kwimage.MultiPolygon.from_geojson(data_geojson)
        """
        if data_geojson['type'] == 'Polygon':
            polys = [Polygon.from_geojson(data_geojson)]
        else:
            polys = [
                Polygon.from_geojson(
                    {'type': 'Polygon', 'coordinates': coords})
                for coords in data_geojson['coordinates']
            ]
        self = MultiPolygon(polys)
        return self

    def to_geojson(self):
        """
        Converts polygon to a geojson structure
        """
        coords = [poly.to_geojson()['coordinates'] for poly in self.data]
        data_geojson = {
            'type': 'MultiPolygon',
            'coordinates': coords,
        }
        return data_geojson

    @classmethod
    def from_coco(cls, data, dims=None):
        """
        Accepts either new-style or old-style coco multi-polygons
        """
        if isinstance(data, list):
            poly_list = [Polygon.from_coco(item, dims=dims)
                         for item in data]
            self = cls(poly_list)
        else:
            raise TypeError(type(data))
        return self

    def _to_coco(self, style='orig'):
        return self.to_coco(style=style)

    def to_coco(self, style='orig'):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = MultiPolygon.random(1, rng=0)
            >>> self.to_coco()
        """
        return [item.to_coco(style=style) for item in self.data]

    def swap_axes(self, inplace=False):
        return self.apply(lambda item: item.swap_axes(inplace=inplace))

    # def draw_on(self, image, color='blue', fill=True, border=False, alpha=1.0):
    #     """
    #     Faster version
    #     """
    #     import kwimage
    #     dtype_fixer = _generic._consistent_dtype_fixer(image)

    #     if alpha == 1.0:
    #         image = kwimage.ensure_uint255(image)
    #         image = kwimage.atleast_3channels(image)
    #         rgba = kwimage.Color(color).as255()
    #     else:
    #         image = kwimage.ensure_float01(image)
    #         image = kwimage.ensure_alpha_channel(image)
    #         rgba = kwimage.Color(color, alpha=alpha).as01()

    #     kwargs = dict(color=color, fill=fill, border=border, alpha=alpha)

    #     for item in self.data:
    #         if item is not None:
    #             image = item.draw_on(image=image, **kwargs)

    #     image = dtype_fixer(image)
    #     return image


class PolygonList(_generic.ObjectList):
    """
    Stores and allows manipluation of multiple polygons, usually within the
    same image.
    """

    def to_mask_list(self, dims=None):
        """
        Converts all items to masks
        """
        import kwimage
        new = kwimage.MaskList([
            None if item is None else item.to_mask(dims=dims)
            for item in self
        ])
        return new

    def to_polygon_list(self):
        return self

    def to_segmentation_list(self):
        """
        Converts all items to segmentation objects
        """
        import kwimage
        new = kwimage.SegmentationList([
            None if item is None else kwimage.Segmentation.coerce(item)
            for item in self
        ])
        return new

    def swap_axes(self, inplace=False):
        return self.apply(lambda item: item.swap_axes(inplace=inplace))

    def to_geojson(self, as_collection=False):
        """
        Converts a list of polygons/multipolygons to a geojson structure

        Args:
            as_collection (bool): if True, wraps the polygon geojson items in a
                geojson feature collection, otherwise just return a list of
                items.

        Returns:
            List[Dict] | Dict: items or geojson data

        Example:
            >>> import kwimage
            >>> data = [kwimage.Polygon.random(),
            >>>         kwimage.Polygon.random(n_holes=1),
            >>>         kwimage.MultiPolygon.random(n_holes=1),
            >>>         kwimage.MultiPolygon.random()]
            >>> self = kwimage.PolygonList(data)
            >>> geojson = self.to_geojson(as_collection=True)
            >>> items = self.to_geojson(as_collection=False)
            >>> print('geojson = {}'.format(ub.repr2(geojson, nl=-2, precision=1)))
            >>> print('items = {}'.format(ub.repr2(items, nl=-2, precision=1)))
        """
        items = [poly.to_geojson() for poly in self.data]
        if as_collection:
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": item,
                        "properties": {}
                    }
                    for item in items
                ]
            }
            return geojson
        else:
            return items
