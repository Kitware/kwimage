"""
A class to represent a polygon (and a MultiPolygon)

Defines :class:`Polygon` and :class:`MultiPolygon`

TODO:
    - [ ]  Make function mask -> polygon list
    - [ ]  Make function multipolygon -> polygon list
    - [ ]  Make function PolygonList -> Boxes
    - [ ]  Make function SegmentationList -> Boxes
    - [ ] First class shapely support (format='shapely' to mitigate format conversion cost) (or use shapely as the primary format).

"""
import cv2
import skimage
import numbers
import ubelt as ub
import numpy as np
from kwimage.structs import _generic
# from . import _generic

try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile


class _ShapelyMixin:
    """
    TODO: make shapely the main "format" to reduce conversion cost

    References:
        - [WikiBoolPolygon] https://en.wikipedia.org/wiki/Boolean_operations_on_polygons
        - [WikiDe91M] https://en.wikipedia.org/wiki/DE-9IM

    Example:
        >>> from kwimage.structs.polygon import *  # NOQA
        >>> import itertools as it
        >>> import kwimage
        >>> poly1 = kwimage.Polygon.random()
        >>> poly2 = kwimage.Polygon.random()
        >>> mpoly1 = kwimage.MultiPolygon.random()
        >>> mpoly2 = kwimage.MultiPolygon.random().buffer(0)
        >>> for self, other in it.combinations([poly1, poly2, mpoly1, mpoly2], 2):
        >>>     self.iou(other)
        >>>     self.iooa(other)
        >>>     self.intersection(other)
        >>>     self.union(other)
        >>>     self.difference(other)
        >>>     self.symmetric_difference(other)
        >>>     self.oriented_bounding_box()
    """

    def oriented_bounding_box(self):
        """
        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random().scale(100, 100).round()
            >>> obox = self.oriented_bounding_box()
            >>> print(f'obox={obox}')
        """
        import cv2
        from collections import namedtuple
        OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'theta'))
        hull = self.convex_hull
        cv2_xy = hull.exterior.data.astype(np.float32)
        center, extent, angle = cv2.minAreaRect(cv2_xy)
        theta = np.deg2rad(angle)
        obox = OrientedBBox(center, extent, theta)
        return obox

    def buffer(self, *args, **kwargs):
        a = self.to_shapely()
        r = a.buffer(*args, **kwargs)
        return _kwimage_from_shapely(r)

    def simplify(self, tolerance, preserve_topology=True):
        a = self.to_shapely()
        r = a.simplify(tolerance, preserve_topology=preserve_topology)
        return _kwimage_from_shapely(r)

    @property
    def __geo_interface__(self):
        """
        Geometry interface standardized in GeoInterface_.

        References:
            .. [GeoInterface] https://gist.github.com/sgillies/2217756

        Example:
            >>> import kwimage
            >>> x = kwimage.Polygon.random()
            >>> geos = x.__geo_interface__
            >>> # xdoctest: +REQUIRES(module:geopandas)
            >>> import geopandas as gpd
            >>> # This allows kwimage Polygons to work with geopandas seemlessly
            >>> gpd.GeoDataFrame({'geometry': [x]})
        """
        return self.to_shapely().__geo_interface__

    # area
    # crosses
    # disjoint
    # distance
    # empty
    # envelope
    # equals
    # almost_equals
    # contains
    # interpolate (conflict)
    # intersects
    # within
    # touches
    # simplify
    # representative_point
    # project
    # relate
    # overlaps
    # normalize
    # minimum_clearance
    # minimum_rotated_rectangle (i.e. oriented_bounding_box)
    # minimum_rotated_rectangle (i.e. oriented_bounding_box)
    # is_valid, is_closed, is_empty, is_ring, is_simple

    # https://shapely.readthedocs.io/en/stable/manual.html#set-theoretic-methods

    def union(self, other):
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        c = a.intersection(b)
        return _kwimage_from_shapely(c)

    def intersection(self, other):
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        c = a.intersection(b)
        return _kwimage_from_shapely(c)

    def difference(self, other):
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        c = a.difference(b)
        return _kwimage_from_shapely(c)

    def symmetric_difference(self, other):
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        c = a.symmetric_difference(b)
        return _kwimage_from_shapely(c)

    # ----

    def iooa(self, other):
        """
        Intersection over other area
        """
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        isect = a.intersection(b)
        iooa = isect.area / b.area
        return iooa

    def iou(self, other):
        """
        Intersection area over union area
        """
        a, b = self.to_shapely(fix=1), other.to_shapely(fix=1)
        isect = a.intersection(b)
        union = a.union(b)
        iou = isect.area / union.area
        return iou

    # --- Properties

    @property
    def area(self):
        """
        Computes area via shapley conversion

        Returns:
            float
        """
        return self.to_shapely().area

    @property
    def convex_hull(self):
        a = self.to_shapely()
        r = a.convex_hull
        return _kwimage_from_shapely(r)

    def is_invalid(self, explain=False):
        """
        Return True if the polygon is invalid according to shapely.

        Args:
            explain (bool): if True, the return value is a string
                explaining why the polygon is invalid.

        Returns:
            bool | str: Always returns False if the polygon is valid.
                Returns True or a string if the polygon is invalid according to
                shapely.
        """
        a = self.to_shapely()
        if a.is_valid:
            return False
        elif explain:
            from shapely import validation
            return validation.explain_validity(a)
        else:
            return True

    def fix(self):
        """
        Attempt to ensure validity

        References:
            https://stackoverflow.com/questions/20833344/fix-invalid-polygon-in-shapely
        """
        # from shapely.geometry.base import geom_factory
        # from shapely.geos import lgeos
        from shapely.validation import make_valid
        a = self.to_shapely()
        if not a.is_valid:
            a = make_valid(a)
            # a = geom_factory(lgeos.GEOSMakeValid(a._geom))
        return _kwimage_from_shapely(a)


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

    def _warp_imgaug(self, augmenter, input_dims, inplace=False):
        """
        Warps by applying an augmenter from the imgaug library

        Args:
            augmenter (imgaug.augmenters.Augmenter):
            input_dims (Tuple): h/w of the input image
            inplace (bool): if True, modifies data inplace

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
            transform (GeometricTransform | ArrayLike | Augmenter | callable):
                scikit-image tranform, a 3x3 transformation matrix,
                an imgaug Augmenter, or generic callable which transforms
                an NxD ndarray.

            input_dims (Tuple): shape of the image these objects correspond to
                (only needed / used when transform is an imgaug augmenter)

            output_dims (Tuple): unused, only exists for compatibility

            inplace (bool): if True, modifies data inplace

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
        if transform is None:
            return new
        elif not isinstance(transform, (np.ndarray, skimage.transform._geometric.GeometricTransform)):
            try:
                import imgaug
            except ImportError:
                pass
                # import warnings
                # warnings.warn('imgaug is not installed')
                # raise TypeError(type(transform))
            else:
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

    def scale(self, factor, about=None, output_dims=None, inplace=False):
        """
        Scale a polygon by a factor

        Args:
            factor (float | Tuple[float, float]):
                scale factor as either a scalar or a (sf_x, sf_y) tuple.

            about (Tuple | None):
                if unspecified scales about the origin (0, 0), otherwise the
                scaling is about this point. Can be "centroid" and will use
                centroid of polygon Using "ymin,xmin" will be the
                topmost,leftmost point on the polygon (wrt image coords).
                See :func:`_PolyWarpMixin._rectify_about` for details bout
                codes.

            output_dims (Tuple): unused in non-raster spatial structures

            inplace (bool): if True, modifies data inplace

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> new = self.scale(10)

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0).translate((0.5))
            >>> new = self.scale(1.5, about='centroid')
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

    def translate(self, offset, output_dims=None, inplace=False):
        """
        Shift the polygon up/down left/right

        Args:
            factor (float | Tuple[float]):
                transation amount as either a scalar or a (t_x, t_y) tuple.
            output_dims (Tuple): unused in non-raster spatial structures
            inplace (bool): if True, modifies data inplace

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

        Args:
            about (str | Tuple): either a numeric coordinate or a
                string code that specifies one. Valid string codes are
                    * "origin" - maps to (0, 0)
                    * "centroid" - maps to the polygon centroid
                    * "center" - alias of centroid
                    * "ymin,xmin-bound" - the top-left (img coords) of the polygon bounding box
                    * "ymin,xmax-bound" - the top-right (img coords) of the polygon bounding box
                    * "ymax,xmin-bound" - the bottom-right (img coords) of the polygon bounding box
                    * "ymax,xmax-bound" - the bottom-left (img coords) of the polygon bounding box
                    * A comma separated code illustrated in the TextArt adapted from [SO67822179]_.

        Returns:
            Tuple[int, int] - the x, y about position

        References:
            .. [SO67822179] https://stackoverflow.com/questions/67822179/poly-topleft-points

        TextArt:

            (0, 0)
            + ---------> +x
            |
            |    ymin,left ─────────►xxxxxxxxxx◄──────── ymin,xmax
            |                     xxxx         xx
            V    xmin,ymin ────►xxx             xx
            y+                  x                xx◄──── xmax,ymin
                                x                 x
                 xmin,ymax ────►xx                x
                                 xx              xx◄──── xmax,ymax
                                  x             xx
                 ymax,left ──────►xxxxxxxxxxxxxxx◄────── ymax,xmax

        Example:
            >>> import kwimage
            >>> mask = kwimage.Mask.from_text(ub.codeblock(
            >>>     '''
            >>>           xxxxxxxxxx
            >>>        xxxx         xx
            >>>      xxx             xx
            >>>      x                xx
            >>>      x                 x
            >>>      xx                x
            >>>       xx              xx
            >>>        x             xx
            >>>        xxxxxxxxxxxxxxx
            >>>    '''), zero_chr=' ')
            >>> poly = mask.to_multi_polygon().data[0]
            >>> print(poly._rectify_about('ymin,xmin'))
            [5 0]
            >>> print(poly._rectify_about('xmin,ymin'))
            [0 2]
            >>> print(poly._rectify_about('xmin,ymax'))
            [0 5]
            >>> print(poly._rectify_about('ymax,xmin'))
            [2 8]
            >>> print(poly._rectify_about('ymin,xmax'))
            [14  0]
            >>> print(poly._rectify_about('xmax,ymin'))
            [18  3]
            >>> print(poly._rectify_about('xmax,ymax'))
            [18  6]
            >>> print(poly._rectify_about('ymax,xmax'))
            [16  8]

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0).scale(10).round().astype(np.int32)
            >>> print(self._rectify_about('centroid'))
            >>> print(self._rectify_about('ymax,xmin'))
            >>> print(self._rectify_about('xmin,ymax'))
            >>> print(self._rectify_about('ymax,xmin-bounds'))
            >>> print(self._rectify_about('xmin,ymax-bounds'))
            (4.325, 3.9)
            [5 8]
            [1 6]
            [1 8]
            [1 8]
        """
        if about is None:
            about_ = None
        else:
            if isinstance(about, str):
                if about == 'origin':
                    about_ = (0., 0.)
                elif about in {'center', 'centroid'}:
                    centroid = self.to_shapely().centroid
                    about_ = (centroid.x, centroid.y)
                else:
                    # NOTE: We may want to generalize this to keypoints /
                    # coordinates as well (or at least keypoints, not sure
                    # about coordinates, i.e. the notion of a top-left doesnt
                    # make sense for general ND-points), but punting on this
                    # for now and just putting it where it is immediately
                    # needed.
                    import parse
                    pattern1 = parse.Parser('{dir1},{dir2}-{qualifier}')
                    pattern2 = parse.Parser('{dir1},{dir2}')
                    found = pattern1.parse(about) or pattern2.parse(about)
                    if found is None:
                        raise KeyError('Unknown code about={}'.format(about))
                    parts = found.named
                    dir1 = parts.get('dir1')
                    dir2 = parts.get('dir2')

                    qualifier = parts.get('qualifier', 'poly')
                    if qualifier == 'bounds':
                        points = self.to_boxes().to_polygons()[0].exterior.data
                    elif qualifier == 'poly':
                        points = self.exterior.data
                    else:
                        raise KeyError(
                            'Unknown qualifier={} in about={}'.format(
                                qualifier, about))

                    def dir_to_axis_and_extremum(dir_):
                        """
                        extremuf is a factor where the minimum of the data
                        multiplied by this factor will be the desired extreme:
                            1 for min and -1 for max.
                        """
                        if dir_ == 'ymax':
                            axis, extremf = 1, -1
                        elif dir_ == 'ymin':
                            axis, extremf = 1, +1
                        elif dir_ == 'xmin':
                            axis, extremf = 0, +1
                        elif dir_ == 'xmax':
                            axis, extremf = 0, -1
                        else:
                            raise KeyError
                        return axis, extremf

                    try:
                        axis1, extremf1 = dir_to_axis_and_extremum(dir1)
                    except KeyError:
                        raise KeyError(
                            'Unknown dir1={} in about={}'.format(dir1, about))
                    try:
                        axis2, extremf2 = dir_to_axis_and_extremum(dir2)
                    except KeyError:
                        raise KeyError(
                            'Unknown dir1={} in about={}'.format(dir1, about))

                    if axis2 == axis1:
                        raise ValueError((
                            'Specified directions in about={} '
                            'cannot be on the same axis').format(about))

                    extremuf_ = np.array([[extremf1, extremf2]])

                    pt = min((points[:, [axis1, axis2]] * extremuf_).tolist())
                    about_ = (np.array(pt) * extremuf_[0])[[axis1, axis2]]
            else:
                about_ = about if ub.iterable(about) else [about] * 2
        return about_

    def round(self, decimals=0, inplace=False):
        """
        Rounds data to the specified decimal place.
        This may make the polygon invalid.

        Args:
            inplace (bool): if True, modifies this object
            decimals (int): number of decimal places to round to

        Returns:
            Polygon: modified polygon

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random(3).scale(10)
            >>> new = self.round()
            >>> assert np.any(self.exterior.data != new.exterior.data)
            >>> assert np.all(self.exterior.data.round() == new.exterior.data)
            >>> # demo a case that makes the polygon invalid
            >>> self = kwimage.Polygon.random(6).scale(0.1)
            >>> new = self.round()
            >>> assert np.any(self.exterior.data != new.exterior.data)
            >>> assert np.all(self.exterior.data.round() == new.exterior.data)
        """
        new = self if inplace else self.copy()
        # new.data['exterior'].round(decimals=decimals, inplace=True)
        # for hole in new.data['interiors']:
        #     hole.round(decimals=decimals, inplace=True)
        # print(f'inplace={inplace}')
        new.data['exterior'] = new.data['exterior'].round(decimals=decimals, inplace=inplace)
        new.data['interiors'][:] = [
            hole.round(decimals=decimals, inplace=inplace)
            for hole in new.data['interiors']
        ]
        return new

    def astype(self, dtype, inplace=False):
        """
        Changes the data type

        Args:
            dtype : new type
            inplace (bool): if True, modifies this object

        Returns:
            Polygon: modified polygon

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random(3, rng=0).scale(10)
            >>> new = self.astype(np.int32)
            >>> assert np.any(self.exterior.data != new.exterior.data)
            >>> assert np.all(self.exterior.data.astype(np.int32) == new.exterior.data)
        """
        new = self if inplace else self.copy()
        new.data['exterior'] = new.data['exterior'].astype(dtype, inplace=inplace)
        new.data['interiors'][:] = [
            hole.astype(dtype, inplace=inplace)
            for hole in new.data['interiors']
        ]
        return new

    def swap_axes(self, inplace=False):
        """
        Swap the x and y coordinate axes

        Args:
            inplace (bool): if True, modifies data inplace

        Returns:
            Polygon: modified polygon
        """
        new = self if inplace else self.copy()
        new.data['exterior'] = new.data['exterior'].reorder_axes(
            (1, 0), inplace=inplace)
        new.data['interiors'] = [
            p.reorder_axes((1, 0), inplace=inplace)
            for p in new.data['interiors']]
        return new


class Polygon(_generic.Spatial, _PolyArrayBackend, _PolyWarpMixin, _ShapelyMixin, ub.NiceRepr):
    """
    Represents a single polygon as set of exterior boundary points and a list
    of internal polygons representing holes.

    By convention exterior boundaries should be counterclockwise and interior
    holes should be clockwise.

    Example:
        >>> import kwimage
        >>> poly1 = kwimage.Polygon(exterior=[[ 5., 10.], [ 1.,  8.], [ 3.,  4.], [ 5.,  3.], [ 8.,  9.], [ 6., 10.]])
        >>> poly2 = kwimage.Polygon.random(rng=34214, n_holes=2).scale(10).round()
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=True)
        >>> poly1.draw(setlim=1.4, vertex=0.2, vertexcolor='kw_orange', color='kw_blue', edgecolor='kw_green', alpha=0.5)
        >>> poly2.draw(setlim=2.9, vertex=0.2, vertexcolor='kw_red', color='kw_darkgreen', edgecolor='kw_darkblue', alpha=0.5)
        >>> kwplot.show_if_requested()

    Example:
        >>> import kwimage
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [
        >>>         np.array([[14, 13], [15, 12], [23, 12], [24, 13], [24, 18],
        >>>                   [23, 19], [13, 19], [12, 18]]),
        >>>         np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11],
        >>>                   [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = kwimage.Polygon(**data)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw(setlim=1.4, vertex=0.2, vertexcolor='kw_orange', color='kw_blue', edgecolor='kw_green')

    Example:
        >>> import kwimage
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [
        >>>         np.array([[14, 13], [15, 12], [23, 12], [24, 13], [24, 18],
        >>>                   [23, 19], [13, 19], [12, 18]]),
        >>>         np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11],
        >>>                   [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = kwimage.Polygon(**data)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw(setlim=1.4, vertex=0.2, vertexcolor='kw_orange', color='kw_blue', edgecolor='kw_green')

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

    Example:
        >>> # Test empty polygon
        >>> import kwimage
        >>> data = {
        >>>     'exterior': np.array([]),
        >>>     'interiors': [],}
        >>> self = kwimage.Polygon(**data)
        >>> geos = self.to_geojson()
        >>> kwimage.Polygon.from_geojson(geos)
        >>> geom = self.to_shapely()
        >>> kwimage.Polygon.from_shapely(geom)
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

        # TODO: Add format option where format can be dict, or shapley
        self.data = data
        self.meta = meta

    @property
    def exterior(self):
        """
        Returns:
            kwimage.Coords
        """
        # if self.format = 'dict':
        # if self.format = 'shapely':
        # self.data.exterior.coords
        return self.data['exterior']

    @property
    def interiors(self):
        """
        Returns:
            List[kwimage.Coords]
        """
        # if self.format = 'dict':
        # if self.format = 'shapely':
        # [d.coords for d in z.interiors]
        return self.data['interiors']

    def __nice__(self):
        """
        Returns:
            str
        """
        return ub.repr2(self.data, nl=1)

    @classmethod
    def circle(cls, xy, r, resolution=64):
        """
        Create a circular or elliptical polygon.

        Might rename to ellipse later?

        Args:
            xy (Iterable[Number]): x and y center coordinate
            r (Number | Tuple[Number, Number]):
                circular radius or major and minor elliptical radius
            resolution (int): number of sides

        Returns:
            Polygon

        Example:
            >>> import kwimage
            >>> xy = (0.5, 0.5)
            >>> r = .3
            >>> # Demo with circle
            >>> circle = kwimage.Polygon.circle(xy, r, resolution=6)
            >>> # Demo with ellipse
            >>> xy = (0.5, 0.5)
            >>> r = (.4, .7)
            >>> ellipse1 = kwimage.Polygon.circle(xy, r, resolution=12)
            >>> ellipse2 = kwimage.Polygon.circle(xy, (.7, .4), resolution=12)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> circle.draw(setlim=True, border=1, fill=0, color='kitware_orange')
            >>> ellipse1.draw(setlim=True, border=1, fill=0, color='kitware_blue')
            >>> ellipse2.draw(setlim=True, border=1, fill=0, color='kitware_green')
            >>> plt.gca().set_xlim(-0.5, 1.5)
            >>> plt.gca().set_ylim(-0.5, 1.5)
            >>> plt.gca().set_aspect('equal')
        """
        tau = 2 * np.pi

        if ub.iterable(r):
            a, b = r
            is_circle = (a == b)
            if is_circle:
                r = a
        else:
            is_circle = True

        if is_circle:
            theta = np.linspace(0, tau, resolution)
            y_offset = np.sin(theta) * r
            x_offset = np.cos(theta) * r
        else:
            # If we have an ellipse (i.e. different radius in each direction),
            # then the problem gets a lot harder, but we can do it! WE JUST
            # GOTTA BELIEVE IN OURSELVES!
            import scipy.optimize
            import scipy
            need_swap = a > b
            if need_swap:
                a, b = b, a

            def angles_in_ellipse(num, a, b):
                """
                References:
                    https://stackoverflow.com/questions/6972331/how-can-i-generate-a-set-of-points-evenly-distributed-along-the-perimeter-of-an
                """
                assert num > 0
                assert a < b
                angles = 2 * np.pi * np.arange(num) / num
                if a != b:
                    e2 = (1.0 - a ** 2.0 / b ** 2.0)
                    tot_size = scipy.special.ellipeinc(2.0 * np.pi, e2)
                    arc_size = tot_size / num
                    arcs = np.arange(num) * arc_size
                    res = scipy.optimize.root(
                        lambda x: (scipy.special.ellipeinc(x, e2) - arcs), angles,
                        # options={'maxiter': 5}
                    )
                    angles = res.x
                return angles

            # Sample theta such that the arclengths between points are nearly
            # the same.
            theta = angles_in_ellipse(resolution, a, b)
            x_offset, y_offset = np.array((a * np.cos(theta), b * np.sin(theta)))
            if need_swap:
                x_offset, y_offset = y_offset, x_offset

        center = np.array(xy)
        xcoords = center[0] + x_offset
        ycoords = center[1] + y_offset
        exterior = np.stack([xcoords.ravel(), ycoords.ravel()], axis=1)
        self = cls(exterior=exterior)
        return self

    @classmethod
    def random(cls, n=6, n_holes=0, convex=True, tight=False, rng=None):
        """
        Args:
            n (int): number of points in the polygon (must be 3 or more)
            n_holes (int): number of holes
            tight (bool): fits the minimum and maximum points
                between 0 and 1
            convex (bool): force resulting polygon will be convex
               (may remove exterior points)

        Returns:
            Polygon

        CommandLine:
            xdoctest -m kwimage.structs.polygon Polygon.random

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random(n=4, rng=None, n_holes=2, convex=1)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> self.draw(color='kw_green', edgecolor='kw_blue', vertexcolor='kw_darkblue', vertex=0.01)

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

    def to_mask(self, dims=None, pixels_are='points'):
        """
        Convert this polygon to a mask

        TODO:
            - [ ] currently not efficient

        Args:
            dims (Tuple): height and width of the output mask
            pixels_are (str): either "points" or "areas"

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
        self.fill(c_mask, value, pixels_are=pixels_are)
        mask = kwimage.Mask(c_mask, 'c_mask')
        return mask

    def to_relative_mask(self, return_offset=False):
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
        if return_offset:
            offset = (x, y)
            return mask, offset
        else:
            return mask

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
        cv_contour_ = [np.expand_dims(c.data, axis=1) for c in coords]
        WORKAROUND_OPENCV_5473 = 1
        if WORKAROUND_OPENCV_5473:
            max_coord = (1 << 16) // 2
            for c in cv_contour_:
                if np.any(c > max_coord):
                    import warnings
                    warnings.warn('Drawing a large polygon with cv2 has bugs')
            cv_contour_ = [c.clip(-max_coord, max_coord) for c in cv_contour_]
        cv_contours = [c.astype(np.int32) for c in cv_contour_]
        return cv_contours

    @classmethod
    def coerce(Polygon, data):
        """
        Routes the input to the proper constructor

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
            >>> kwimage.Polygon.coerce(self)
            >>> kwimage.Polygon.coerce(self.exterior)
            >>> kwimage.Polygon.coerce(self.exterior.data)
            >>> kwimage.Polygon.coerce(self.data)
            >>> kwimage.Polygon.coerce(self.to_geojson())
            >>> kwimage.Polygon.coerce('POLYGON ((0.11 0.61, 0.07 0.588, 0.015 0.50, 0.11 0.61))')
        """
        # TODO: fix single list case from old coco style
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
        if len(geom.exterior.coords) == 0:
            exterior = np.empty((0, 2))
        else:
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

        Returns:
            Polygon

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
            # quick check_leftmost_depth of a nested struct
            item = data
            depth = 0
            while isinstance(item, (list, tuple)):
                if len(item) == 0:
                    return None  # empty data
                    # raise Exception('no child node')
                item = item[0]
                depth += 1
            return depth
        depth = check_leftmost_depth(coords)
        if depth is None:
            exterior = np.empty((0, 2))
            interiors = []
        elif depth == 2:
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

    def to_shapely(self, fix=False):
        """
        Args:
            fix (bool):
                if True, will check for validity and if any simple fixes
                can be applied, otherwise it returns the data as is.

        Returns:
            shapely.geometry.polygon.Polygon

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
        shell_data = self.data['exterior'].data
        if shell_data.size == 0:
            # Empty polygon
            geom = shapely.geometry.Polygon()
        else:
            geom = shapely.geometry.Polygon(
                shell=shell_data,
                holes=[c.data for c in self.data['interiors']]
            )
        if fix:
            if not geom.is_valid:
                geom = geom.buffer(0)
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

        Returns:
            str

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random()
            >>> print(self.to_wkt())
        """
        shp = self.to_shapely()
        if hasattr(shp, 'wkt'):
            return shp.wkt  # new version (~1.8a1)
        else:
            return shp.to_wkt()

    @classmethod
    def from_coco(cls, data, dims=None):
        """
        Accepts either new-style or old-style coco polygons

        Args:
            data (List[Number] | Dict):
                A new or old-style coco polygon

            dims (None | Tuple[int, ...]):
                the shape dimensions of the canvas. Unused. Exists for
                compatibility with masks.

        Returns:
            Polygon
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
        Args:
            style(str): can be "orig" or "new"

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
        """
        Returns:
            MultiPolygon
        """
        return MultiPolygon([self])

    def to_boxes(self):
        """
        Deprecated: lossy conversion use 'bounding_box' instead

        Returns:
            kwimage.Boxes
        """
        return self.bounding_box()

    @property
    def centroid(self):
        """
        Returns:
            Tuple[Number, Number]
        """
        shp_centroid = self.to_shapely().centroid
        xy = (shp_centroid.x, shp_centroid.y)
        return xy

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

        Note:
            This Polygon will be a Box, not a convex hull! Use shapely for
            convex hulls.

        Returns:
            kwimage.Polygon
        """
        new = self.bounding_box().to_polygons()[0]
        return new

    def copy(self):
        """
        Returns:
            Polygon: a copy
        """
        self2 = self.__class__(self.data.copy(), self.meta.copy())
        self2.data['exterior'] = self2.data['exterior'].copy()
        self2.data['interiors'] = [x.copy() for x in self2.data['interiors']]
        return self2

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip polygon to specified boundaries.

        Returns:
            Polygon: clipped polygon

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

    def fill(self, image, value=1, pixels_are='points'):
        """
        Inplace fill in an image based on this polyon.

        Args:
            image (ndarray): image to draw on
            value (int | Tuple[int]): value fill in with. Defaults to 1.
            pixels_are (str): either points or areas

        Returns:
            ndarray: the image that has been modified in place

        Example:
            >>> # xdoctest: +REQUIRES(module:rasterio)
            >>> import kwimage
            >>> mask = kwimage.Mask.random()
            >>> self = mask.to_multi_polygon(pixels_are='areas').data[0]
            >>> image = np.zeros_like(mask.data)
            >>> self.fill(image, pixels_are='areas')

        Example:
            >>> # Test case where there are multiple channels
            >>> import kwimage
            >>> mask = kwimage.Mask.random(shape=(4, 4), rng=0)
            >>> self = mask.to_multi_polygon()
            >>> image = np.zeros(mask.shape[0:2] + (2,))
            >>> fill_v1 = self.fill(image.copy(), value=1)
            >>> fill_v2 = self.fill(image.copy(), value=(1, 2))
            >>> assert np.all((fill_v1 > 0) == (fill_v2 > 0))
        """
        if pixels_are == 'areas':
            # rasterio hac: todo nicer organization
            from rasterio import features
            shapes = [self.translate((0.5, 0.5)).to_geojson()]
            features.rasterize(shapes, out=image, default_value=1)
        elif pixels_are == 'points':
            # line_type = cv2.LINE_AA
            cv_contours = self._to_cv_countours()
            line_type = cv2.LINE_8
            # Modification happens inplace
            if len(image.shape) == 2:
                cv2.fillPoly(image, cv_contours, value, line_type, shift=0)
            elif len(image.shape) == 3 and image.shape[2] < 4:
                if isinstance(value, numbers.Number):
                    value = (value,) * image.shape[2]
                cv2.fillPoly(image, cv_contours, value, line_type, shift=0)
            else:
                # handle bands > 3
                for bx in enumerate(range(image.shape[2])):
                    tmp = np.ascontiguousarray(image[..., bx])
                    cv2.fillPoly(tmp, cv_contours, value, line_type, shift=0)
                    image[..., bx] = tmp

        return image

    @profile
    def draw_on(self, image, color='blue', fill=True, border=False, alpha=1.0,
                edgecolor=None, facecolor=None, copy=False):
        """
        Rasterizes a polygon on an image. See `draw` for a vectorized
        matplotlib version.

        Args:
            image (ndarray): image to raster polygon on.

            color (str | tuple): data coercable to a color

            fill (bool): draw the center mass of the polygon.
                Note: this will be deprecated. Use facecolor instead.

            border (bool): draw the border of the polygon
                Note: this will be deprecated. Use edgecolor instead.

            alpha (float): polygon transparency (setting alpha < 1
                makes this function much slower). Defaults to 1.0

            copy (bool): if False only copies if necessary

            edgecolor (str | tuple): color for the border

            facecolor (str | tuple): color for the fill

        Returns:
            np.ndarray

        Note:
            This function will only be inplace if alpha=1.0 and the input has 3
            or 4 channels. Otherwise the output canvas is coerced so colors can
            be drawn on it. In the case where alpha < 1.0,

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1).scale(128)
            >>> image_in = np.zeros((128, 128), dtype=np.float32)
            >>> image_out = self.draw_on(image_in)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(image_out, fnum=1)

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # Demo drawing on a RGBA canvas
            >>> # If you initialize an zero rgba canvas, the alpha values are
            >>> # filled correctly.
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> s = 16
            >>> self = Polygon.random(n_holes=1, rng=32).scale(s)
            >>> image_in = np.zeros((s, s, 4), dtype=np.float32)
            >>> image_out = self.draw_on(image_in, color='black')
            >>> assert np.all(image_out[..., 0:3] == 0)
            >>> assert not np.all(image_out[..., 3] == 1)
            >>> assert not np.all(image_out[..., 3] == 0)

        Example:
            >>> import kwimage
            >>> color = 'blue'
            >>> self = kwimage.Polygon.random(n_holes=1).scale(128)
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> # Test drawing on all channel + dtype combinations
            >>> im3 = np.random.rand(128, 128, 3)
            >>> im_chans = {
            >>>     'im3': im3,
            >>>     'im1': kwimage.convert_colorspace(im3, 'rgb', 'gray'),
            >>>     #'im0': im3[..., 0],
            >>>     'im4': kwimage.convert_colorspace(im3, 'rgb', 'rgba'),
            >>> }
            >>> inputs = {}
            >>> for k, im in im_chans.items():
            >>>     inputs[k + '_f01'] = (kwimage.ensure_float01(im.copy()), {'alpha': None})
            >>>     inputs[k + '_u255'] = (kwimage.ensure_uint255(im.copy()), {'alpha': None})
            >>>     inputs[k + '_f01_a'] = (kwimage.ensure_float01(im.copy()), {'alpha': 0.5})
            >>>     inputs[k + '_u255_a'] = (kwimage.ensure_uint255(im.copy()), {'alpha': 0.5})
            >>> # Check cases when image is/isnot written inplace Construct images
            >>> # with different dtypes / channels and run a draw_on with different
            >>> # keyword args.  For each combination, demo if that results in an
            >>> # implace operation or not.
            >>> rows = []
            >>> outputs = {}
            >>> for k, v in inputs.items():
            >>>     im, kw = v
            >>>     outputs[k] = self.draw_on(im, color=color, **kw)
            >>>     inplace = outputs[k] is im
            >>>     rows.append({'key': k, 'inplace': inplace})
            >>> # xdoc: +REQUIRES(module:pandas)
            >>> import pandas as pd
            >>> df = pd.DataFrame(rows).sort_values('inplace')
            >>> print(df.to_string())
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=2, doclf=True)
            >>> kwplot.autompl()
            >>> pnum_ = kwplot.PlotNums(nCols=2, nRows=len(inputs))
            >>> for k in inputs.keys():
            >>>     kwplot.imshow(inputs[k][0], fnum=2, pnum=pnum_(), title=k)
            >>>     kwplot.imshow(outputs[k], fnum=2, pnum=pnum_(), title=k)
            >>> kwplot.show_if_requested()

        Example:
            >>> # Test empty polygon draw
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.from_coco([])
            >>> image_in = np.zeros((128, 128), dtype=np.float32)
            >>> image_out = self.draw_on(image_in)

        Example:
            >>> # Test stupid large polygon draw
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> from kwimage.structs.polygon import _generic
            >>> import kwimage
            >>> self = kwimage.Polygon.random().scale(2e11)
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> image_out = self.draw_on(image)

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(kwimage.Polygon.draw_on))
        """
        import kwimage

        is_empty = len(self.data['exterior']) == 0
        if is_empty:
            if copy:
                image = image.copy()
            return image

        # Note: opencv#5473
        # https://github.com/opencv/opencv/issues/5473
        # https://stackoverflow.com/questions/37392128/wrong-result-using-function-fillpoly-in-opencv-for-very-large-images
        # There is a bug where polygons do not draw correctly over the size
        # of 2 ** 16

        # return shape of contours to openCV contours
        dtype_fixer = _generic._consistent_dtype_fixer(image)

        # print('--- A')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))

        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8

        cv_contours = self._to_cv_countours()

        if alpha == 1.0:
            alpha = None

        if alpha is None:
            # image = kwimage.ensure_uint255(image)
            image = kwimage.atleast_3channels(image, copy=copy)
        else:
            image = kwimage.ensure_float01(image)
            image = kwimage.ensure_alpha_channel(image)

        from kwimage.im_cv2 import _cv2_imputation
        image = _cv2_imputation(image)

        color = kwimage.Color(color, alpha=alpha)._forimage(image)
        # print('--- B')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))
        # print('rgba = {!r}'.format(rgba))

        if facecolor is None:
            if fill:
                facecolor = color
        elif facecolor is True:
            facecolor = color
        else:
            facecolor = kwimage.Color(facecolor, alpha=alpha)._forimage(image)

        if fill:
            if alpha is None or alpha == 1.0:
                # Modification happens inplace
                # NOTE: This takes a very long time if contours have
                # large coordinates (even if the image is small)
                image = cv2.fillPoly(image, cv_contours, facecolor, line_type, shift=0)
            else:
                # FIXME: This is very slow when there are a lot of polygons to
                # draw. An alternative is to draw all polygons on an empty
                # canvas and then blend that canvas with the original. The
                # downside is that the polygons wont blend together.
                # This logic needs to happen outside of this scope at the
                # PolygonList level.
                orig = image.copy()
                mask = np.zeros_like(orig)
                mask = cv2.fillPoly(mask, cv_contours, facecolor, line_type, shift=0)
                # TODO: could use add weighted
                image = kwimage.overlay_alpha_images(mask, orig)
                # facecolor = kwimage.Color(facecolor)._forimage(image)

        # print('--- C')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))
        # print('rgba = {!r}'.format(rgba))

        if edgecolor is None:
            if border:
                edgecolor = color
        elif edgecolor is True:
            edgecolor = color
        else:
            edgecolor = kwimage.Color(edgecolor, alpha=alpha)._forimage(image)

        if edgecolor:
            thickness = 4
            contour_idx = -1
            if alpha is None or alpha == 1.0:
                # Modification happens inplace
                image = cv2.drawContours(image, cv_contours, contour_idx,
                                         edgecolor, thickness, line_type)
            else:
                orig = image.copy()
                mask = np.zeros_like(orig)
                mask = cv2.drawContours(mask, cv_contours, contour_idx,
                                        edgecolor, thickness, line_type)
                image = kwimage.overlay_alpha_images(mask, orig)
                # edgecolor = kwimage.Color(edgecolor)._forimage(image)

        # image = kwimage.ensure_float01(image)[..., 0:3]
        # print('--- D')
        # print('image.dtype = {!r}'.format(image.dtype))
        # print('image.max() = {!r}'.format(image.max()))

        image = dtype_fixer(image, copy=False)
        return image

    def draw(self, color='blue', ax=None, alpha=1.0, radius=1, setlim=False,
             border=None, linewidth=None, edgecolor=None, facecolor=None,
             fill=True, vertex=False, vertexcolor=None):
        r"""
        Draws polygon in a matplotlib axes. See `draw_on` for in-memory image
        modification.

        Args:
            setlim (bool): if True ensures the limits of the axes contains the
                polygon

            color (str | Tuple): coercable color.
                Default color if specific colors are not given.

            alpha (float): fill transparency

            fill (bool):
                if True fill the polygon with facecolor, otherwise
                just draw the border if linewidth > 0

            setlim (bool): if True, modify the x and y limits of the matplotlib
                axes such that the polygon is can be seen.

            border (bool):
                if True, draws an edge border on the polygon.
                DEPRECATED. Use linewidth instead.

            linewidth (bool):
                width of the border

            edgecolor (None | Any):
                if None, uses the value of ``color``.
                Otherwise the color of the border when linewidth > 0.
                Extended types Coercable[kwimage.Color].

            facecolor (None | Any):
                if None, uses the value of ``color``.
                Otherwise, color of the border when fill=True.
                Extended types Coercable[kwimage.Color].

            vertex (float):
                if non-zero, draws vertexes on the polygon with this radius.

            vertexcolor (Any):
                color of vertexes
                Extended types Coercable[kwimage.Color].

        Returns:
            matplotlib.patches.PathPatch | None :
                None for am empty polygon

        TODO:
            - [ ] Rework arguments in favor of matplotlib standards

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1)
            >>> self = self.scale(100)
            >>> # xdoc: +REQUIRES(--show)
            >>> kwargs = dict(edgecolor='orangered', facecolor='dodgerblue', linewidth=10)
            >>> self.draw(**kwargs)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw(setlim=True, **kwargs)

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoc: +REQUIRES(--show)
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(n_holes=1, rng=33202)
            >>> import textwrap
            >>> # Test over a range of parameters
            >>> basis = {
            >>>     'linewidth': [0, 4],
            >>>     'edgecolor': [None, 'gold'],
            >>>     'facecolor': ['purple'],
            >>>     'fill': [True, False],
            >>>     'alpha': [1.0, 0.5],
            >>>     'vertex': [0, 0.01],
            >>>     'vertexcolor': ['green'],
            >>> }
            >>> grid = list(ub.named_product(basis))
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pnum_ = kwplot.PlotNums(nSubplots=len(grid))
            >>> for kwargs in grid:
            >>>     fig = kwplot.figure(fnum=1, pnum=pnum_())
            >>>     ax = fig.gca()
            >>>     self.draw(ax=ax, **kwargs)
            >>>     title = ub.repr2(kwargs, compact=True)
            >>>     title = '\n'.join(textwrap.wrap(
            >>>         title.replace(',', ' '), break_long_words=False,
            >>>         width=60))
            >>>     ax.set_title(title, fontdict={'fontsize': 8})
            >>>     ax.grid(False)
            >>>     ax.set_xticks([])
            >>>     ax.set_yticks([])
            >>> fig.subplots_adjust(wspace=0.5, hspace=0.3, bottom=0.001, top=0.97)
            >>> kwplot.show_if_requested()
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        import kwimage
        if ax is None:
            ax = plt.gca()

        if border is not None:
            ub.schedule_deprecation(
                modname='kwimage', migration='use linewidth instead',
                name='border', type='kwarg to Polygon.draw_on',
                deprecate='0.8.7', error='1.0.0', remove='1.1.0',
            )

        data = self.data

        exterior = data['exterior'].data.tolist()
        if len(exterior) == 0:
            return None  # empty

        color = list(kwimage.Color(color).as01())

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

        if border is None:
            border = (edgecolor is not None or linewidth is not None)
        else:
            if not border:
                linewidth = 0

        if facecolor is None:
            facecolor = color

        kw = {}
        # TODO:
        # depricate border kwarg in favor of standard matplotlib args
        if border:
            if linewidth is None:
                linewidth = 2
            kw['linewidth'] = linewidth
            if edgecolor is None:
                try:
                    edgecolor = list(kwimage.Color(border).as01())
                except Exception:
                    edgecolor = list(color)
                    # hack to darken
                    edgecolor[0] -= .1
                    edgecolor[1] -= .1
                    edgecolor[2] -= .1
                    edgecolor = [min(1, max(0, c)) for c in edgecolor]
            else:
                edgecolor = kwimage.Color(edgecolor).as01('rgba')
            kw['edgecolor'] = edgecolor
        else:
            kw['linewidth'] = 0
        kw['facecolor'] = facecolor

        patch = mpl.patches.PathPatch(path, alpha=alpha, fill=fill, **kw)
        ax.add_patch(patch)

        if vertex:
            if vertexcolor is None:
                vertexcolor = color
            data['exterior'].draw(color=vertexcolor, radius=vertex)
            for hole in interiors:
                hole.draw(color=vertexcolor, radius=vertex)

        if setlim:
            xmin, ymin, xmax, ymax = self.to_boxes().to_ltrb().data[0]
            _generic._setlim(xmin, ymin, xmax, ymax, setlim=setlim, ax=ax)
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

    def interpolate(self, other, alpha):
        ub.schedule_deprecation(
            modname='kwimage', migration='use morph instead',
            name='interpolate', type='method',
            # deprecate='0.8.7', error='1.0.0', remove='1.1.0',
            deprecate='now', error='soon', remove='soon',
        )
        return self.morph(other, alpha)

    def morph(self, other, alpha):
        """
        Perform polygon-to-polygon morphing.

        Note:
            This current algorithm is very basic and does not yet prevent
            self-intersections in intermediate polygons.

        Args:
            other (kwimage.Polygon): the other polygon to morph into

            alpha (float | List[float]):
                A value between 0 and 1, indicating the fractional position of
                the new interpolated polygon between ``self`` and ``other``.
                If given as a list multiple interpolations are returned.

        Returns:
            Polygon | List[Polygon]: one ore more interpolated polygons

        TODO:
            - [ ] Implement level set method [LevelSet]_ which rasterizes each
            polygon, interpolates the raster, and computes the interpolated
            polygon as contours in that interpolated raster.

            - [ ] Implement methods from [Albrecht2006]_ and [PolygonMorph]_

        References:
            .. [InterpPoly] http://lambdafunk.com/2017-02-21-Interpolating-Polygons/
            .. [LevelSet] https://en.wikipedia.org/wiki/Level-set_method
            .. [HaudrenShapes] https://github.com/haudren/shapes/blob/master/shapes/shapes.py
            .. [PolygonMorph] https://github.com/micycle1/Polygon-Morphing
            .. [Albrecht2006] http://www2.inf.uos.de/prakt/pers/dipl/svalbrec/thesis.pdf
            .. [KamvysselisMorph] http://web.mit.edu/manoli/www/ecimorph/ecimorph.html#code

        Example:
            >>> import kwimage
            >>> self = kwimage.Polygon.random(3, convex=0)
            >>> other = kwimage.Polygon.random(4, convex=0).translate((2, 2))
            >>> results = self.morph(other, np.linspace(0, 1, 5))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(doclf=1)
            >>> self.draw(setlim='grow', color='kw_blue', alpha=0.5, vertex=0.02)
            >>> other.draw(setlim='grow', color='kw_green', alpha=0.5, vertex=0.02)
            >>> colors = kwimage.Color('kw_blue').morph(
            >>>     'kw_green', np.linspace(0, 1, 5))
            >>> for new, c in zip(results, colors):
            >>>     pt = new.exterior.data[0]
            >>>     new.draw(color=c, alpha=0.5, vertex=0.01)
            >>> intepolation_lines = np.array([new.exterior.data for new in results])
            >>> for interp_line in intepolation_lines.transpose(1, 0, 2)[::8]:
            >>>     plt.plot(*interp_line.T, '--x')
        """
        from shapely import geometry
        import kwimage
        # Create a variant of each polygon
        shape1 = self.to_shapely()
        shape2 = other.to_shapely()

        if len(shape1.interiors) > 0:
            raise NotImplementedError('no holes in interpolation yet')

        if len(shape2.interiors) > 0:
            raise NotImplementedError('no holes in interpolation yet')

        ring1 = shape1.exterior
        ring2 = shape2.exterior

        ring1_coords = np.array(ring1.xy).T
        ring2_coords = np.array(ring2.xy).T

        # Rotate each ring to align the leftmost point
        # Rotate the coordinates such that vertices are in better
        # correspondence. There are different ways we can (and should implement
        # options to) do this, but this one is reasonable
        idx1 = ring1_coords.argmin(axis=0)[0]
        idx2 = ring2_coords.argmin(axis=0)[0]
        aligned_ring_coords1 = np.roll(ring1_coords, -idx1, axis=0)
        aligned_ring_coords2 = np.roll(ring2_coords, -idx2, axis=0)

        ring1 = geometry.polygon.LinearRing(aligned_ring_coords1)
        ring2 = geometry.polygon.LinearRing(aligned_ring_coords2)

        # For each polygon exterior, find the normalized fractional point each
        # vertex lives on.
        ring_dist1 = [ring1.project(geometry.Point(pt), normalized=True)
                      for pt in zip(*ring1.xy)]
        ring_dist2 = [ring2.project(geometry.Point(pt), normalized=True)
                      for pt in zip(*ring2.xy)]
        # Get a common set of normalized ring distances
        ring_distB = sorted(set(ring_dist1 + ring_dist2))
        interps = np.array(ring_distB)

        # Determine the coordinates for the common fractional points on each
        # exterior.
        length1 = shape1.exterior.length
        length2 = shape2.exterior.length

        interps1 = (interps * length1) % length1
        interps2 = (interps * length2) % length2

        coords1 = np.array([ring1.interpolate(i).xy for i in interps1])[..., 0]
        coords2 = np.array([ring2.interpolate(i).xy for i in interps2])[..., 0]

        # # Find the left-most point and use that as the base for the
        # # correspondence.
        # idx1 = coords1.argmin(axis=0)[0]
        # idx2 = coords2.argmin(axis=0)[0]

        # # centered_coords1 = coords1 - coords1.mean(axis=1, keepdims=1)
        # # centered_coords2 = coords2 - coords2.mean(axis=1, keepdims=1)
        # # kwarray.algo_assignment.mindist_assignment(centered_coords1, centered_coords2)
        # # kwarray.algo_assignment.mindist_assignment(coords1, coords2)

        # aligned_coords1 = np.roll(coords1, -idx1, axis=0)
        # aligned_coords2 = np.roll(coords2, -idx2, axis=0)

        was_iterable = ub.iterable(alpha)
        if not was_iterable:
            alpha = [alpha]

        alpha2 = np.array(alpha).ravel()[:, None, None]
        alpha1 = 1 - alpha2

        interpolated_coords = (
            (coords1[None, :] * alpha1) +
            (coords2[None, :] * alpha2)
        )
        result = [kwimage.Polygon(exterior=xy) for xy in interpolated_coords]
        if not was_iterable:
            result = result[0]
        return result


class MultiPolygon(_generic.ObjectList, _ShapelyMixin):
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

    def fill(self, image, value=1, pixels_are='points'):
        """
        Inplace fill in an image based on this multi-polyon.

        Args:
            image (ndarray):
                image to draw on (inplace)

            value (int | Tuple[int, ...]):
                value fill in with. Defaults to 1.0

        Returns:
            ndarray: the image that has been modified in place
        """
        for p in self.data:
            p.fill(image, value=value, pixels_are=pixels_are)
        return image

    def to_multi_polygon(self):
        """
        Returns:
            MultiPolygon
        """
        return self

    def to_boxes(self):
        """
        Deprecated: lossy conversion use 'bounding_box' instead

        Returns:
            kwimage.Boxes
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

    def to_mask(self, dims=None, pixels_are='points'):
        """
        Returns a mask object indication regions occupied by this multipolygon

        Returns:
            kwimage.Mask

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> s = 100
            >>> self = MultiPolygon.random(rng=0).scale(s)
            >>> dims = (s, s)
            >>> mask = self.to_mask(dims)
            >>> # xdoc: +REQUIRES(--show)
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> kwplot.figure(fnum=1, doclf=True)
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
                p.fill(c_mask, value=1, pixels_are=pixels_are)
        mask = kwimage.Mask(c_mask, 'c_mask')
        return mask

    def to_relative_mask(self, return_offset=False):
        """
        Returns a translated mask such the mask dimensions are minimal.

        In other words, we move the polygon all the way to the top-left and
        return a mask just big enough to fit the polygon.

        Returns:
            kwimage.Mask
        """
        # dims (Tuple[int, int] | None):
        #     if you know *exactly* how big the polygon is you can specify
        #     this, otherwise it will be computed.
        # if dims is not None:
        x, y, w, h = self.to_boxes().quantize().to_xywh().data[0]
        mask = self.translate((-x, -y)).to_mask(dims=(h, w))
        if return_offset:
            offset = (x, y)
            return mask, offset
        else:
            return mask

    @classmethod
    def coerce(cls, data, dims=None):
        """
        Attempts to construct a MultiPolygon instance from the input data

        See Segmentation.coerce

        Returns:
            None | MultiPolygon

        Example:
            >>> import kwimage
            >>> dims = (32, 32)
            >>> kw_poly = kwimage.Polygon.random().scale(dims)
            >>> kw_multi_poly = kwimage.MultiPolygon.random().scale(dims)
            >>> forms = [kw_poly, kw_multi_poly]
            >>> forms.append(kw_poly.to_shapely())
            >>> forms.append(kw_poly.to_mask((32, 32)))
            >>> forms.append(kw_poly.to_geojson())
            >>> forms.append(kw_poly.to_coco(style='orig'))
            >>> forms.append(kw_poly.to_coco(style='new'))
            >>> forms.append(kw_multi_poly.to_shapely())
            >>> forms.append(kw_multi_poly.to_mask((32, 32)))
            >>> forms.append(kw_multi_poly.to_geojson())
            >>> forms.append(kw_multi_poly.to_coco(style='orig'))
            >>> forms.append(kw_multi_poly.to_coco(style='new'))
            >>> for data in forms:
            >>>     result = kwimage.MultiPolygon.coerce(data, dims=dims)
            >>>     assert isinstance(result, kwimage.MultiPolygon)
        """
        from kwimage.structs.segmentation import _coerce_coco_segmentation
        self = _coerce_coco_segmentation(data, dims=dims)
        if self is not None:
            self = self.to_multi_polygon()
        return self

    def to_shapely(self, fix=False):
        """
        Args:
            fix (bool):
                if True, will check for validity and if any simple fixes
                can be applied, otherwise it returns the data as is.

        Returns:
            shapely.geometry.MultiPolygon

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
        if fix:
            if not geom.is_valid:
                geom = geom.buffer(0)
        return geom

    @classmethod
    def from_shapely(MultiPolygon, geom):
        """
        Convert a shapely polygon or multipolygon to a kwimage.MultiPolygon

        Args:
            geom (shapely.geometry.MultiPolygon | shapely.geometry.Polygon):

        Returns:
            MultiPolygon

        Example:
            >>> import kwimage
            >>> sh_poly = kwimage.Polygon.random().to_shapely()
            >>> sh_multi_poly = kwimage.MultiPolygon.random().to_shapely()
            >>> kwimage.MultiPolygon.from_shapely(sh_poly)
            >>> kwimage.MultiPolygon.from_shapely(sh_multi_poly)
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

        Args:
            data_geojson (Dict): geojson data

        Returns:
            MultiPolygon

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

        Returns:
            Dict
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

        Args:
            data (List[List[Number] | Dict]):
                a new or old style coco multi polygon

            dims (None | Tuple[int, ...]):
                the shape dimensions of the canvas. Unused. Exists for
                compatibility with masks.

        Returns:
            MultiPolygon
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
        Args:
            style(str): can be "orig" or "new"

        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = MultiPolygon.random(1, rng=0)
            >>> self.to_coco()
        """
        return [item.to_coco(style=style) for item in self.data]

    def swap_axes(self, inplace=False):
        """
        Swap x and y axis

        Args:
            inplace (bool):

        Returns:
            MultiPolygon
        """
        return self.apply(lambda item: item.swap_axes(inplace=inplace))

    def draw_on(self, image, **kwargs):
        Polygon.draw_on.__doc__
        for item in self.data:
            if item is not None:
                image = item.draw_on(image, **kwargs)
        return image

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

    def to_mask_list(self, dims=None, pixels_are='points'):
        """
        Converts all items to masks

        Returns:
            kwimage.MaskList
        """
        import kwimage
        new = kwimage.MaskList([
            None if item is None else item.to_mask(dims=dims, pixels_are=pixels_are)
            for item in self
        ])
        return new

    def to_polygon_list(self):
        """
        Returns:
            PolygonList
        """
        return self

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

    def swap_axes(self, inplace=False):
        """
        Returns:
            PolygonList
        """
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

    def fill(self, image, value=1, pixels_are='points'):
        """
        Inplace fill in an image based on these polygons

        Args:
            image (ndarray): image to draw on (inplace)
            value (int | Tuple[int, ...]):
                value fill in with

        Returns:
            ndarray: the image that has been modified in place
        """
        for p in self.data:
            if p is not None:
                p.fill(image, value=value, pixels_are=pixels_are)
        return image

    def draw_on(self, *args, **kw):
        """
        Ignore:
            >>> # Test that we can draw a lot of polygons quickly by default
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--slow)
            >>> import kwimage
            >>> s = 512
            >>> canvas = kwimage.grab_test_image(dsize=(s, s))
            >>> kwimage.ensure_float01(canvas)
            >>> data = [kwimage.MultiPolygon.random().scale(s) for _ in ub.ProgIter(range(1), desc='gen poly')]
            >>> #data = [kwimage.Polygon.random().scale(s) for _ in ub.ProgIter(range(5), desc='gen poly')]
            >>> self = kwimage.PolygonList(data)
            >>> with ub.Timer('regular draw'):
            >>>     out_canvas1 = self.draw_on(canvas.copy(), fill=0, border=1)
            >>> with ub.Timer('alpha draw'):
            >>>     out_canvas2 = self.draw_on(canvas.copy(), alpha=0.5, fill=1, border=1, edgecolor='red')
            >>> # Disabling fast-draw will make drawing multiples much slower
            >>> with ub.Timer('alpha draw, nofast'):
            >>>     out_canvas3 = self.draw_on(canvas.copy(), alpha=0.5, fastdraw=False, fill=1, border=1, edgecolor='red')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(out_canvas1, pnum=(1, 3, 1), fnum=1)
            >>> kwplot.imshow(out_canvas2, pnum=(1, 3, 2), fnum=1)
            >>> kwplot.imshow(out_canvas3, pnum=(1, 3, 3), fnum=1)
        """
        Polygon.draw_on.__doc__
        # ^ docstring
        return super().draw_on(*args, **kw)

    def unary_union(self):
        from shapely.ops import unary_union
        from kwimage.structs.polygon import _kwimage_from_shapely
        polys_sh = [p.to_shapely() for p in self]
        union_sh = unary_union(polys_sh)
        new = _kwimage_from_shapely(union_sh)
        return new


def _kwimage_from_shapely(geom):
    import kwimage
    if geom.geom_type == 'Polygon':
        return kwimage.Polygon.from_shapely(geom)
    elif geom.geom_type == 'MultiPolygon':
        return kwimage.MultiPolygon.from_shapely(geom)
    else:
        raise TypeError(geom.geom_type)


def _is_clockwise(verts):
    """
    References:
        https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order

    Ignore:
        verts = poly.data['exterior'].data[::-1]
    """
    x1 = verts[:-1][:, 0]
    y1 = verts[:-1][:, 1]
    x2 = verts[1:][:, 0]
    y2 = verts[1:][:, 1]
    is_clockwise = ((x2 - x1) * (y2 + y1)).sum() > 0
    # cross_product = np.cross(verts[:-1], verts[1:])
    # is_clockwise = cross_product.sum() > 0
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
