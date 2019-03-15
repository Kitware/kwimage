import cv2
import numpy as np
from . import _generic
# from .points import Points
import torch


# class Polygon(Points):
#     """
#     A polygon is a list of points, so we reuse much of that implementation

#     A polygon is an external ring of coordinates

#     Notes: this implementation is similar to a MultiPolygon geojson object

#     TODO:
#         - [ ] support holes
#         - [ ] support disjoint polygons

#     Example:
#         >>> xy = np.array([[0, 0], [.5, 0], [.5, .6], [.1, .2]])
#         >>> Polygon(xy=xy)
#     """
#     __datakeys__ = ['xy']
#     __metakeys__ = ['classes']

#     def draw_on(self, image, color='blue', fill=True):
#         """
#         Example:
#             >>> from kwimage.structs.polygon import *  # NOQA
#             >>> image = np.zeros((128, 128), dtype=np.float32)
#             >>> self = Polygon.random(10, rng=0).scale(128)
#             >>> image = self.draw_on(image)
#             >>> # xdoc: +REQUIRES(--show)
#             >>> import kwplot
#             >>> kwplot.autompl()
#             >>> kwplot.imshow(image, fnum=1)
#         """
#         import kwplot
#         import kwimage

#         def adjust_hsv_of_rgb(rgb, hue_adjust=0.0, sat_adjust=0.0, val_adjust=0.0):
#             """ works on a single rgb tuple """
#             import colorsys
#             # assert_base01(rgb)
#             #assert_base01([sat_adjust, val_adjust])
#             numpy_input = isinstance(rgb, np.ndarray)
#             # For some reason numpy input does not work well
#             if numpy_input:
#                 dtype = rgb.dtype
#                 rgb = rgb.tolist()
#             #print('rgb=%r' % (rgb,))
#             alpha = None
#             if len(rgb) == 4:
#                 (R, G, B, alpha) = rgb
#             else:
#                 (R, G, B) = rgb
#             hsv = colorsys.rgb_to_hsv(R, G, B)
#             (H, S, V) = hsv
#             H_new = (H + hue_adjust)
#             if H_new > 0 or H_new < 1:
#                 # is there a way to more ellegantly get this?
#                 H_new %= 1.0
#             S_new = max(min(S + sat_adjust, 1.0), 0.0)
#             V_new = max(min(V + val_adjust, 1.0), 0.0)
#             #print('hsv=%r' % (hsv,))
#             hsv_new = (H_new, S_new, V_new)
#             #print('hsv_new=%r' % (hsv_new,))
#             new_rgb = colorsys.hsv_to_rgb(*hsv_new)
#             if alpha is not None:
#                 new_rgb = list(new_rgb) + [alpha]
#             #print('new_rgb=%r' % (new_rgb,))
#             # assert_base01(new_rgb)
#             # Return numpy if given as numpy
#             if numpy_input:
#                 new_rgb = np.array(new_rgb, dtype=dtype)
#             return new_rgb

#         # return shape of contours to openCV contours
#         thickness = 1
#         xy = self.data['xy'].data
#         contours = np.expand_dims(xy, axis=1)
#         contours = contours.astype(np.int)

#         contour_idx = -1
#         line_type = cv2.LINE_AA
#         image = kwimage.ensure_uint255(image)
#         image = kwimage.atleast_3channels(image)

#         color01 = kwplot.Color(color).as01()
#         color2 = adjust_hsv_of_rgb(color01, val_adjust=-.3, sat_adjust=-.3)

#         rgb2 = kwplot.Color(color2).as255()
#         rgb1 = kwplot.Color(color).as255()

#         image = cv2.fillPoly(image, [contours], rgb2, line_type, shift=0)
#         image = cv2.drawContours(image, [contours], contour_idx, rgb1,
#                                  thickness, line_type)
#         print(image.sum())
#         image = kwimage.ensure_float01(image)
#         print(image.sum())
#         return image

#     def draw(self, color='blue', ax=None, alpha=None, radius=1):
#         """
#         Example:
#             >>> from kwimage.structs.polygon import *  # NOQA
#             >>> self = Polygon.random(10, rng=0)
#             >>> self.draw()
#             >>> # xdoc: +REQUIRES(--show)
#             >>> import kwplot
#             >>> kwplot.autompl()
#             >>> from matplotlib import pyplot as plt
#             >>> kwplot.figure(fnum=2)
#             >>> self.draw()
#             >>> ax = plt.gca()
#             >>> ax.invert_yaxis()
#         """
#         import matplotlib as mpl
#         from matplotlib import pyplot as plt
#         if ax is None:
#             ax = plt.gca()

#         xy = self.data['xy'].data
#         poly = mpl.patches.Polygon(xy)

#         # print('sseg_polys = {!r}'.format(sseg_polys))
#         poly_col = mpl.collections.PatchCollection([poly], 2, alpha=0.4)
#         ax.add_collection(poly_col)

class Polygon(object):
    """
    A polygon is a list of points, so we reuse much of that implementation

    A polygon is an external ring of coordinates

    Notes: this implementation is similar to a MultiPolygon geojson object

    Example:
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [np.array([[13, 13], [14, 12], [24, 12], [25, 13], [25, 18], [24, 19], [14, 19], [13, 18]]),
        >>>                   np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11], [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = Polygon(data)
    """
    __datakeys__ = ['exterior', 'interiors']
    __metakeys__ = []

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

            import kwimage
            if 'exterior' in data:
                if isinstance(data['exterior'], (np.ndarray, torch.Tensor)):
                    data['exterior'] = kwimage.Coords(data['exterior'])
            if 'interiors' in data:
                holes = []
                for hole in data['interiors']:
                    if isinstance(data['interiors'], (np.ndarray, torch.Tensor)):
                        hole = kwimage.Coords(hole)
                    holes.append(hole)
                data['interiors'] = holes

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def draw_on(self, image, color='blue', fill=True):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0).scale(128)
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
        contours = [data['exterior']] + data['interiors']
        contours = [np.expand_dims(c.astype(np.int), axis=1) for c in contours]

        rgb = kwplot.Color(color).as255()

        image = cv2.fillPoly(image, contours, rgb, line_type, shift=0)

        if 0:
            image = cv2.drawContours(image, contours, contour_idx, rgb, thickness,
                                     line_type)
        image = kwimage.ensure_float01(image)
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0)
            >>> self.draw()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw()
            >>> ax = plt.gca()
            >>> ax.invert_yaxis()
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        data = self.data

        exterior = data['exterior'].tolist()
        exterior.append(exterior[0])
        n = len(exterior)
        verts = []
        verts.extend(exterior)
        codes = [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        interiors = data['interiors']
        for hole in interiors:
            hole = hole.tolist()
            hole.append(hole[0])
            n = len(hole)
            verts.extend(hole)
            codes += [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        verts = np.array(verts)
        path = Path(verts, codes)
        patch = mpl.patches.PathPatch(path, alpha=.4)
        ax.add_patch(patch)


class PolygonList(_generic.ObjectList):
    pass
