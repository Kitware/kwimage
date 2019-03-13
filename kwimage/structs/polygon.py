import cv2
import numpy as np
from . import _generic
from .points import Points


class Polygon(Points):
    """
    A polygon is a list of points, so we reuse much of that implementation

    TODO:
        - [ ] support holes
        - [ ] support disjoint polygons

    Example:
        >>> xy = np.array([[0, 0], [.5, 0], [.5, .6], [.1, .2]])
        >>> Polygon(xy=xy)
    """
    __datakeys__ = ['xy']
    __metakeys__ = ['classes']

    def draw_on(self, image, color='blue', fill=True):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> self = Polygon.random(10, rng=0).scale(128)
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
        xy = self.data['xy'].data
        contours = np.expand_dims(xy, axis=1)
        contours = contours.astype(np.int)

        contour_idx = -1
        line_type = cv2.LINE_AA
        image = kwimage.ensure_uint255(image)
        image = kwimage.atleast_3channels(image)

        def adjust_hsv_of_rgb(rgb, hue_adjust=0.0, sat_adjust=0.0, val_adjust=0.0):
            """ works on a single rgb tuple """
            import colorsys
            # assert_base01(rgb)
            #assert_base01([sat_adjust, val_adjust])
            numpy_input = isinstance(rgb, np.ndarray)
            # For some reason numpy input does not work well
            if numpy_input:
                dtype = rgb.dtype
                rgb = rgb.tolist()
            #print('rgb=%r' % (rgb,))
            alpha = None
            if len(rgb) == 4:
                (R, G, B, alpha) = rgb
            else:
                (R, G, B) = rgb
            hsv = colorsys.rgb_to_hsv(R, G, B)
            (H, S, V) = hsv
            H_new = (H + hue_adjust)
            if H_new > 0 or H_new < 1:
                # is there a way to more ellegantly get this?
                H_new %= 1.0
            S_new = max(min(S + sat_adjust, 1.0), 0.0)
            V_new = max(min(V + val_adjust, 1.0), 0.0)
            #print('hsv=%r' % (hsv,))
            hsv_new = (H_new, S_new, V_new)
            #print('hsv_new=%r' % (hsv_new,))
            new_rgb = colorsys.hsv_to_rgb(*hsv_new)
            if alpha is not None:
                new_rgb = list(new_rgb) + [alpha]
            #print('new_rgb=%r' % (new_rgb,))
            # assert_base01(new_rgb)
            # Return numpy if given as numpy
            if numpy_input:
                new_rgb = np.array(new_rgb, dtype=dtype)
            return new_rgb

        color01 = kwplot.Color(color).as01()
        color2 = adjust_hsv_of_rgb(color01, val_adjust=-.3, sat_adjust=-.3)

        rgb2 = kwplot.Color(color2).as255()
        rgb1 = kwplot.Color(color).as255()

        image = cv2.fillPoly(image, [contours], rgb2, line_type, shift=0)
        image = cv2.drawContours(image, [contours], contour_idx, rgb1,
                                 thickness, line_type)
        print(image.sum())
        image = kwimage.ensure_float01(image)
        print(image.sum())
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
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        xy = self.data['xy'].data
        poly = mpl.patches.Polygon(xy)

        # print('sseg_polys = {!r}'.format(sseg_polys))
        poly_col = mpl.collections.PatchCollection([poly], 2, alpha=0.4)
        ax.add_collection(poly_col)


class PolygonList(_generic.ObjectList):
    pass
