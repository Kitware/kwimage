import skimage
import numpy as np
import kwplot
import ubelt as ub
import cv2
import kwplot
import kwimage
import shapely


def check_fill_poly_properties():
    """
    Notes:
        It seems as if cv2.fillPoly will draw multiple polygons, but it will
        toggle between drawing holes and filling depending on if the next
        polygon is inside of a previous one.

        skimage.draw.polygon is very slow

        PIL is very slow for floats, but ints aren't too bad. cv2 is better.

    """
    kwplot.autompl()

    shape = (1208, 1208)
    self = kwimage.Polygon.random(n=10, n_holes=1, convex=False).scale(1208)

    cv_contours = self._to_cv_countours()

    value = 1

    mask = np.zeros((128, 128), dtype=np.uint8)
    value = 1
    line_type = cv2.LINE_8

    mask = np.zeros((128, 128), dtype=np.uint8)
    # Modification happens inplace
    cv2.fillPoly(mask, cv_contours, value, line_type, shift=0)

    kwplot.autompl()
    kwplot.imshow(mask)

    extra = cv_contours[1] + 40
    cv_contours3 = cv_contours + [extra, extra + 2]
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.fillPoly(mask, cv_contours3, value, line_type, shift=0)

    kwplot.imshow(mask)

    geom = shapely.geometry.Polygon(
        shell=self.data['exterior'].data,
        holes=[c.data for c in self.data['interiors']]
    )

    xs, ys = self.data['exterior'].data.T
    rr, cc = skimage.draw.polygon(xs, ys)


    mask = np.zeros(shape, dtype=np.uint8)

    ti = ub.Timerit(10, bestof=3, verbose=2, unit='us')
    if False:
        # Not general enough
        for timer in ti.reset('fillConvexPoly'):
            mask[:, :] = 0
            with timer:
                cv_contours = self._to_cv_countours()
                cv2.fillConvexPoly(mask, cv_contours[0], value)

    for timer in ti.reset('fillPoly'):
        mask[:, :] = 0
        with timer:
            cv_contours = self._to_cv_countours()
            cv2.fillPoly(mask, cv_contours[0:1], value)

    for timer in ti.reset('skimage.draw.polygon'):
        mask = np.zeros(shape, dtype=np.uint8)
        with timer:
            xs, ys = self.data['exterior'].data.T
            rr, cc = skimage.draw.polygon(xs, ys)

    from PIL import Image, ImageDraw
    for timer in ti.reset('PIL'):
        height, width = shape
        pil_img = Image.new('L', (width, height), 0)
        with timer:
            draw_obj = ImageDraw.Draw(pil_img)
            pil_poly = self.data['exterior'].data.astype(np.int).ravel().tolist()
            pil_poly = pil_poly + pil_poly[0:2]
            draw_obj.polygon(pil_poly, outline=0, fill=255)
            mask = np.array(pil_img)


def fastfill_multipolygon():
    kwplot.autompl()
    shape = (1208, 1208)
    self = kwimage.MultiPolygon.random(10).scale(shape)

    ti = ub.Timerit(3, bestof=1, verbose=2, unit='us')
    for timer in ti.reset('draw_on'):
        with timer:
            mask = np.zeros(shape, dtype=np.uint8)
            mask = self.draw_on(mask)

    for timer in ti.reset('custom'):
        with timer:
            mask = np.zeros(shape, dtype=np.uint8)
            for p in self.data:
                if p is not None:
                    p.fill(mask, value=255)

    for timer in ti.reset('to_mask'):
        with timer:
            self.to_mask(shape)

    kwplot.imshow(mask)
