# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import numpy as np
import cv2


def draw_text_on_image(img, text, org, return_info=False, **kwargs):
    r"""
    Draws multiline text on an image using opencv

    Note:
        This function also exists in kwplot

        The image is modified inplace. If the image is non-contiguous then this
        returns a UMat instead of a ndarray, so be carefull with that.

    Args:
        img (ndarray): image to draw on (inplace)
        text (str): text to draw
        org (tuple): x, y location of the text string in the image.
            if bottomLeftOrigin=True this is the bottom-left corner of the text
            otherwise it is the top-left corner (default).
        return_info (bool, default=False):
            if True, also returns information about the positions the text
            was drawn on.
        **kwargs:
            color (tuple): default blue
            thickneess (int): defaults to 2
            fontFace (int): defaults to cv2.FONT_HERSHEY_SIMPLEX
            fontScale (float): defaults to 1.0
            valign (str, default='bottom'): either top, center, or bottom
            halign (str, default='left'): either left, center, or right

    References:
        https://stackoverflow.com/questions/27647424/

    Example:
        >>> import kwimage
        >>> img = kwimage.grab_test_image(space='rgb')
        >>> img2 = kwimage.draw_text_on_image(img.copy(), 'FOOBAR', org=(0, 0), valign='top')
        >>> assert img2.shape == img.shape
        >>> assert np.any(img2 != img)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img2, fontScale=10)
        >>> kwplot.show_if_requested()

    Example:
        >>> import kwimage
        >>> # Test valign
        >>> img = kwimage.grab_test_image(space='rgb', dsize=(500, 500))
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(0, 0), valign='top', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(150, 0), valign='center', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(300, 0), valign='bottom', border=2)
        >>> # Test halign
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(250, 100), halign='right', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(250, 250), halign='center', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(250, 400), halign='left', border=2)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img2, fontScale=10)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Ensure the function works with float01 or uint255 images
        >>> import kwimage
        >>> img = kwimage.grab_test_image(space='rgb')
        >>> img = kwimage.ensure_float01(img)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(0, 0), valign='top', border=2)
    """
    import kwimage

    if 'color' not in kwargs:
        kwargs['color'] = 'red'

    # Get the color that is compatible with the input image encoding
    if img is None:
        kwargs['color'] = kwimage.Color(kwargs['color']).as255()
    else:
        kwargs['color'] = kwimage.Color(kwargs['color'])._forimage(img)

    if 'thickness' not in kwargs:
        kwargs['thickness'] = 2

    if 'fontFace' not in kwargs:
        kwargs['fontFace'] = cv2.FONT_HERSHEY_SIMPLEX

    if 'fontScale' not in kwargs:
        kwargs['fontScale'] = 1.0

    if 'lineType' not in kwargs:
        kwargs['lineType'] = cv2.LINE_AA

    if 'bottomLeftOrigin' in kwargs:
        raise ValueError('Do not use bottomLeftOrigin, use valign instead')

    border = kwargs.pop('border', None)
    if border:
        # recursive call
        subkw = kwargs.copy()
        subkw['color'] = 'black'
        subkw.pop('return_info', None)
        basis = list(range(-border, border + 1))
        for i, j in it.product(basis, basis):
            if i == 0 and j == 0:
                continue
            org = np.array(org)
            img = draw_text_on_image(img, text, org=org + [i, j], **subkw)

    valign = kwargs.pop('valign', None)
    halign = kwargs.pop('halign', None)

    getsize_kw = {
        k: kwargs[k]
        for k in ['fontFace', 'fontScale', 'thickness']
        if k in kwargs
    }
    x0, y0 = list(map(int, org))
    thickness = kwargs.get('thickness', 2)
    ypad = thickness + 4

    lines = text.split('\n')
    # line_sizes2 = np.array([cv2.getTextSize(line, **getsize_kw) for line in lines])
    # print('line_sizes2 = {!r}'.format(line_sizes2))
    line_sizes = np.array([cv2.getTextSize(line, **getsize_kw)[0] for line in lines])

    line_org = []
    y = y0
    for w, h in line_sizes:
        next_y = y + (h + ypad)
        line_org.append((x0, y))
        y = next_y
    line_org = np.array(line_org)

    # the absolute top and bottom position of text
    all_top_y = line_org[0, 1]
    all_bottom_y = (line_org[-1, 1] + line_sizes[-1, 1])

    first_h = line_sizes[0, 1]

    total_h = (all_bottom_y - all_top_y)
    total_w = line_sizes.T[0].max()

    if valign is not None:
        if valign == 'bottom':
            # This is the default for the one-line case
            # in the multiline case we need to subtract the total
            # height of all lines but the first to ensure the last
            # line is on the bottom.
            line_org[:, 1] -= (total_h - first_h)
        elif valign == 'center':
            # Change from bottom to center
            line_org[:, 1] += first_h - total_h // 2
        elif valign == 'top':
            # Because bottom is the default we just need to add height of the
            # first line.
            line_org[:, 1] += first_h
        else:
            raise KeyError(valign)

    if halign is not None:
        if halign == 'left':
            # This is the default case, no modification needed
            pass
        elif halign == 'center':
            # When the x-orgin should be the center, subtract half of
            # the line width to get the leftmost point.
            line_org[:, 0] = x0 - (line_sizes[:, 0] / 2)
        elif halign == 'right':
            # The x-orgin should be the rightmost point, subtract
            # the width of each line to find the leftmost point.
            line_org[:, 0] = x0 - line_sizes[:, 0]
        else:
            raise KeyError(halign)

    if img is None:
        # if image is unspecified allocate just enough space for text
        total_w = (line_org.T[0] + line_sizes.T[0]).max()
        # TODO: does not account for origin offset
        img = np.zeros((total_h + thickness, total_w, 3), dtype=np.uint8)

    for i, line in enumerate(lines):
        (x, y) = line_org[i]
        img = cv2.putText(img, line, (x, y), **kwargs)

    if return_info:
        info = {
            'line_org': line_org,
            'line_sizes': line_sizes,
        }
        return img, info
    else:
        return img


def draw_clf_on_image(im, classes, tcx=None, probs=None, pcx=None, border=1):
    """
    Draws classification label on an image.

    Works best with image chips sized between 200x200 and 500x500

    Args:
        im (ndarray): the image
        classes (Sequence | CategoryTree): list of class names
        tcx (int, default=None): true class index if known
        probs (ndarray): predicted class probs for each class
        pcx (int, default=None): predicted class index.
            (if None but probs is specified uses argmax of probs)

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> import kwarray
        >>> import kwimage
        >>> rng = kwarray.ensure_rng(0)
        >>> im = (rng.rand(300, 300) * 255).astype(np.uint8)
        >>> classes = ['cls_a', 'cls_b', 'cls_c']
        >>> tcx = 1
        >>> probs = rng.rand(len(classes))
        >>> probs[tcx] = 0
        >>> probs = torch.FloatTensor(probs).softmax(dim=0).numpy()
        >>> im1_ = kwimage.draw_clf_on_image(im, classes, tcx, probs)
        >>> probs[tcx] = .9
        >>> probs = torch.FloatTensor(probs).softmax(dim=0).numpy()
        >>> im2_ = kwimage.draw_clf_on_image(im, classes, tcx, probs)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(im1_, colorspace='rgb', pnum=(1, 2, 1), fnum=1, doclf=True)
        >>> kwplot.imshow(im2_, colorspace='rgb', pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    im_ = kwimage.atleast_3channels(im)
    w, h = im.shape[0:2][::-1]

    if pcx is None and probs is not None:
        import kwarray
        probs = kwarray.ArrayAPI.numpy(probs)
        pcx = probs.argmax()

    if probs is not None:
        pred_score = None if pcx is None else probs[pcx]
        true_score = None if tcx is None else probs[tcx]

    org1 = np.array((2, h - 5))
    org2 = np.array((2, 5))

    true_label = None
    if tcx is not None:
        true_name = classes[tcx]
        if pcx == tcx:
            true_label = 't:{tcx}:{true_name}'.format(**locals())
        elif probs is None:
            true_label = 't:{tcx}:\n{true_name}'.format(**locals())
        else:
            true_label = 't:{tcx}@{true_score:.2f}:\n{true_name}'.format(**locals())

    pred_label = None
    if pcx is not None:
        pred_name = classes[pcx]
        if probs is None:
            pred_label = 'p:{pcx}:\n{pred_name}'.format(**locals())
        else:
            pred_label = 'p:{pcx}@{pred_score:.2f}:\n{pred_name}'.format(**locals())

    fontkw = {
        'fontScale': 1.0,
        'thickness': 2
    }
    color = 'dodgerblue' if pcx == tcx else 'orangered'

    # im_ = draw_text_on_image(im_, pred_label, org=org1 - 2,
    #                          color='white', valign='bottom', **fontkw)
    # im_ = draw_text_on_image(im_, true_label, org=org2 - 2,
    #                          color='white', valign='top', **fontkw)

    if pred_label is not None:
        im_ = draw_text_on_image(im_, pred_label, org=org1, color=color,
                                 border=border, valign='bottom', **fontkw)
    if true_label is not None:
        im_ = draw_text_on_image(im_, true_label, org=org2, color='lawngreen',
                                 valign='top', border=border, **fontkw)
    return im_


def draw_boxes_on_image(img, boxes, color='blue', thickness=1,
                        box_format=None, colorspace='rgb'):
    """
    Draws boxes on an image.

    Args:
        img (ndarray): image to copy and draw on
        boxes (nh.util.Boxes): boxes to draw
        colorspace (str): string code of the input image colorspace

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> color = 'dodgerblue'
        >>> thickness = 1
        >>> boxes = kwimage.Boxes([[1, 1, 8, 8]], 'ltrb')
        >>> img2 = draw_boxes_on_image(img, boxes, color, thickness)
        >>> assert tuple(img2[1, 1]) == (30, 144, 255)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.imshow(img2)
    """
    import kwimage
    import cv2
    if not isinstance(boxes, kwimage.Boxes):
        if box_format is None:
            raise ValueError('specify box_format')
        boxes = kwimage.Boxes(boxes, box_format)

    color = kwimage.Color(color)._forimage(img, colorspace)
    ltrb = boxes.to_ltrb().data
    img2 = img.copy()
    for x1, y1, x2, y2 in ltrb:
        # pt1 = (int(round(x1)), int(round(y1)))
        # pt2 = (int(round(x2)), int(round(y2)))
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # Note cv2.rectangle does work inplace
        img2 = cv2.rectangle(img2, pt1, pt2, color, thickness=thickness)
    return img2


def draw_line_segments_on_image(
        img, pts1, pts2, color='blue', colorspace='rgb', thickness=1,
        **kwargs):
    """
    Draw line segments between pts1 and pts2 on an image.

    Args:
        pts1 (ndarray): xy coordinates of starting points
        pts2 (ndarray): corresponding xy coordinates of ending points
        color (str | List):
            color code or a list of colors for each line segment
        colorspace (str, default='rgb'): colorspace of image
        thickness (int, default=1)
        lineType (int, default=cv2.LINE_AA) option for cv2.line

    Returns:
        ndarray: the modified image (inplace if possible)

    Example:
        >>> from kwimage.im_draw import *  # NOQA
        >>> pts1 = np.array([[2, 0], [2, 20], [2.5, 30]])
        >>> pts2 = np.array([[10, 5], [30, 28], [100, 50]])
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        >>> color = 'blue'
        >>> colorspace = 'rgb'
        >>> img2 = draw_line_segments_on_image(img, pts1, pts2, thickness=2)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.imshow(img2)

    Example:
        >>> import kwimage
        >>> pts1 = kwimage.Points.random(10).scale(512).xy
        >>> pts2 = kwimage.Points.random(10).scale(512).xy
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        >>> color = kwimage.Color.distinct(10)
        >>> img2 = kwimage.draw_line_segments_on_image(img, pts1, pts2, color=color)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.imshow(img2)
    """
    import cv2
    # color = kwimage.Color(color)._forimage(img, colorspace)
    num = len(pts1)
    colors = _broadcast_colors(color, num, img, colorspace)

    if 'lineType' not in kwargs:
        kwargs['lineType'] = cv2.LINE_AA

    pts1_ = pts1.tolist()
    pts2_ = pts2.tolist()
    for xy1, xy2, col in zip(pts1_, pts2_, colors):
        xy1 = tuple(map(int, xy1))
        xy2 = tuple(map(int, xy2))
        cv2.line(img, xy1, xy2, color=col, thickness=thickness, **kwargs)
    return img


def _broadcast_colors(color, num, img, colorspace):
    """
    Determine if color applies a single color to all ``num`` items, or if it is
    a list of colors for each item. Return as a list of colors for each item.

    TODO:
        - [ ] add as classmethod of kwimage.Color

    Example:
        >>> img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        >>> colorspace = 'rgb'
        >>> color = color_str_list = ['red', 'green', 'blue']
        >>> color_str = 'red'
        >>> num = 3
        >>> print(_broadcast_colors(color_str_list, num, img, colorspace))
        >>> print(_broadcast_colors(color_str, num, img, colorspace))
        >>> colors_tuple_list = _broadcast_colors(color_str_list, num, img, colorspace)
        >>> print(_broadcast_colors(colors_tuple_list, num, img, colorspace))
        >>> #
        >>> # FIXME: This case seems broken
        >>> colors_ndarray_list = np.array(_broadcast_colors(color_str_list, num, img, colorspace))
        >>> print(_broadcast_colors(colors_ndarray_list, num, img, colorspace))
    """
    # Note there is an ambiguity when num=3 and color=[int, int, int]
    # that must be resolved by checking num channels in the image
    import kwimage
    import ubelt as ub
    import numbers

    needs_broadcast = True  # assume the list wasnt given by default
    if ub.iterable(color):
        first = ub.peek(color)
        if len(color) == num:
            if len(color) <= 4 and isinstance(first, numbers.Number):
                # ambiguous case, interpret as a single broadcastable color
                needs_broadcast = True
            else:
                # This is the only case we dont need broadcast
                needs_broadcast = False

    if needs_broadcast:
        color = kwimage.Color(color)._forimage(img, colorspace)
        colors = [color] * num
    else:
        colors = [kwimage.Color(c)._forimage(img, colorspace) for c in color]
    return colors


def make_heatmask(probs, cmap='plasma', with_alpha=1.0, space='rgb',
                  dsize=None):
    """
    Colorizes a single-channel intensity mask (with an alpha channel)

    Args:
        probs (ndarray): 2D probability map with values between 0 and 1
        cmap (str): mpl colormap
        with_alpha (float): between 0 and 1, uses probs as the alpha multipled
            by this number.
        space (str): output colorspace
        dsize (tuple): if not None, then output is resized to W,H=dsize

    SeeAlso:
        kwimage.overlay_alpha_images

    Example:
        >>> # xdoc: +REQUIRES(module:matplotlib)
        >>> probs = np.tile(np.linspace(0, 1, 10), (10, 1))
        >>> heatmask = make_heatmask(probs, with_alpha=0.8, dsize=(100, 100))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.imshow(heatmask, fnum=1, doclf=True, colorspace='rgb')
        >>> kwplot.show_if_requested()
    """
    import kwimage
    import matplotlib as mpl
    import matplotlib.cm  # NOQA
    assert len(probs.shape) == 2
    cmap_ = mpl.cm.get_cmap(cmap)
    probs = kwimage.ensure_float01(probs)
    heatmask = cmap_(probs).astype(np.float32)
    heatmask = kwimage.convert_colorspace(heatmask, 'rgba', space, implicit=True)
    if with_alpha is not False and with_alpha is not None:
        heatmask[:, :, 3] = (probs * with_alpha)  # assign probs to alpha channel
    if dsize is not None:
        import cv2
        heatmask = cv2.resize(
            heatmask, tuple(dsize),
            interpolation=cv2.INTER_NEAREST)
    return heatmask


def make_orimask(radians, mag=None, alpha=1.0):
    """
    Makes a colormap in HSV space where the orientation changes color and mag
    changes the saturation/value.

    Args:
        radians (ndarray): orientation in radians
        mag (ndarray): magnitude (must be normalized between 0 and 1)
        alpha (float | ndarray):
            if False or None, then the image is returned without alpha
            if a float, then mag is scaled by this and used as the alpha channel
            if an ndarray, then this is explicilty set as the alpha channel

    Returns:
        ndarray[float32]: an rgb / rgba image in 01 space

    SeeAlso:
        kwimage.overlay_alpha_images

    Example:
        >>> # xdoc: +REQUIRES(module:matplotlib)
        >>> x, y = np.meshgrid(np.arange(64), np.arange(64))
        >>> dx, dy = x - 32, y - 32
        >>> radians = np.arctan2(dx, dy)
        >>> mag = np.sqrt(dx ** 2 + dy ** 2)
        >>> orimask = make_orimask(radians, mag)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.imshow(orimask, fnum=1, doclf=True, colorspace='rgb')
        >>> kwplot.show_if_requested()
    """
    import matplotlib as mpl
    import matplotlib.cm  # NOQA
    TAU = np.pi * 2
    # Map radians to 0 to 1
    ori01 = (radians % TAU) / TAU
    cmap_ = mpl.cm.get_cmap('hsv')
    color_rgb = cmap_(ori01)[..., 0:3].astype(np.float32)
    if mag is not None:
        import kwimage
        if mag.max() > 1:
            mag = mag / mag.max()
        color_hsv = kwimage.convert_colorspace(color_rgb, 'rgb', 'hsv')
        color_hsv[..., 1:3] = mag[..., None]
        color_rgb = kwimage.convert_colorspace(color_hsv, 'hsv', 'rgb')
    else:
        mag = 1
    orimask = np.array(color_rgb, dtype=np.float32)

    if isinstance(alpha, np.ndarray):
        # Alpha specified as explicit numpy array
        orimask = kwimage.ensure_alpha_channel(orimask)
        orimask[:, :, 3] = alpha
    elif alpha is not False and alpha is not None:
        orimask = kwimage.ensure_alpha_channel(orimask)
        orimask[:, :, 3] = mag * alpha
    return orimask


def make_vector_field(dx, dy, stride=0.02, thresh=0.0, scale=1.0, alpha=1.0,
                      color='red', thickness=1, tipLength=0.1, line_type='aa'):
    """
    Create an image representing a 2D vector field.

    Args:
        dx (ndarray): grid of vector x components
        dy (ndarray): grid of vector y components
        stride (int | float): sparsity of vectors, int specifies stride step in
            pixels, a float specifies it as a percentage.
        thresh (float): only plot vectors with magnitude greater than thres
        scale (float): multiply magnitude for easier visualization
        alpha (float): alpha value for vectors. Non-vector regions receive 0
            alpha (if False, no alpha channel is used)
        color (str | tuple | kwimage.Color): RGB color of the vectors
        thickness (int, default=1): thickness of arrows
        tipLength (float, default=0.1): fraction of line length
        line_type (int): either cv2.LINE_4, cv2.LINE_8, or cv2.LINE_AA

    Returns:
        ndarray[float32]: vec_img: an rgb/rgba image in 0-1 space

    SeeAlso:
        kwimage.overlay_alpha_images

    DEPRECATED USE: draw_vector_field instead

    Example:
        >>> x, y = np.meshgrid(np.arange(512), np.arange(512))
        >>> dx, dy = x - 256.01, y - 256.01
        >>> radians = np.arctan2(dx, dy)
        >>> mag = np.sqrt(dx ** 2 + dy ** 2)
        >>> dx, dy = dx / mag, dy / mag
        >>> img = make_vector_field(dx, dy, scale=10, alpha=False)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()
    """
    import warnings
    warnings.warn('Deprecated, use draw_vector_field instead', DeprecationWarning)
    import cv2
    import kwimage
    color = kwimage.Color(color).as255('rgb')
    vecmask = np.zeros(dx.shape + (3,), dtype=np.uint8)

    line_type_lookup = {'aa': cv2.LINE_AA}
    line_type = line_type_lookup.get(line_type, line_type)

    width = dx.shape[1]
    height = dy.shape[0]

    x_grid = np.arange(0, width, 1)
    y_grid = np.arange(0, height, 1)
    # Vector locations and directions
    X, Y = np.meshgrid(x_grid, y_grid)
    U, V = dx, dy

    XYUV = [X, Y, U, V]

    if isinstance(stride, float):
        if stride < 0 or stride > 1:
            raise ValueError('Floating point strides must be between 0 and 1')
        stride = int(np.ceil(stride * min(width, height)))

    # stride the points
    if stride is not None and stride > 1:
        XYUV = [a[::stride, ::stride] for a in XYUV]

    # flatten the points
    XYUV = [a.ravel() for a in XYUV]

    # Filter out points with low magnitudes
    if thresh is not None and thresh > 0:
        M = np.sqrt((XYUV[2] ** 2) + (XYUV[3] ** 2)).ravel()
        XYUV = np.array(XYUV)
        flags = M > thresh
        XYUV = [a[flags] for a in XYUV]

    # Adjust vector magnitude for visibility
    if scale is not None:
        XYUV[2] *= scale
        XYUV[3] *= scale

    for (x, y, u, v) in zip(*XYUV):
        pt1 = (int(x), int(y))
        pt2 = tuple(map(int, map(np.round, (x + u, y + v))))
        cv2.arrowedLine(vecmask, pt1, pt2, color=color, thickness=thickness,
                        tipLength=tipLength,
                        line_type=line_type)

    vecmask = kwimage.ensure_float01(vecmask)
    if isinstance(alpha, np.ndarray):
        # Alpha specified as explicit numpy array
        vecmask = kwimage.ensure_alpha_channel(vecmask)
        vecmask[:, :, 3] = alpha
    elif alpha is not False and alpha is not None:
        # Alpha specified as a scale factor
        vecmask = kwimage.ensure_alpha_channel(vecmask)
        # vecmask[:, :, 3] = (vecmask[:, :, 0:3].sum(axis=2) > 0) * alpha
        vecmask[:, :, 3] = vecmask[:, :, 0:3].sum(axis=2) * alpha
    return vecmask


def draw_vector_field(image, dx, dy, stride=0.02, thresh=0.0, scale=1.0,
                      alpha=1.0, color='red', thickness=1, tipLength=0.1,
                      line_type='aa'):
    """
    Create an image representing a 2D vector field.

    Args:
        image (ndarray): image to draw on
        dx (ndarray): grid of vector x components
        dy (ndarray): grid of vector y components
        stride (int | float): sparsity of vectors, int specifies stride step in
            pixels, a float specifies it as a percentage.
        thresh (float): only plot vectors with magnitude greater than thres
        scale (float): multiply magnitude for easier visualization
        alpha (float): alpha value for vectors. Non-vector regions receive 0
            alpha (if False, no alpha channel is used)
        color (str | tuple | kwimage.Color): RGB color of the vectors
        thickness (int, default=1): thickness of arrows
        tipLength (float, default=0.1): fraction of line length
        line_type (int): either cv2.LINE_4, cv2.LINE_8, or cv2.LINE_AA

    Returns:
        ndarray[float32]: The image with vectors overlaid. If image=None, then an
            rgb/a image is created and returned.

    Example:
        >>> import kwimage
        >>> width, height = 512, 512
        >>> image = kwimage.grab_test_image(dsize=(width, height))
        >>> x, y = np.meshgrid(np.arange(height), np.arange(width))
        >>> dx, dy = x - width / 2, y - height / 2
        >>> radians = np.arctan2(dx, dy)
        >>> mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-3
        >>> dx, dy = dx / mag, dy / mag
        >>> img = kwimage.draw_vector_field(image, dx, dy, scale=10, alpha=False)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()
    """
    import cv2
    import kwimage

    if image is None:
        # Create a default image
        image = np.zeros(dx.shape + (3,), dtype=np.uint8)
        # image = kwimage.atleast_3channels(image)

    color = kwimage.Color(color)._forimage(image)

    line_type_lookup = {'aa': cv2.LINE_AA}
    line_type = line_type_lookup.get(line_type, line_type)

    height, width = dx.shape[0:2]

    x_grid = np.arange(0, width, 1)
    y_grid = np.arange(0, height, 1)
    # Vector locations and directions
    X, Y = np.meshgrid(x_grid, y_grid)
    U, V = dx, dy

    XYUV = [X, Y, U, V]

    if isinstance(stride, float):
        if stride < 0 or stride > 1:
            raise ValueError('Floating point strides must be between 0 and 1')
        stride = int(np.ceil(stride * min(width, height)))

    # stride the points
    if stride is not None and stride > 1:
        XYUV = [a[::stride, ::stride] for a in XYUV]

    # flatten the points
    XYUV = [a.ravel() for a in XYUV]

    # Filter out points with low magnitudes
    if thresh is not None and thresh > 0:
        M = np.sqrt((XYUV[2] ** 2) + (XYUV[3] ** 2)).ravel()
        XYUV = np.array(XYUV)
        flags = M > thresh
        XYUV = [a[flags] for a in XYUV]

    # Adjust vector magnitude for visibility
    if scale is not None:
        XYUV[2] *= scale
        XYUV[3] *= scale

    if alpha is not None and alpha is not False and alpha != 1:
        raise NotImplementedError

    for (x, y, u, v) in zip(*XYUV):
        pt1 = (int(x), int(y))
        pt2 = tuple(map(int, map(np.round, (x + u, y + v))))
        cv2.arrowedLine(image, pt1, pt2, color=color, thickness=thickness,
                        tipLength=tipLength,
                        line_type=line_type)

    if isinstance(alpha, np.ndarray):
        # Alpha specified as explicit numpy array
        image = kwimage.ensure_float01(image)
        image = kwimage.ensure_alpha_channel(image)
        image[:, :, 3] = alpha
    elif alpha is not False and alpha is not None:
        # Alpha specified as a scale factor
        image = kwimage.ensure_float01(image)
        image = kwimage.ensure_alpha_channel(image)
        # image[:, :, 3] = (image[:, :, 0:3].sum(axis=2) > 0) * alpha
        image[:, :, 3] = image[:, :, 0:3].sum(axis=2) * alpha
    return image
