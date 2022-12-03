import itertools as it
import numpy as np
import cv2


def draw_text_on_image(img, text, org=None, return_info=False, **kwargs):
    r"""
    Draws multiline text on an image using opencv

    Args:
        img (ndarray | None | dict):
            Generally a numpy image to draw on (inplace).
            Otherwise a canvas will be constructed such that the text will fit.
            The user may specify a dictionary with keys width and height
            to have more control over the constructed canvas.

        text (str): text to draw

        org (Tuple[int, int]):
            The x, y location of the text string "anchor" in the image as
            specified by halign and valign.  For instance, If valign='bottom',
            halign='left', this where the bottom left corner of the text will
            be placed.

        return_info (bool):
            if True, also returns information about the positions the text
            was drawn on.

        **kwargs:
            color (tuple): default blue

            thickness (int): defaults to 2

            fontFace (int): defaults to cv2.FONT_HERSHEY_SIMPLEX

            fontScale (float): defaults to 1.0

            valign (str):
            either top, center, or bottom.
            Defaults to "bottom"
            NOTE: this default may change to "top" in the future.

            halign (str):
            either left, center, or right. Defaults to "left".

            border (dict | int):
            If specified as an integer, draws a black border with that
            given thickness.  If specified as a dictionary, draws a border
            with color specified parameters.
            "color": border color, defaults to "black".
            "thickness": border thickness, defaults to 1.

    Returns:
        ndarray | Tuple[ndarray, dict] :
            The image that was drawn on and optionally an information
            dictionary if return_info was True.

    Note:
        The image is modified inplace. If the image is non-contiguous then this
        returns a UMat instead of a ndarray, so be careful with that.

    Related:
        The logic in this function is related to the following stack overflow
        posts [SO27647424]_ [SO51285616]_

    References:
        .. [SO27647424] https://stackoverflow.com/questions/27647424/
        .. [SO51285616] https://stackoverflow.com/questions/51285616/opencvs-gettextsize-and-puttext-return-wrong-size-and-chop-letters-with-low

    Example:
        >>> import kwimage
        >>> img = kwimage.grab_test_image(space='rgb')
        >>> img2 = kwimage.draw_text_on_image(img.copy(), 'FOOBAR', org=(0, 0), valign='top')
        >>> assert img2.shape == img.shape
        >>> assert np.any(img2 != img)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img2)
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
        >>> kwplot.imshow(img2)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Ensure the function works with float01 or uint255 images
        >>> import kwimage
        >>> img = kwimage.grab_test_image(space='rgb')
        >>> img = kwimage.ensure_float01(img)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(0, 0), valign='top', border=2)

    Example:
        >>> # Test dictionary border
        >>> import kwimage
        >>> img = kwimage.draw_text_on_image(None, 'Battery\nFraction', org=(100, 100), valign='top', halign='center', border={'color': 'green', 'thickness': 9})
        >>> #img = kwimage.draw_text_on_image(None, 'hello\neveryone', org=(0, 0), valign='top')
        >>> #img = kwimage.draw_text_on_image(None, 'hello', org=(0, 60), valign='top', halign='center', border=0)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Test dictionary image
        >>> import kwimage
        >>> img = kwimage.draw_text_on_image({'width': 300}, 'Unscrew\nGetting', org=(150, 0), valign='top', halign='center', border={'color': 'green', 'thickness': 0})
        >>> print('img.shape = {!r}'.format(img.shape))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()

    Example:
        >>> import ubelt as ub
        >>> import kwimage
        >>> grid = list(ub.named_product({
        >>>     'halign': ['left', 'center', 'right', None],
        >>>     'valign': ['top', 'center', 'bottom', None],
        >>>     'border': [0, 3]
        >>> }))
        >>> canvases = []
        >>> text = 'small-line\na-much-much-much-bigger-line\nanother-small\n.'
        >>> for kw in grid:
        >>>     header = kwimage.draw_text_on_image({}, ub.repr2(kw, compact=1), color='blue')
        >>>     canvas = kwimage.draw_text_on_image({'color': 'white'}, text, org=None, **kw)
        >>>     canvases.append(kwimage.stack_images([header, canvas], axis=0, bg_value=(255, 255, 255), pad=5))
        >>> # xdoc: +REQUIRES(--show)
        >>> canvas = kwimage.stack_images_grid(canvases, pad=10, bg_value=(255, 255, 255))
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """
    import kwimage

    if 'color' not in kwargs:
        # kwargs['color'] = 'red'
        kwargs['color'] = 'strawberry'

    # Get the color that is compatible with the input image encoding
    if img is None or isinstance(img, dict):
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
    if border is not None:
        if isinstance(border, int):
            border = {'color': 'black', 'thickness': border}
        subkw = kwargs.copy()
        subkw['color'] = border.get('color', 'black')
        subkw.pop('return_info', None)
        border_thickness = border.get('thickness', 1)
    else:
        border_thickness = 0

    valign = kwargs.pop('valign', None)
    halign = kwargs.pop('halign', None)
    if halign is None:
        halign = 'left'
    if valign is None:
        valign = 'top'

    if img is None:
        img = {'width': None, 'height': None}

    if org is None:
        org = (None, None)

    x0, y0 = org

    if isinstance(img, dict):
        given_w = img.get('width', None)
        given_h = img.get('height', None)
    else:
        given_h, given_w = img.shape[0:2]

    needs_x0 = x0 is None and halign != 'left'
    needs_y0 = y0 is None and valign != 'top'

    if needs_x0 or needs_y0:
        # Special case: when the alignment is non left-top, AND we don't have
        # an origin we need to do a bit of extra computation to figure out what
        # the width / height need to be
        text_w, text_h = _text_sizes(text, (1, 1), border_thickness, kwargs, None, halign='left')[0:2]
        if given_w is None:
            given_w = text_w
        if given_h is None:
            given_h = text_h

    if x0 is None:
        if halign == 'left':
            x0 = 1
        elif halign == 'center':
            x0 = given_w // 2
        elif halign == 'right':
            x0 = given_w - 1
        else:
            raise KeyError(halign)

    if y0 is None:
        if valign == 'top':
            y0 = 1
        elif valign == 'center':
            y0 = given_h // 2
        elif valign == 'bottom':
            y0 = given_h - 1
        else:
            raise KeyError(valign)

    org = (x0, y0)
    text_w, text_h, x0, lines, abs_top_y, first_h, total_h, total_w, final_baseline, line_sizes, line_org = _text_sizes(text, org, border_thickness, kwargs, valign, halign)

    if isinstance(img, dict):
        # if image is unspecified allocate just enough space for text
        # allow users to specify partial parameters
        bg_color = kwimage.Color(img.get('color', (0, 0, 0))).as255()
        alloc_w = given_w
        alloc_h = given_h
        if alloc_w is None:
            alloc_w = text_w
        if alloc_h is None:
            alloc_h = text_h
        img = np.zeros((alloc_h, alloc_w, 3), dtype=np.uint8)
        img[...] = np.array(bg_color)[None, None, :]

    if border_thickness > 0:
        # recursive call
        basis = list(range(-border_thickness, border_thickness + 1))
        org = np.array(org)
        for i, j in it.product(basis, basis):
            if i == 0 and j == 0:
                continue
            img = draw_text_on_image(img, text, org=org + [i, j], **subkw)

    for i, line in enumerate(lines):
        xy = tuple(line_org[i])
        img = cv2.putText(img, line, xy, **kwargs)

    if return_info:
        info = {
            'line_org': line_org,
            'line_sizes': line_sizes,
        }
        return img, info
    else:
        return img


def _text_sizes(text, org, border_thickness, kwargs, valign, halign):
    getsize_kw = {
        k: kwargs[k]
        for k in ['fontFace', 'fontScale', 'thickness']
        if k in kwargs
    }
    x0, y0 = list(map(int, org))
    thickness = kwargs.get('thickness', 2)

    vertical_spacing = 4  # space between vertical lines
    ypad = thickness + vertical_spacing

    lines = text.split('\n')

    line_sizes = []
    final_baseline = 0
    for line in lines:
        # TODO: better handling of baseline
        # https://en.wikipedia.org/wiki/Baseline_(typography)
        (line_width, line_height), baseline = cv2.getTextSize(line, **getsize_kw)
        line_sizes.append((line_width, line_height))
        final_baseline = baseline
    line_sizes = np.array(line_sizes)

    line_org = []
    y = y0
    for w, h in line_sizes:
        next_y = y + (h + ypad)
        line_org.append((x0, y))
        y = next_y
    line_org = np.array(line_org)

    # the absolute top and bottom position of text
    abs_top_y = line_org[0, 1]
    abs_bot_y = (line_org[-1, 1] + line_sizes[-1, 1]) + thickness

    first_h = line_sizes[0, 1]

    total_h = (abs_bot_y - abs_top_y)
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

    abs_left_x = line_org[:, 0].min()
    text_w = total_w + border_thickness + abs_left_x
    text_h = total_h + border_thickness + abs_top_y + final_baseline

    return text_w, text_h, x0, lines, abs_top_y, first_h, total_h, total_w, final_baseline, line_sizes, line_org


def draw_clf_on_image(im, classes, tcx=None, probs=None, pcx=None, border=1):
    """
    Draws classification label on an image.

    Works best with image chips sized between 200x200 and 500x500

    Args:
        im (ndarray): the image
        classes (Sequence[str] | kwcoco.CategoryTree): list of class names
        tcx (int): true class index if known
        probs (ndarray): predicted class probs for each class
        pcx (int): predicted class index.
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
        boxes (kwimage.Boxes | ndarray): boxes to draw
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
        colorspace (str): colorspace of image. Defaults to 'rgb'
        thickness (int): Defaults to 1
        lineType (int): option for cv2.line

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
        >>> # xdoc: +REQUIRES(module:matplotlib)
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
        >>> from kwimage.im_draw import *  # NOQA
        >>> probs = np.tile(np.linspace(0, 1, 10), (10, 1))
        >>> heatmask = make_heatmask(probs, with_alpha=0.8, dsize=(100, 100))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(heatmask, fnum=1, doclf=True, colorspace='rgb',
        >>>               title='make_heatmask')
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
        ndarray[Any, Float32]:
            an rgb / rgba image in 01 space

    SeeAlso:
        kwimage.overlay_alpha_images

    Example:
        >>> # xdoc: +REQUIRES(module:matplotlib)
        >>> from kwimage.im_draw import *  # NOQA
        >>> x, y = np.meshgrid(np.arange(64), np.arange(64))
        >>> dx, dy = x - 32, y - 32
        >>> radians = np.arctan2(dx, dy)
        >>> mag = np.sqrt(dx ** 2 + dy ** 2)
        >>> orimask = make_orimask(radians, mag)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(orimask, fnum=1, doclf=True,
        >>>               colorspace='rgb', title='make_orimask')
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
                      color='strawberry', thickness=1, tipLength=0.1, line_type='aa'):
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

        thickness (int): thickness of arrows

        tipLength (float): fraction of line length

        line_type (int | str):
            either cv2.LINE_4, cv2.LINE_8, or cv2.LINE_AA or a string
            code.

    Returns:
        ndarray[Any, Float32]:
            vec_img - an rgb/rgba image in 0-1 space

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
    # import warnings
    import ubelt as ub
    ub.schedule_deprecation('kwimage', 'make_vector_field', 'method',
                            migration='use draw_vector_field instead',
                            deprecate='0.9.1', error='1.0.0', remove='1.1.0')
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
                      alpha=1.0, color='strawberry', thickness=1, tipLength=0.1,
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

        thickness (int): thickness of arrows

        tipLength (float): fraction of line length

        line_type (int | str):
            either cv2.LINE_4, cv2.LINE_8, or cv2.LINE_AA or 'aa'

    Returns:
        ndarray[Any, Float32]:
            The image with vectors overlaid. If image=None, then an rgb/a image
            is created and returned.

    Example:
        >>> from kwimage.im_draw import *  # NOQA
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
        >>> kwplot.imshow(img, title='draw_vector_field')
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


def draw_header_text(image, text, fit=False, color='strawberry', halign='center',
                     stack='auto', bg_color='black', **kwargs):
    """
    Places a black bar on top of an image and writes text in it

    Args:

        image (ndarray | dict | None):
            numpy image or dictionary containing a key width

        text (str) :
            text to draw

        fit (bool | str):
            If False, will draw as much text within the given width as possible.
            If True, will draw all text and then resize to fit in the given width
            If "shrink", will only resize the text if it is too big to fit, in
            other words this is like fit=True, but it wont enlarge the text.

        color (str | Tuple) :
            a color coercable to :class:`kwimage.Color`.

        halign (str) :
            Horizontal alignment. Can be left, center, or right.

        stack (bool | str):
            if True returns the stacked image, otherwise just returns the
            header. If 'auto', will only stack if an image is given as an
            ndarray.

        **kwargs: used only for parameter aliases. Currently accepts
            bg_value as an alias for bg_color.

    Returns:
        ndarray

    Example:
        >>> from kwimage.im_draw import *  # NOQA
        >>> import kwimage
        >>> image = kwimage.grab_test_image()
        >>> tiny_image = kwimage.imresize(image, dsize=(64, 64))
        >>> canvases = []
        >>> canvases += [draw_header_text(image=image, text='unfit long header ' * 5, fit=False)]
        >>> canvases += [draw_header_text(image=image, text='shrunk long header ' * 5, fit='shrink')]
        >>> canvases += [draw_header_text(image=image, text='left header', fit=False, halign='left')]
        >>> canvases += [draw_header_text(image=image, text='center header', fit=False, halign='center')]
        >>> canvases += [draw_header_text(image=image, text='right header', fit=False, halign='right')]
        >>> canvases += [draw_header_text(image=image, text='shrunk header', fit='shrink', halign='left')]
        >>> canvases += [draw_header_text(image=tiny_image, text='shrunk header-center', fit='shrink', halign='center')]
        >>> canvases += [draw_header_text(image=image, text='fit header', fit=True, halign='left')]
        >>> canvases += [draw_header_text(image={'width': 200}, text='header only', fit=True, halign='left')]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nCols=3, nSubplots=len(canvases))
        >>> for c in canvases:
        >>>     kwplot.imshow(c, pnum=pnum_())
        >>> kwplot.show_if_requested()
    """
    # import cv2
    import kwimage

    if stack == 'auto':
        stack = isinstance(image, np.ndarray)

    if isinstance(image, dict):
        width = image['width']
        if stack:
            raise ValueError('Must pass in the actual image if stack is True')
    else:
        width = image.shape[1]

    if stack:
        # Handle very small image case
        h, w = image.shape[0:2]
        min_pixels = 32
        if w < min_pixels or h < min_pixels:
            image = kwimage.imresize(image, min_dim=min_pixels)
        width = image.shape[1]

    if 'bg_value' in kwargs:
        bg_color = kwargs.pop('bg_value')

    if kwargs:
        raise ValueError('Unexpected kwargs = {}'.format(kwargs))

    bginfo = {'color': bg_color}

    if fit:
        # TODO: allow a shrink-to-fit only option
        try:
            # needs new kwimage to work
            header = kwimage.draw_text_on_image(
                bginfo, text, org=None,
                valign='top', halign=halign, color=color)
        except Exception:
            header = kwimage.draw_text_on_image(
                bginfo, text, org=(1, 1),
                valign='top', halign='left', color=color)

        if fit == 'shrink':
            if header.shape[1] > width:
                header = kwimage.imresize(header, dsize=(width, None))
            elif header.shape[1] < width:
                header = np.pad(header, [(0, 0), ((width - header.shape[1]) // 2, 0), (0, 0)])
            else:
                pass
        else:
            header = kwimage.imresize(header, dsize=(width, None))
    else:
        # Allows for however much height is needed
        if halign == 'left':
            org = (1, 1)
        elif halign == 'center':
            org = (width // 2, 1)
        elif halign == 'right':
            org = (width - 1, 1)
        else:
            raise KeyError(halign)

        bginfo['width'] = width
        header = kwimage.draw_text_on_image(
            bginfo, text, org=org,
            valign='top', halign=halign, color=color)

    if stack:
        stacked = kwimage.stack_images([header, image], axis=0, overlap=-1)
        return stacked
    else:
        return header


def fill_nans_with_checkers(canvas, square_shape=8,
                            on_value='auto', off_value='auto'):
    """
    Fills nan or masked values with a 2d checkerboard pattern.

    Args:
        canvas (np.ndarray): data replace nans in

        square_shape (int | Tuple[int, int] | str):
            Size of the checker squares. Defaults to 8.

        on_value (Number | str):
            The value of one checker. Defaults to 1 for floats and 255 for
            ints.

        off_value (Number | str):
            The value off the other checker. Defaults to 0.

    Returns:
        np.ndarray: the inplace modified canvas

    SeeAlso:
        :func:`nodata_checkerboard` - similar, but operates on nans or masked arrays.

    Example:
        >>> from kwimage.im_draw import *  # NOQA
        >>> import kwimage
        >>> orig_img = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> poly1 = kwimage.Polygon.random(rng=1).scale(orig_img.shape[0] // 2)
        >>> poly2 = kwimage.Polygon.random(rng=3).scale(orig_img.shape[0])
        >>> poly3 = kwimage.Polygon.random(rng=4).scale(orig_img.shape[0] // 2)
        >>> poly3 = poly3.translate((0, 200))
        >>> poly4 = poly2.translate((100, 0))
        >>> poly5 = poly2.translate((50, 100))
        >>> img = orig_img.copy()
        >>> img = poly1.fill(img, np.nan)
        >>> img = poly3.fill(img, 0)
        >>> img[:, :, 0] = poly2.fill(np.ascontiguousarray(img[:, :, 0]), np.nan)
        >>> img[:, :, 2] = poly4.fill(np.ascontiguousarray(img[:, :, 2]), np.nan)
        >>> img[:, :, 1] = poly5.fill(np.ascontiguousarray(img[:, :, 1]), np.nan)
        >>> input_img = img.copy()
        >>> canvas = fill_nans_with_checkers(input_img, on_value=0.3)
        >>> assert input_img is canvas
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img, pnum=(1, 2, 1), title='matplotlib treats nans as zeros')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='checkers highlight real nans')

    Example:
        >>> # Test grayscale
        >>> from kwimage.im_draw import *  # NOQA
        >>> import kwimage
        >>> orig_img = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> poly1 = kwimage.Polygon.random().scale(orig_img.shape[0] // 2)
        >>> poly2 = kwimage.Polygon.random().scale(orig_img.shape[0])
        >>> img = orig_img.copy()
        >>> img = poly1.fill(img, np.nan)
        >>> img[:, :, 0] = poly2.fill(np.ascontiguousarray(img[:, :, 0]), np.nan)
        >>> img = kwimage.convert_colorspace(img, 'rgb', 'gray')
        >>> canvas = img.copy()
        >>> canvas = fill_nans_with_checkers(canvas)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img, pnum=(1, 2, 1))
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2))

    Ignore:
        >>> import kwarray
        >>> import numpy as np
        >>> img = np.array([[
        >>>     [   0.5,    0.5,    0.5],
        >>>     [np.nan,    0.5,    0.5],
        >>>     [   0.5, np.nan,    0.5],
        >>>     [np.nan, np.nan,    0.5],
        >>>     [   0.5,    0.5, np.nan],
        >>>     [np.nan,    0.5, np.nan],
        >>>     [   0.5, np.nan, np.nan],
        >>>     [np.nan, np.nan, np.nan],
        >>> ]])
        >>> canvas = kwimage.fill_nans_with_checkers(img, square_shape=1)
        >>> print(ub.repr2({'canvas': canvas}, nl=2, with_dtype=False))
        >>> print(canvas)
    """
    invalid_mask = np.isnan(canvas)
    return _masked_checkerboard(canvas, invalid_mask, square_shape, on_value=on_value, off_value=off_value)


def _masked_checkerboard(canvas, invalid_mask, square_shape, on_value, off_value):
    import kwimage
    import kwarray
    canvas = kwarray.atleast_nd(canvas, 3)
    invalid_mask = kwarray.atleast_nd(invalid_mask, 3)
    allchan_invalid_mask = invalid_mask.all(axis=2, keepdims=True)
    anychan_invalid_mask = invalid_mask.any(axis=2, keepdims=True)

    some_invalid_mask = (~allchan_invalid_mask) * anychan_invalid_mask
    dsize = canvas.shape[0:2][::-1]

    if on_value == 'auto':
        if canvas.dtype.kind == 'u' and canvas.dtype.itemsize == 1:
            on_value = 255
        else:
            on_value = 1
    if off_value == 'auto':
        off_value = 0

    any_total_nans = np.any(allchan_invalid_mask)
    any_partial_nans = np.any(some_invalid_mask)

    if any_total_nans or any_partial_nans:
        checkers2d = kwimage.checkerboard(
            square_shape=square_shape, dsize=dsize, dtype=canvas.dtype,
            on_value=on_value)

        if any_total_nans:
            # canvas = kwimage.ensure_alpha_channel(canvas, (1 - invalid_mask))
            # checkers = kwimage.ensure_alpha_channel(checkers, 1)
            locs = np.where(allchan_invalid_mask)
            canvas[locs[0:2]] = checkers2d[..., None][locs[0:2]]

        if any_partial_nans:
            for chan_idx in range(invalid_mask.shape[2]):
                chan_mask = invalid_mask[..., chan_idx]
                locs3d = np.where(chan_mask[..., None])
                locs3d[2][:] = chan_idx
                locs2d = locs3d[0:2]
                canvas[locs3d] = checkers2d[locs2d]

    return canvas


def nodata_checkerboard(canvas, square_shape=8, on_value='auto', off_value='auto'):
    """
    Fills nans or masked values with a checkerbord pattern.

    Args:
        canvas (ndarray): A 2D image with any number of channels.

        square_shape (int): the pixel size of the checkers

        on_value (Number | str):
            The value of one checker. Defaults to 1 for floats and 255 for
            ints.

        off_value (Number | str):
            The value off the other checker. Defaults to 0.


    Returns:
        ndarray : an output array with imputed values.
            if the input was a masked array, the mask will still exist.

    SeeAlso:
        :func:`fill_nans_with_checkers` - similar, but only operates on nan
        values.

    Example:
        >>> import kwimage
        >>> # Test a masked array WITH nan values
        >>> data = kwimage.grab_test_image(space='rgb')
        >>> na_circle = kwimage.Polygon.circle((256 - 96, 256), 128)
        >>> ma_circle = kwimage.Polygon.circle((256 + 96, 256), 128)
        >>> ma_mask = na_circle.fill(np.zeros(data.shape, dtype=np.uint8), value=1).astype(bool)
        >>> na_mask = ma_circle.fill(np.zeros(data.shape, dtype=np.uint8), value=1).astype(bool)
        >>> # Hack the channels to make a ven diagram
        >>> ma_mask[..., [0, 1]] = False
        >>> na_mask[..., [0, 2]] = False
        >>> data = kwimage.ensure_float01(data)
        >>> data[na_mask] = np.nan
        >>> canvas = np.ma.MaskedArray(data, ma_mask)
        >>> kwimage.draw_text_on_image(canvas, 'masked values', (256 - 96, 256 - 128), halign='center', valign='bottom', border=2)
        >>> kwimage.draw_text_on_image(canvas, 'nan values',    (256 + 96, 256 + 128), halign='center', valign='top', border=2)
        >>> kwimage.draw_text_on_image(canvas, 'kwimage.nodata_checkerboard',    (256, 5), halign='center', valign='top', border=2)
        >>> kwimage.draw_text_on_image(canvas, '(pip install kwimage)', (512, 512 - 10), halign='right', valign='bottom', border=2, fontScale=0.8)
        >>> result = kwimage.nodata_checkerboard(canvas, on_value=0.5)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(result)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Simple test with a masked array
        >>> import kwimage
        >>> data = kwimage.grab_test_image(space='rgb', dsize=(64, 64))
        >>> data = kwimage.ensure_uint255(data)
        >>> circle = kwimage.Polygon.circle((32, 32), 16)
        >>> mask = circle.fill(np.zeros(data.shape, dtype=np.uint8), value=1).astype(bool)
        >>> img = np.ma.MaskedArray(data, mask)
        >>> canvas = img.copy()
        >>> result = kwimage.nodata_checkerboard(canvas)
        >>> canvas.data is result.data
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(result, title='nodata_checkers with masked uint8')
        >>> kwplot.show_if_requested()
    """
    is_masked = isinstance(canvas, np.ma.MaskedArray)
    masks = []
    if is_masked:
        masks.append(canvas.mask)
        canvas = canvas.data
    if canvas.dtype.kind == 'f':
        masks.append(np.isnan(canvas))

    if masks:
        invalid_mask = np.logical_or.reduce(masks)
    else:
        invalid_mask = None

    if invalid_mask is not None:
        canvas = _masked_checkerboard(canvas, invalid_mask, square_shape,
                                      on_value, off_value)

    if is_masked:
        canvas = np.ma.MaskedArray(data=canvas, mask=invalid_mask)

    return canvas
