# -*- coding: utf-8 -*-
"""
Wrappers around cv2 functions

Note: all functions in kwimage work with RGB input by default instead of BGR.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import six
import numpy as np
import itertools as it
from . import im_core


_CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}


def _rectify_interpolation(interpolation, default=cv2.INTER_LANCZOS4):
    """
    Converts interpolation into flags suitable cv2 functions

    Args:
        interpolation (int or str): string or cv2-style interpolation type
        default (int): cv2 flag to use if `interpolation` is None

    Returns:
        int: flag specifying interpolation type that can be passed to
           functions like cv2.resize, cv2.warpAffine, etc...
    """
    if interpolation is None:
        return default
    elif isinstance(interpolation, six.text_type):
        try:
            return _CV2_INTERPOLATION_TYPES[interpolation]
        except KeyError:
            print('Valid values for interpolation are {}'.format(
                list(_CV2_INTERPOLATION_TYPES.keys())))
            raise
    else:
        return interpolation


def imscale(img, scale, interpolation=None, return_scale=False):
    """
    Resizes an image by a scale factor.

    Because the result image must have an integer number of pixels, the scale
    factor is rounded, and the rounded scale factor is optionaly returned.

    Args:
        img (ndarray): image to resize

        scale (float or Tuple[float, float]):
            desired floating point scale factor. If a tuple, the dimension
            ordering is x,y.

        interpolation (str | int): interpolation key or code (e.g. linear lanczos)

        return_scale (bool, default=False):
            if True returns both the new image and the actual scale factor used
            to achive the new integer image size.

    SeeAlso:
        imresize

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> new_img, new_scale = kwimage.imscale(img, scale=.85,
        >>>                                      interpolation='nearest',
        >>>                                      return_scale=True)
        >>> assert new_scale == (.8, .8)
        >>> assert new_img.shape == (8, 8, 3)
    """
    dsize = img.shape[0:2][::-1]

    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(np.round(w * sx))
    new_h = int(np.round(h * sy))

    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)

    interpolation = _rectify_interpolation(interpolation)
    new_img = cv2.resize(img, new_dsize, interpolation=interpolation)

    if return_scale:
        return new_img, new_scale
    else:
        return new_img


def imresize(img, scale=None, dsize=None, max_dim=None, min_dim=None,
             interpolation=None, return_info=False):
    """
    Resize an image based on a scale factor, final size, or size and aspect
    ratio.

    Slightly more general than cv2.resize and kwimage.imscale, allows for
    specification of either a scale factor, a final size, or the final size for
    a particular dimension.

    Args:
        img (ndarray): image to resize

        scale (float or Tuple[float, float]):
            desired floating point scale factor. If a tuple, the dimension
            ordering is x,y. Mutually exclusive with dsize, max_dim, and
            min_dim.

        dsize (Tuple[None | int, None | int]): the desired with and height
            of the new image. If a dimension is None, then it is automatically
            computed to preserve aspect ratio. Mutually exclusive with size,
            max_dim, and min_dim.

        max_dim (int): new size of the maximum dimension, the other
            dimension is scaled to maintain aspect ratio. Mutually exclusive
            with size, dsize, and min_dim.

        min_dim (int): new size of the minimum dimension, the other
            dimension is scaled to maintain aspect ratio.Mutually exclusive
            with size, dsize, and max_dim.

        interpolation (str | int): interpolation key or code (e.g. linear lanczos)

        return_info (bool, default=False):
            if True returns information about the final transformation in a
            dictionary.

    Returns:
        ndarray | Tuple[ndarray, Dict] :
            the new image and optionally an info dictionary

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> # Test scale
        >>> img = np.zeros((16, 10, 3), dtype=np.uint8)
        >>> new_img, info = kwimage.imresize(img, scale=.85,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [.8, 0.875]
        >>> # Test dsize without None
        >>> new_img, info = kwimage.imresize(img, dsize=(5, 12),
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.5 , 0.75]
        >>> # Test dsize with None
        >>> new_img, info = kwimage.imresize(img, dsize=(6, None),
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.6, 0.625]
        >>> # Test max_dim
        >>> new_img, info = kwimage.imresize(img, max_dim=6,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.4  , 0.375]
        >>> # Test min_dim
        >>> new_img, info = kwimage.imresize(img, min_dim=6,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.6  , 0.625]
    """
    old_w, old_h = img.shape[0:2][::-1]

    _mutex_args = [scale, dsize, max_dim, min_dim]
    if sum(a is not None for a in _mutex_args) != 1:
        raise ValueError(
            'Must specify EXACTLY one of scale, dsize, max_dim, xor min_dim')

    if scale is not None:
        try:
            sx, sy = scale
        except TypeError:
            sx = sy = scale
        new_w = old_w * sx
        new_h = old_h * sy
    elif dsize is not None:
        new_w, new_h = dsize
    elif max_dim is not None:
        if old_w > old_h:
            new_w, new_h = max_dim, None
        else:
            new_w, new_h = None, max_dim
    elif min_dim is not None:
        if old_w > old_h:
            new_w, new_h = None, min_dim
        else:
            new_w, new_h = min_dim, None
    else:
        raise AssertionError('impossible')

    if new_w is None:
        assert new_h is not None
        new_w = new_h * old_w / old_h
    elif new_h is None:
        assert new_w is not None
        new_h = new_w * old_h / old_w

    # Use np.round over python round, which has incompatible behavior
    new_dsize = (int(np.round(new_w)), int(np.round(new_h)))

    interpolation = _rectify_interpolation(interpolation)
    new_img = cv2.resize(img, new_dsize, interpolation=interpolation)

    if return_info:
        old_dsize = (old_w, old_h)
        new_scale = np.array(new_dsize) / np.array(old_dsize)
        info = {
            'scale': new_scale,
            'dsize': new_dsize,
        }
        return new_img, info
    else:
        return new_img


def convert_colorspace(img, src_space, dst_space, copy=False,
                       implicit=False, dst=None):
    """
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]): image data

        src_space (str): input image colorspace. (e.g. BGR, GRAY)

        dst_space (str): desired output colorspace. (e.g. RGB, HSV, LAB)

        implicit (bool):
            if False, the user must correctly specify if the input/output
                colorspaces contain alpha channels.
            If True and the input image has an alpha channel, we modify
                src_space and dst_space to ensure they both end with "A".

        dst (ndarray[uint8_t, ndim=2], optional): inplace-output array.

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    Note:
        Note the LAB and HSV colorspaces in float do not go into the 0-1 range.

        For HSV the floating point range is:
            0:360, 0:1, 0:1
        For LAB the floating point range is:
            0:100, -86.1875:98.234375, -107.859375:94.46875
            (Note, that some extreme combinations of a and b are not valid)

    Example:
        >>> import numpy as np
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'RGB', 'LAB')
        >>> convert_colorspace(np.array([[[0, 1, 0]]], dtype=np.float32), 'RGB', 'LAB')
        >>> convert_colorspace(np.array([[[1, 0, 0]]], dtype=np.float32), 'RGB', 'LAB')
        >>> convert_colorspace(np.array([[[1, 1, 1]]], dtype=np.float32), 'RGB', 'LAB')
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'RGB', 'HSV')

    Ignore:
        # Check LAB output ranges
        import itertools as it
        s = 1
        _iter = it.product(range(0, 256, s), range(0, 256, s), range(0, 256, s))
        minvals = np.full(3, np.inf)
        maxvals = np.full(3, -np.inf)
        for r, g, b in ub.ProgIter(_iter, total=(256 // s) ** 3):
            img255 = np.array([[[r, g, b]]], dtype=np.uint8)
            img01 = (img255 / 255.0).astype(np.float32)
            lab = convert_colorspace(img01, 'rgb', 'lab')
            np.minimum(lab[0, 0], minvals, out=minvals)
            np.maximum(lab[0, 0], maxvals, out=maxvals)
        print('minvals = {}'.format(ub.repr2(minvals, nl=0)))
        print('maxvals = {}'.format(ub.repr2(maxvals, nl=0)))
    """
    src_space = src_space.upper()
    dst_space = dst_space.upper()

    if implicit:
        # Assume the user meant grayscale if there is only one channel
        if im_core.num_channels(img) == 1:
            src_space = 'gray'
        # We give the caller some slack by assuming RGB means RGBA if the input
        # image has an alpha channel.
        elif im_core.num_channels(img) == 4:
            if src_space[-1] != 'A':
                src_space = src_space + 'A'
            if dst_space[-1] != 'A':
                dst_space = dst_space + 'A'

        if img.dtype.kind == 'f':
            # opencv requires float32 input
            if img.dtype.itemsize == 8:
                img = img.astype(np.float32)

    if src_space == dst_space:
        img2 = img
        if dst is not None:
            dst[...] = img[...]
            img2 = dst
        elif copy:
            img2 = img2.copy()
    else:
        code = _lookup_cv2_colorspace_conversion_code(src_space, dst_space)
        # Note the conversion to colorspaces like LAB and HSV in float form
        # do not go into the 0-1 range. Instead they go into
        # (0-100, -111-111ish, -111-111ish) and (0-360, 0-1, 0-1) respectively
        img2 = cv2.cvtColor(img, code, dst=dst)
    return img2


def _lookup_cv2_colorspace_conversion_code(src_space, dst_space):
    src = src_space.upper()
    dst = dst_space.upper()
    convert_attr = 'COLOR_{}2{}'.format(src, dst)
    if not hasattr(cv2, convert_attr):
        prefix = 'COLOR_{}2'.format(src)
        valid_dst_spaces = [
            key.replace(prefix, '')
            for key in cv2.__dict__.keys() if key.startswith(prefix)]
        raise KeyError(
            '{} does not exist, valid conversions from {} are to {}'.format(
                convert_attr, src_space, valid_dst_spaces))
    else:
        code = getattr(cv2, convert_attr)
    return code


def draw_boxes_on_image(img, boxes, color='blue', thickness=1,
                        box_format=None, colorspace='rgb'):
    """
    Draws boxes on an image.

    Note:
        This function also exists in kwplot

    Args:
        img (ndarray): image to copy and draw on
        boxes (nh.util.Boxes): boxes to draw
        colorspace (str): string code of the input image colorspace

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwimage
        >>> import numpy as np
        >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> color = 'dodgerblue'
        >>> thickness = 1
        >>> boxes = kwimage.Boxes([[1, 1, 8, 8]], 'tlbr')
        >>> img2 = draw_boxes_on_image(img, boxes, color, thickness)
        >>> assert tuple(img2[1, 1]) == (30, 144, 255)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.imshow(img2)
    """
    import kwimage
    import kwplot
    if not isinstance(boxes, kwimage.Boxes):
        if box_format is None:
            raise ValueError('specify box_format')
        boxes = kwimage.Boxes(boxes, box_format)

    color = tuple(kwplot.Color(color).as255(colorspace))
    tlbr = boxes.to_tlbr().data
    img2 = img.copy()
    for x1, y1, x2, y2 in tlbr:
        # pt1 = (int(round(x1)), int(round(y1)))
        # pt2 = (int(round(x2)), int(round(y2)))
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # Note cv2.rectangle does work inplace
        img2 = cv2.rectangle(img2, pt1, pt2, color, thickness=thickness)
    return img2


def draw_text_on_image(img, text, org, **kwargs):
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
        **kwargs:
            color (tuple): default blue
            thickneess (int): defaults to 2
            fontFace (int): defaults to cv2.FONT_HERSHEY_SIMPLEX
            fontScale (float): defaults to 1.0
            valign (str, default=bottom): either top, center, or bottom

    References:
        https://stackoverflow.com/questions/27647424/

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
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
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwimage
        >>> img = kwimage.grab_test_image(space='rgb')
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(0, 0), valign='top', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(150, 0), valign='center', border=2)
        >>> img2 = kwimage.draw_text_on_image(img, 'FOOBAR\nbazbiz\nspam', org=(300, 0), valign='bottom', border=2)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img2, fontScale=10)
        >>> kwplot.show_if_requested()
    """
    import kwplot

    if 'color' not in kwargs:
        kwargs['color'] = (255, 0, 0)

    kwargs['color'] = kwplot.Color(kwargs['color']).as255('rgb')

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
        basis = list(range(-border, border + 1))
        for i, j in it.product(basis, basis):
            if i == 0 and j == 0:
                continue
            org = np.array(org)
            img = draw_text_on_image(img, text, org=org + [i, j], **subkw)

    valign = kwargs.pop('valign', None)

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

    if valign is not None:
        # TODO: halign
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

    if img is None:
        # if image is unspecified allocate just enough space for text
        total_w = line_sizes.T[0].max()
        # TODO: does not account for origin offset
        img = np.zeros((total_h + thickness, total_w), dtype=np.uint8)

    for i, line in enumerate(lines):
        (x, y) = line_org[i]
        img = cv2.putText(img, line, (x, y), **kwargs)
    return img


def gaussian_patch(shape=(7, 7), sigma=None):
    """
    Creates a 2D gaussian patch with a specific size and sigma

    Args:
        shape (Tuple[int, int]): patch height and width
        sigma (float | Tuple[float, float]): gaussian standard deviation

    References:
        http://docs.opencv.org/modules/imgproc/doc/filtering.html#getgaussiankernel

    TODO:
        - [ ] Look into this C-implementation
        https://kwgitlab.kitware.com/computer-vision/heatmap/blob/master/heatmap/heatmap.c

    CommandLine:
        xdoctest -m kwimage.im_cv2 gaussian_patch --show

    Example:
        >>> import numpy as np
        >>> shape = (88, 24)
        >>> sigma = None  # 1.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> assert np.all(np.isclose(sum_, 1.0))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> norm = (gausspatch - gausspatch.min()) / (gausspatch.max() - gausspatch.min())
        >>> kwplot.imshow(norm)
        >>> kwplot.show_if_requested()

    Example:
        >>> import numpy as np
        >>> shape = (24, 24)
        >>> sigma = 3.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> assert np.all(np.isclose(sum_, 1.0))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> norm = (gausspatch - gausspatch.min()) / (gausspatch.max() - gausspatch.min())
        >>> kwplot.imshow(norm)
        >>> kwplot.show_if_requested()
    """
    if sigma is None:
        sigma1 = 0.3 * ((shape[0] - 1) * 0.5 - 1) + 0.8
        sigma2 = 0.3 * ((shape[1] - 1) * 0.5 - 1) + 0.8
    elif isinstance(sigma, (float, int)):
        sigma1 = sigma2 = sigma
    else:
        sigma1, sigma2 = sigma
    # see hesaff/src/helpers.cpp : computeCircularGaussMask
    kernel_d0 = cv2.getGaussianKernel(shape[0], sigma1)
    if shape[0] == shape[1] and sigma2 == sigma1:
        kernel_d1 = kernel_d0
    else:
        kernel_d1 = cv2.getGaussianKernel(shape[1], sigma2)
    gausspatch = kernel_d0.dot(kernel_d1.T)
    return gausspatch
