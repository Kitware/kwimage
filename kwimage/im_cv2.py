# -*- coding: utf-8 -*-
"""
Wrappers around cv2 functions

Note: all functions in kwimage work with RGB input by default instead of BGR.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import six
import numpy as np
import ubelt as ub
import numbers
from . import im_core


_CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}


def _coerce_interpolation(interpolation, default=cv2.INTER_LANCZOS4,
                          grow_default=cv2.INTER_LANCZOS4,
                          shrink_default=cv2.INTER_AREA, scale=None):
    """
    Converts interpolation into flags suitable cv2 functions

    Args:
        interpolation (int or str): string or cv2-style interpolation type

        default (int): cv2 flag to use if `interpolation` is None and scale is
            None.

        grow_default (int): cv2 flag to use if `interpolation` is None and
            scale is greater than or equal to 1.

        shrink_default (int): cv2 flag to use if `interpolation` is None and
            scale is less than 1.

        scale (float): indicate if the interpolation will be used to scale the
            image.

    Returns:
        int: flag specifying interpolation type that can be passed to
           functions like cv2.resize, cv2.warpAffine, etc...

    Example:
        >>> flag = _coerce_interpolation('linear')
        >>> assert flag == cv2.INTER_LINEAR
        >>> flag = _coerce_interpolation(cv2.INTER_LINEAR)
        >>> assert flag == cv2.INTER_LINEAR
        >>> flag = _coerce_interpolation('auto', default='lanczos')
        >>> assert flag == cv2.INTER_LANCZOS4
        >>> flag = _coerce_interpolation(None, default='lanczos')
        >>> assert flag == cv2.INTER_LANCZOS4
        >>> flag = _coerce_interpolation('auto', shrink_default='area', scale=0.1)
        >>> assert flag == cv2.INTER_AREA
        >>> flag = _coerce_interpolation('auto', grow_default='cubic', scale=10.)
        >>> assert flag == cv2.INTER_CUBIC
        >>> # xdoctest: +REQUIRES(module:pytest)
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     _coerce_interpolation(3.4)
        >>> import pytest
        >>> with pytest.raises(KeyError):
        >>>     _coerce_interpolation('foobar')
    """
    # Handle auto-defaulting
    if interpolation is None or interpolation == 'auto':
        if scale is None:
            interpolation = default
        else:
            if scale >= 1:
                interpolation = grow_default
            else:
                interpolation = shrink_default

    # Handle coercion from string to cv2 integer flag
    if isinstance(interpolation, six.text_type):
        try:
            return _CV2_INTERPOLATION_TYPES[interpolation]
        except KeyError:
            raise KeyError(
                'Invalid interpolation value={!r}. '
                'Valid strings for interpolation are {}'.format(
                    interpolation, list(_CV2_INTERPOLATION_TYPES.keys())))
    elif isinstance(interpolation, numbers.Integral):
        return int(interpolation)
    else:
        raise TypeError(
            'Invalid interpolation value={!r}. '
            'Type must be int or string but got {!r}'.format(
                interpolation, type(interpolation)))


def imscale(img, scale, interpolation=None, return_scale=False):
    """
    DEPRECATED and removed: use imresize instead
    """
    raise Exception('imscale is deprecated, use imresize instead')


def imresize(img, scale=None, dsize=None, max_dim=None, min_dim=None,
             interpolation=None, grow_interpolation=None, letterbox=False,
             return_info=False, antialias=False):
    """
    Resize an image based on a scale factor, final size, or size and aspect
    ratio.

    Slightly more general than cv2.resize, allows for specification of either a
    scale factor, a final size, or the final size for a particular dimension.

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

        interpolation (str | int): interpolation key or code (e.g. linear
            lanczos). By default "area" is used if the image is shrinking and
            "lanczos" is used if the image is growing. Note, if this is
            explicitly set, then it will be used regardless of if the image is
            growing or shrinking. Set ``grow_interpolation`` to change
            the default for an enlarging interpolation.

        grow_interpolation (str | int): The interpolation key or code to use
            when the image is being enlarged. Does nothing if "interpolation"
            is explicitly given. Defaults to "lanczos".

        letterbox (bool, default=False): if used in conjunction with
            dsize, then the image is scaled and translated to fit in the
            center of the new image while maintaining aspect ratio. Black
            padding is added if necessary.

            - [ ] TODO: add padding options

        return_info (bool, default=False):
            if True returns information about the final transformation in a
            dictionary. If there is an offset, the scale is applied before the
            offset when transforming to the new resized space.

        antialias (bool, default=False):
            if True blurs to anti-alias before downsampling.

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

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> # Test letterbox resize
        >>> img = np.ones((5, 10, 3), dtype=np.float32)
        >>> new_img, info = kwimage.imresize(img, dsize=(19, 19),
        >>>                                  letterbox=True,
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['offset'].tolist() == [0, 4]
        >>> img = np.ones((10, 5, 3), dtype=np.float32)
        >>> new_img, info = kwimage.imresize(img, dsize=(19, 19),
        >>>                                  letterbox=True,
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['offset'].tolist() == [4, 0]

        >>> import kwimage
        >>> import numpy as np
        >>> # Test letterbox resize
        >>> img = np.random.rand(100, 200)
        >>> new_img, info = kwimage.imresize(img, dsize=(300, 300), letterbox=True, return_info=True)

    Exammple:
        >>> # Check aliasing
        >>> import kwimage
        >>> img = kwimage.grab_test_image('checkerboard')
        >>> img = kwimage.grab_test_image('astro')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dsize = (14, 14)
        >>> dsize = (64, 64)
        >>> # When we set "grow_interpolation" for a "shrinking" resize it should
        >>> # still do the "area" interpolation to antialias the results. But if we
        >>> # use explicit interpolation it should alias.
        >>> pnum_ = kwplot.PlotNums(nSubplots=12, nCols=4)
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True,  interpolation='area'), pnum=pnum_(), title='resize aa area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='linear'), pnum=pnum_(), title='resize aa linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='nearest'), pnum=pnum_(), title='resize aa nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='cubic'), pnum=pnum_(), title='resize aa cubic')

        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='area'), pnum=pnum_(), title='resize aa grow area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='linear'), pnum=pnum_(), title='resize aa grow linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='nearest'), pnum=pnum_(), title='resize aa grow nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='cubic'), pnum=pnum_(), title='resize aa grow cubic')

        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='area'), pnum=pnum_(), title='resize no-aa area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='linear'), pnum=pnum_(), title='resize no-aa linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='nearest'), pnum=pnum_(), title='resize no-aa nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='cubic'), pnum=pnum_(), title='resize no-aa cubic')

    FIXME:
        - [ ] When interpolation is area and the number of channels > 4
              cv2.resize will error but it is fine for linear interpolation
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

    grow_interpolation = _coerce_interpolation(grow_interpolation)

    def _aa_resize(a, scale, dsize, interpolation):
        sx, sy = scale
        if sx < 1 or sy < 1:
            a, sx, sy = _prepare_downscale(a, sx, sy)
        return cv2.resize(a, dsize=dsize, interpolation=interpolation)

    def _regular_resize(a, scale, dsize, interpolation):
        return cv2.resize(a, dsize=dsize, interpolation=interpolation)

    if antialias:
        _chosen_resize = _aa_resize
    else:
        _chosen_resize = _regular_resize

    def _patched_resize(img, scale, dsize, interpolation):
        sx, sy = scale
        num_chan = im_core.num_channels(img)
        if num_chan > 512 or (num_chan > 4 and interpolation == cv2.INTER_AREA):
            parts = np.split(img, img.shape[-1], -1)
            newparts = [
                _chosen_resize(chan, scale, dsize=dsize, interpolation=interpolation)[..., None]
                for chan in parts
            ]
            newimg = np.concatenate(newparts, axis=2)
            return newimg
        newimg = _chosen_resize(img, scale, dsize, interpolation)
        return newimg

    if letterbox:
        if dsize is None:
            raise ValueError('letterbox can only be used with dsize')
        orig_size = np.array(img.shape[0:2][::-1])
        target_size = np.array(dsize)
        # Determine to use the x or y scale factor
        unequal_sxy = (target_size / orig_size)
        equal_sxy = unequal_sxy.min()
        # Whats the closest integer size we can resize to?
        embed_size = np.round(orig_size * equal_sxy).astype(int)
        # Determine how much padding we need for the top/left side
        # Note: the right/bottom side might need an extra pixel of padding
        # depending on rounding issues.
        offset = np.round((target_size - embed_size) / 2).astype(int)
        scale = embed_size / orig_size

        left, top = offset
        right, bot = target_size - (embed_size + offset)

        interpolation = _coerce_interpolation(
            interpolation, scale=equal_sxy, grow_default=grow_interpolation)

        embed_dsize = tuple(embed_size)
        embed_img = _patched_resize(img, scale, embed_dsize,
                                    interpolation=interpolation)
        new_img = cv2.copyMakeBorder(
            embed_img, top, bot, left, right, borderType=cv2.BORDER_CONSTANT,
            value=0)
        if return_info:
            info = {
                'offset': offset,
                'scale': scale,
                'dsize': dsize,
                'embed_size': embed_size,
            }
            return new_img, info
        else:
            return new_img

    else:
        # Use np.round over python round, which has incompatible behavior
        old_dsize = (old_w, old_h)
        new_dsize = (int(np.round(new_w)), int(np.round(new_h)))
        new_scale = np.array(new_dsize) / np.array(old_dsize)
        interpolation = _coerce_interpolation(
            interpolation, scale=new_scale.min(),
            grow_default=grow_interpolation)
        new_img = _patched_resize(img, new_scale, new_dsize, interpolation=interpolation)
        if return_info:
            # import kwimage
            # transform = kwimage.Affine.affine(scale=scale)
            info = {
                'offset': 0,
                'scale': new_scale,
                # 'matrix': transform.matrix,
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
        img (ndarray): image data with float32 or uint8 precision

        src_space (str): input image colorspace. (e.g. BGR, GRAY)

        dst_space (str): desired output colorspace. (e.g. RGB, HSV, LAB)

        implicit (bool):
            if False, the user must correctly specify if the input/output
                colorspaces contain alpha channels.
            If True and the input image has an alpha channel, we modify
                src_space and dst_space to ensure they both end with "A".

        dst (ndarray[uint8_t, ndim=2], optional): inplace-output array.

    Returns:
        ndarray: img -  image data

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


def warp_affine(image, transform, dsize=None, antialias=False,
                interpolation='linear'):
    """
    Applies an affine transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image

        transform (ndarray | Affine): a coercable affine matrix

        dsize (Tuple[int, int] | None | str):
            width and height of the resulting image. If "auto", it is computed
            such that the positive coordinates of the warped image will fit in
            the new canvas. If None, then the image size will not change.

        antialias (bool, default=False):
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur.

        interpolation (str):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area.

    Example:
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> #image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale(0.05)
        >>> transform = Affine.scale(0.02)
        >>> warped1 = warp_affine(image, transform, dsize='auto', antialias=1, interpolation='nearest')
        >>> warped2 = warp_affine(image, transform, dsize='auto', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()

    Example:
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale((.1, 1.2))
        >>> warped1 = warp_affine(image, transform, dsize='auto', antialias=1)
        >>> warped2 = warp_affine(image, transform, dsize='auto', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()
    """
    from kwimage import im_cv2
    from kwimage.transform import Affine
    import kwimage
    transform = Affine.coerce(transform)
    flags = im_cv2._coerce_interpolation(interpolation)

    # TODO: expose these params
    # borderMode = cv2.BORDER_DEFAULT
    # borderMode = cv2.BORDER_CONSTANT
    borderMode = None
    borderValue = None

    """
    Variations that could change in the future:

        * In _gauss_params I'm not sure if we want to compute integer or
            fractional "number of downsamples".

        * The fudge factor bothers me, but seems necessary
    """

    if dsize is None:
        dsize = tuple(image.shape[0:2][::-1])
    elif dsize == 'auto':
        h, w = image.shape[0:2]
        boxes = kwimage.Boxes(np.array([[0, 0, w, h]]), 'xywh')
        poly = boxes.to_polygons()[0]
        warped_poly = poly.warp(transform.matrix)
        warped_box = warped_poly.to_boxes().to_ltrb().quantize()
        dsize = tuple(map(int, warped_box.data[0, 2:4]))

    if not antialias:
        M = np.asarray(transform)
        result = cv2.warpAffine(image, M[0:2],
                                dsize=dsize, flags=flags,
                                borderMode=borderMode,
                                borderValue=borderValue)
    else:
        # Decompose the affine matrix into its 6 core parameters
        params = transform.decompose()
        sx, sy = params['scale']

        if sx >= 1 and sy > 1:
            # No downsampling detected, no need to antialias
            M = np.asarray(transform)
            result = cv2.warpAffine(image, M[0:2], dsize=dsize, flags=flags,
                                    borderMode=borderMode,
                                    borderValue=borderValue)
        else:
            # At least one dimension is downsampled

            # Compute the transform with all scaling removed
            noscale_warp = Affine.affine(**ub.dict_diff(params, {'scale'}))

            # Execute part of the downscale with iterative pyramid downs
            downscaled, residual_sx, residual_sy = _prepare_downscale(
                image, sx, sy)

            # Compute the transform from the downsampled image to the destination
            rest_warp = noscale_warp @ Affine.scale((residual_sx, residual_sy))

            result = cv2.warpAffine(downscaled, rest_warp.matrix[0:2],
                                    dsize=dsize, flags=flags,
                                    borderMode=borderMode,
                                    borderValue=borderValue)

    return result


def _prepare_downscale(image, sx, sy):
    """
    Does a partial downscale with antialiasing and prepares for a final
    downsampling. Only downscales by factors of 2, any residual scaling to
    be done is returned.

    Example:
        >>> s = 523
        >>> image = np.random.rand(s, s)
        >>> sx = sy = 1 / 11
        >>> downsampled, rx, ry = _prepare_downscale(image, sx, sy)
    """
    max_scale = max(sx, sy)
    # The "fudge" factor limits the number of downsampled pyramid
    # operations. A bigger fudge factor means means that the final
    # gaussian kernel for the antialiasing operation will be bigger.
    # It essentials say that at most "fudge" downsampling ops will
    # be handled by the final blur rather than the pyramid downsample.
    # It seems to help with border effects at only a small runtime cost
    # I don't entirely understand why the border artifact is introduced
    # when this is enabled though

    # TODO: should we allow for this fudge factor?
    # TODO: what is the real name of this? num_down_prevent ?
    # skip_final_downs?
    fudge = 2
    # TODO: should final antialiasing be on?
    # Note, if fudge is non-zero it is important to do this.
    do_final_aa = 1
    # TODO: should fractional be True or False by default?
    # If fudge is 0 and fractional=0, then I think is the same as
    # do_final_aa=0.
    fractional = 1

    num_downs = max(int(np.log2(1 / max_scale)) - fudge, 0)
    pyr_scale = 1 / (2 ** num_downs)

    # Downsample iteratively with antialiasing
    downscaled = _pyrDownK(image, num_downs)

    residual_sx = sx / pyr_scale
    residual_sy = sy / pyr_scale

    # Do a final small blur to acount for the potential aliasing
    # in any remaining scaling operations.
    if do_final_aa:
        # Computed as the closest sigma to the [1, 4, 6, 4, 1] approx
        # used in cv2.pyrDown.
        """
        import cv2
        import numpy as np
        import scipy
        import ubelt as ub
        def sigma_error(sigma):
            sigma = np.asarray(sigma).ravel()[0]
            got = (cv2.getGaussianKernel(5, sigma) * 16).ravel()
            want = np.array([1, 4, 6, 4, 1])
            loss = ((got - want) ** 2).sum()
            return loss
        result = scipy.optimize.minimize(sigma_error, x0=1.0, method='Nelder-Mead')
        print('result = {}'.format(ub.repr2(result, nl=1)))
        # This gives a number like 1.06992187 which is not exactly what
        # we use.
        #
        # The actual optimal result was gotten with a search over
        # multiple optimization methods, Can be valided via:
        assert sigma_error(1.0699027846904146) <= sigma_error(result.x)
        """
        aa_sigma0 = 1.0699027846904146
        aa_k0 = 5
        k_x, sigma_x = _gauss_params(scale=residual_sx, k0=aa_k0,
                                     sigma0=aa_sigma0, fractional=fractional)
        k_y, sigma_y = _gauss_params(scale=residual_sy, k0=aa_k0,
                                     sigma0=aa_sigma0, fractional=fractional)

        # Note: when k=1, no blur occurs
        # blurBorderType = cv2.BORDER_REPLICATE
        # blurBorderType = cv2.BORDER_CONSTANT
        blurBorderType = cv2.BORDER_DEFAULT
        downscaled = cv2.GaussianBlur(
            downscaled, (k_x, k_y), sigma_x, sigma_y,
            borderType=blurBorderType
        )

    return downscaled, residual_sx, residual_sy


def _gauss_params(scale, k0=5, sigma0=1, fractional=True):
    """
    Compute a gaussian to mitigate aliasing for a requested downsample

    Args:
        scale: requested downsample factor
        k0 (int): kernel size for one downsample operation
        sigma0 (float): sigma for one downsample operation
        fractional (bool): controls if we compute params for integer downsample
        ops
    """
    num_downs = np.log2(1 / scale)
    if not fractional:
        num_downs = max(int(num_downs), 0)
    if num_downs <= 0:
        k = 1
        sigma = 0
    else:
        # The kernel size and sigma doubles for each 2x downsample
        sigma = sigma0 * (2 ** (num_downs - 1))
        k = int(np.ceil(k0 * (2 ** (num_downs - 1))))
        k = k + int(k % 2 == 0)
    return k, sigma


def _pyrDownK(a, k=1):
    """
    Downsamples by (2 ** k)x with antialiasing
    """
    if k == 0:
        a = a.copy()
    borderType = cv2.BORDER_DEFAULT
    # Note: pyrDown removes even pixels, which may introduce a bias towards the
    # bottom right of the image.
    for _ in range(k):
        a = cv2.pyrDown(a, borderType=borderType)
    return a
