"""
Numpy implementation of alpha blending based on information in [SO25182421]_
and [WikiAlphaBlend]_.

References:
    .. [SO25182421] http://stackoverflow.com/questions/25182421/overlay-numpy-alpha
    .. [WikiAlphaBlend] https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
"""

import numpy as np
from . import im_core


def overlay_alpha_layers(layers, keepalpha=True, dtype=np.float32):
    """
    Stacks a sequences of layers on top of one another. The first item is the
    topmost layer and the last item is the bottommost layer.

    Args:
        layers (Sequence[ndarray]): stack of images
        keepalpha (bool): if False, the alpha channel is removed after blending
        dtype (np.dtype): format for blending computation (defaults to float32)

    Returns:
        ndarray: raster: the blended images

    Example:
        >>> import kwimage
        >>> keys = ['astro', 'carl', 'stars']
        >>> layers = [kwimage.grab_test_image(k, dsize=(100, 100)) for k in keys]
        >>> layers = [kwimage.ensure_alpha_channel(g, alpha=.5) for g in layers]
        >>> stacked = kwimage.overlay_alpha_layers(layers)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(stacked)
        >>> kwplot.show_if_requested()
    """
    layer_iter = iter(layers)
    img1 = next(layer_iter)
    rgb1, alpha1 = _prep_rgb_alpha(img1, dtype=dtype)

    for img2 in layer_iter:
        rgb2, alpha2 = _prep_rgb_alpha(img2, dtype=dtype)
        rgb1, alpha1 = _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2)

    if keepalpha:
        raster = np.dstack([rgb1, alpha1[..., None]])
    else:
        raster = rgb1
    return raster


def overlay_alpha_images(img1, img2, keepalpha=True, dtype=np.float32,
                         impl='inplace'):
    """
    Places img1 on top of img2 respecting alpha channels.
    Works like the Photoshop layers with opacity.

    Args:
        img1 (ndarray): top image to overlay over img2
        img2 (ndarray): base image to superimpose on
        keepalpha (bool): if False, the alpha channel is removed after blending
        dtype (np.dtype): format for blending computation (defaults to float32)
        impl (str): code specifying the backend implementation

    Returns:
        ndarray: raster: the blended images

    TODO:
        - [ ] Make fast C++ version of this function

    Example:
        >>> import kwimage
        >>> img1 = kwimage.grab_test_image('astro', dsize=(100, 100))
        >>> img2 = kwimage.grab_test_image('carl', dsize=(100, 100))
        >>> img1 = kwimage.ensure_alpha_channel(img1, alpha=.5)
        >>> img3 = kwimage.overlay_alpha_images(img1, img2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img3)
        >>> kwplot.show_if_requested()

    Ignore:
        import numpy as np
        import kwimage
        poly = kwimage.Polygon.random().scale((10, 10))

        img2 = np.zeros((10, 10, 4))

        img1 = np.zeros((10, 10))
        indicator = poly.fill(img1)

        to_overlay = np.zeros((10, 10) + (4,), dtype=np.float32)
        to_overlay = kwimage.Mask(indicator, format='c_mask').draw_on(to_overlay, color='lime')
        to_overlay = kwimage.ensure_alpha_channel(to_overlay)
        to_overlay[..., 3] = (indicator > 0).astype(np.float32) * 0.5

        raster = kwimage.overlay_alpha_images(to_overlay, img2)

        # xdoctest: +REQUIRES(--show)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(raster)
        kwplot.show_if_requested()
    """
    rgb1, alpha1 = _prep_rgb_alpha(img1, dtype=dtype)
    rgb2, alpha2 = _prep_rgb_alpha(img2, dtype=dtype)

    # Perform the core alpha blending algorithm
    if impl == 'simple':
        rgb3, alpha3 = _alpha_blend_simple(rgb1, alpha1, rgb2, alpha2)
    elif impl == 'inplace':
        rgb3, alpha3 = _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2)
    elif impl == 'numexpr1':
        rgb3, alpha3 = _alpha_blend_numexpr1(rgb1, alpha1, rgb2, alpha2)
    elif impl == 'numexpr2':
        rgb3, alpha3 = _alpha_blend_numexpr2(rgb1, alpha1, rgb2, alpha2)
    else:
        raise ValueError('unknown impl={}'.format(impl))

    if keepalpha:
        raster = np.dstack([rgb3, alpha3[..., None]])
        # Note: if we want to output a 255 img we could do something like this
        # out = np.zeros_like(img1)
        # out[..., :3] = rgb3
        # out[..., 3] = alpha3
    else:
        raster = rgb3
    return raster


def _prep_rgb_alpha(img, dtype=np.float32):
    img = im_core.ensure_float01(img, dtype=dtype, copy=False)
    img = im_core.atleast_3channels(img, copy=False)
    c = im_core.num_channels(img)

    if c == 4:
        # rgb = np.ascontiguousarray(img[..., 0:3])
        # alpha = np.ascontiguousarray(img[..., 3])
        rgb = img[..., 0:3]
        alpha = img[..., 3]
    else:
        rgb = img
        alpha = np.ones_like(img[..., 0])
    return rgb, alpha


def _alpha_blend_simple(rgb1, alpha1, rgb2, alpha2):
    """
    Core alpha blending algorithm

    SeeAlso:
        _alpha_blend_inplace - alternative implementation
    """
    c_alpha1 = (1.0 - alpha1)
    alpha3 = alpha1 + alpha2 * c_alpha1

    numer1 = (rgb1 * alpha1[..., None])
    numer2 = (rgb2 * (alpha2 * c_alpha1)[..., None])
    with np.errstate(invalid='ignore'):
        rgb3 = (numer1 + numer2) / alpha3[..., None]
    rgb3[alpha3 == 0] = 0
    return rgb3, alpha3


def _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2):
    """
    Uglier but faster(? maybe not) version of the core alpha blending algorithm
    using preallocation and in-place computation where possible.

    SeeAlso:
        _alpha_blend_simple - alternative implementation

    Example:
        >>> rng = np.random.RandomState(0)
        >>> rgb1, rgb2 = rng.rand(10, 10, 3), rng.rand(10, 10, 3)
        >>> alpha1, alpha2 = rng.rand(10, 10), rng.rand(10, 10)
        >>> f1, f2 = _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2)
        >>> s1, s2 = _alpha_blend_simple(rgb1, alpha1, rgb2, alpha2)
        >>> assert np.all(f1 == s1) and np.all(f2 == s2)
        >>> alpha1, alpha2 = np.zeros((10, 10)), np.zeros((10, 10))
        >>> f1, f2 = _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2)
        >>> s1, s2 = _alpha_blend_simple(rgb1, alpha1, rgb2, alpha2)
        >>> assert np.all(f1 == s1) and np.all(f2 == s2)
    """
    rgb3 = np.empty_like(rgb1)
    temp_rgb = np.empty_like(rgb1)
    alpha3 = np.empty_like(alpha1)
    temp_alpha = np.empty_like(alpha1)

    # hold (1 - alpha1)
    np.subtract(1, alpha1, out=temp_alpha)

    # alpha3
    np.copyto(dst=alpha3, src=temp_alpha)
    np.multiply(alpha2, alpha3, out=alpha3)
    np.add(alpha1, alpha3, out=alpha3)

    # numer1
    np.multiply(rgb1, alpha1[..., None], out=rgb3)

    # numer2
    np.multiply(alpha2, temp_alpha, out=temp_alpha)
    np.multiply(rgb2, temp_alpha[..., None], out=temp_rgb)

    # (numer1 + numer2)
    np.add(rgb3, temp_rgb, out=rgb3)

    # removing errstate is actually a significant speedup
    with np.errstate(invalid='ignore'):
        np.divide(rgb3, alpha3[..., None], out=rgb3)
    if not np.all(alpha3):
        rgb3[alpha3 == 0] = 0
    return rgb3, alpha3


def _alpha_blend_numexpr1(rgb1, alpha1, rgb2, alpha2):
    """ Alternative. Not well optimized """
    import numexpr
    alpha1_ = alpha1[..., None]  # NOQA
    alpha2_ = alpha2[..., None]  # NOQA
    alpha3 = numexpr.evaluate('alpha1 + alpha2 * (1.0 - alpha1)')
    alpha3_ = alpha3[..., None]  # NOQA
    rgb3 = numexpr.evaluate('((rgb1 * alpha1_) + (rgb2 * alpha2_ * (1.0 - alpha1_))) / alpha3_')
    rgb3[alpha3 == 0] = 0


def _alpha_blend_numexpr2(rgb1, alpha1, rgb2, alpha2):
    """ Alternative. Not well optimized """
    import numexpr
    c_alpha1 = numexpr.evaluate('1.0 - alpha1')
    alpha3 = numexpr.evaluate('alpha1 + alpha2 * c_alpha1')

    c_alpha1_ = c_alpha1[..., None]  # NOQA
    alpha1_ = alpha1[..., None]  # NOQA
    alpha2_ = alpha2[..., None]  # NOQA
    alpha3_ = alpha3[..., None]  # NOQA

    numer1 = numexpr.evaluate('rgb1 * alpha1_')  # NOQA
    numer2 = numexpr.evaluate('rgb2 * (alpha2_ * c_alpha1_)')  # NOQA
    with np.errstate(invalid='ignore'):
        rgb3 = numexpr.evaluate('(numer1 + numer2) / alpha3_')
    rgb3[alpha3 == 0] = 0
    return rgb3, alpha3


def ensure_alpha_channel(img, alpha=1.0, dtype=np.float32, copy=False):
    """
    Returns the input image with 4 channels.

    Args:
        img (ndarray):
            an image with shape [H, W], [H, W, 1], [H, W, 3], or [H, W, 4].

        alpha (float | ndarray):
            default scalar value for missing alpha channel, or
            an ndarray with the same height / width to use explicitly.

        dtype (type):
            The final output dtype. Should be numpy.float32 or numpy.float64.

        copy (bool):
            always copy if True, else copy if needed.

    Returns:
        ndarray: an image with specified dtype with shape [H, W, 4].

    Raises:
        ValueError - if the input image does not have 1, 3, or 4 input channels
            or if the image cannot be converted into a float01 representation

    Example:
        >>> # Demo with a scalar default alpha value
        >>> import kwimage
        >>> data0 = np.zeros((5, 5))
        >>> data1 = np.zeros((5, 5, 1))
        >>> data2 = np.zeros((5, 5, 3))
        >>> data3 = np.zeros((5, 5, 4))
        >>> ensured0 = kwimage.ensure_alpha_channel(data0, alpha=0.5)
        >>> ensured1 = kwimage.ensure_alpha_channel(data1, alpha=0.5)
        >>> ensured2 = kwimage.ensure_alpha_channel(data2, alpha=0.5)
        >>> ensured3 = kwimage.ensure_alpha_channel(data3, alpha=0.5)
        >>> assert np.all(ensured0[..., 3] == 0.5), 'should have been populated'
        >>> assert np.all(ensured1[..., 3] == 0.5), 'should have been populated'
        >>> assert np.all(ensured2[..., 3] == 0.5), 'should have been populated'
        >>> assert np.all(ensured3[..., 3] == 0.0), 'last image already had alpha'

    Example:
        >>> import kwimage
        >>> # Demo with a explicit alpha channel
        >>> alpha = np.random.rand(5, 5)
        >>> data0 = np.zeros((5, 5))
        >>> data1 = np.zeros((5, 5, 1))
        >>> data2 = np.zeros((5, 5, 3))
        >>> data3 = np.zeros((5, 5, 4))
        >>> ensured0 = kwimage.ensure_alpha_channel(data0, alpha=alpha)
        >>> ensured1 = kwimage.ensure_alpha_channel(data1, alpha=alpha)
        >>> ensured2 = kwimage.ensure_alpha_channel(data2, alpha=alpha)
        >>> ensured3 = kwimage.ensure_alpha_channel(data3, alpha=alpha)
        >>> assert np.all(ensured0[..., 3] == alpha), 'should have been populated'
        >>> assert np.all(ensured1[..., 3] == alpha), 'should have been populated'
        >>> assert np.all(ensured2[..., 3] == alpha), 'should have been populated'
        >>> assert np.all(ensured3[..., 3] == 0.0), 'last image already had alpha'
    """
    img = im_core.ensure_float01(img, dtype=dtype, copy=copy)
    c = im_core.num_channels(img)
    if c == 4:
        return img
    else:
        if isinstance(alpha, np.ndarray):
            alpha_channel = alpha
        else:
            alpha_channel = np.full(img.shape[0:2], fill_value=alpha, dtype=img.dtype)
        if c == 3:
            return np.dstack([img, alpha_channel])
        elif c == 1:
            return np.dstack([img, img, img, alpha_channel])
        else:
            raise ValueError(
                'Cannot ensure alpha. Input image has c={} channels'.format(c))
