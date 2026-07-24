import numpy as np


def benchmark_alphablend_impls():
    """
    Ignore:
        from kwimage.imutil.im_alphablend import *
    """
    import ubelt as ub
    import xdev

    import kwimage
    from kwimage.im_alphablend import (
        _alpha_blend_inplace,
        _alpha_blend_numexpr1,
        _alpha_blend_numexpr2,
        _alpha_blend_simple,
        _prep_rgb_alpha,
        overlay_alpha_images,
    )

    H = W = 32
    rng = np.random.RandomState(0)

    rgb1, rgb2 = rng.rand(H, W, 3), rng.rand(H, W, 3)
    alpha1, alpha2 = rng.rand(H, W), rng.rand(H, W)

    dtype = np.float32
    # dtype = np.float64
    rgb1 = rgb1.astype(dtype)
    rgb2 = rgb2.astype(dtype)
    alpha1 = alpha1.astype(dtype)
    alpha2 = alpha2.astype(dtype)

    # If num is set too low it may seem like simple beats inplace, but that is
    # actually not the case. inplace is slightly faster as expected.
    ti = ub.Timerit(num=2000, bestof=100, unit='us', verbose=1)
    ti.reset(label='simple').call(
        lambda: _alpha_blend_simple(rgb1, alpha1, rgb2, alpha2)
    )
    ti.reset(label='inplace').call(
        lambda: _alpha_blend_inplace(rgb1, alpha1, rgb2, alpha2)
    )
    ti.reset(label='numexpr1').call(
        lambda: _alpha_blend_numexpr1(rgb1, alpha1, rgb2, alpha2)
    )
    ti.reset(label='numexpr2').call(
        lambda: _alpha_blend_numexpr2(rgb1, alpha1, rgb2, alpha2)
    )

    # It looks like the simple algorithm is winning ATM
    ub.Timerit(label='inplace', unit='us').call(
        lambda: overlay_alpha_images(rgb1, rgb2, impl='inplace')
    )
    ub.Timerit(label='simple', unit='us').call(
        lambda: overlay_alpha_images(rgb1, rgb2, impl='simple')
    )

    _ = xdev.profile_now(overlay_alpha_images)(rgb1, rgb2, impl='simple')
    _ = xdev.profile_now(overlay_alpha_images)(rgb1, rgb2, impl='inplace')

    _ = xdev.profile_now(kwimage.ensure_float01)(rgb1)
    _ = xdev.profile_now(_prep_rgb_alpha)(rgb1)
    _ = xdev.profile_now(_prep_rgb_alpha)(rgb2)

    _ = xdev.profile_now(_alpha_blend_simple)(rgb1, alpha1, rgb2, alpha2)
    _ = xdev.profile_now(_alpha_blend_inplace)(rgb1, alpha1, rgb2, alpha2)
    _ = xdev.profile_now(_alpha_blend_numexpr1)(rgb1, alpha1, rgb2, alpha2)
    _  # NOQA
