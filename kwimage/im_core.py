# -*- coding: utf-8 -*-
"""
Not sure how to best classify these functions
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def num_channels(img):
    """
    Returns the number of color channels in an image

    Args:
        img (ndarray): an image with 2 or 3 dimensions.

    Returns:
        int : the number of color channels (1, 3, or 4)

    Example:
        >>> H = W = 3
        >>> assert num_channels(np.empty((W, H))) == 1
        >>> assert num_channels(np.empty((W, H, 1))) == 1
        >>> assert num_channels(np.empty((W, H, 3))) == 3
        >>> assert num_channels(np.empty((W, H, 4))) == 4
        >>> # xdoctest: +REQUIRES(module:pytest)
        >>> import pytest
        >>> with pytest.raises(ValueError):
        ...     num_channels(np.empty((W, H, 2)))
    """
    ndims = img.ndim
    if ndims == 2:
        n_channels = 1
    elif ndims == 3 and img.shape[2] == 3:
        n_channels = 3
    elif ndims == 3 and img.shape[2] == 4:
        n_channels = 4
    elif ndims == 3 and img.shape[2] == 1:
        n_channels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return n_channels


def ensure_float01(img, dtype=np.float32, copy=True):
    """
    Ensure that an image is encoded using a float32 properly

    Args:
        img (ndarray): an image in uint255 or float01 format.
            Other formats will raise errors.
        dtype (type, default=np.float32): a numpy floating type
        copy (bool, default=False): always copy if True, else copy if needed.

    Returns:
        ndarray: an array of floats in the range 0-1

    Raises:
        ValueError : if the image type is integer and not in [0-255]

    Example:
        >>> ensure_float01(np.array([[0, .5, 1.0]]))
        array([[0. , 0.5, 1. ]], dtype=float32)
        >>> ensure_float01(np.array([[0, 1, 200]]))
        array([[0..., 0.0039..., 0.784...]], dtype=float32)
    """
    if img.dtype.kind in ('i', 'u'):
        if img.min() < 0 or img.max() > 255:
            import kwarray
            raise ValueError(
                'The image type is int, but its values are not '
                'between 0 and 255. Image stats are {}'.format(
                    kwarray.stats_dict(img)))
        img_ = img.astype(dtype, copy=copy) / 255.0
    else:
        img_ = img.astype(dtype, copy=copy)
    return img_


def ensure_uint255(img, copy=True):
    """
    Ensure that an image is encoded using a uint8 properly. Either

    Args:
        img (ndarray): an image in uint255 or float01 format.
            Other formats will raise errors.
        copy (bool, default=False): always copy if True, else copy if needed.

    Returns:
        ndarray: an array of bytes in the range 0-255

    Raises:
        ValueError : if the image type is float and not in [0-1]
        ValueError : if the image type is integer and not in [0-255]

    Example:
        >>> ensure_uint255(np.array([[0, .5, 1.0]]))
        array([[  0, 127, 255]], dtype=uint8)
        >>> ensure_uint255(np.array([[0, 1, 200]]))
        array([[  0,   1, 200]], dtype=uint8)
    """
    if img.dtype.kind == 'u':
        img_ = img.astype(np.uint8, copy=copy)
    elif img.dtype.kind == 'i':
        if img.min() < 0 or img.max() > 255:
            import kwarray
            raise AssertionError(
                'The image type is signed int, but its values are not '
                'between 0 and 255. Image stats are {}'.format(
                    kwarray.stats_dict(img)))
        img_ = img.astype(np.uint8, copy=copy)
    else:
        # If the image is a float check that it is between 0 and 1
        # Use a +- epsilon of 1e-3 to account for floating errors
        eps = 1e-3
        if (img.min() < (0 - eps) or img.max() > (1 + eps)):
            import kwarray
            raise ValueError(
                'The image type is float, but its values are not '
                'between 0 and 1. Image stats are {}'.format(
                    kwarray.stats_dict(img)))
        img_ = (img.clip(0, 1) * 255).astype(np.uint8, copy=copy)
    return img_


# def _cast_type(*arrs):
#     # SeeAlso: np.common_type
#     for a in arrs:
#         if a.dtype.kind in ['f']:
#             pass


def make_channels_comparable(img1, img2, atleast3d=False):
    """
    Broadcasts image arrays so they can have elementwise operations applied

    Args:
        img1 (ndarray): first image
        img2 (ndarray): second image
        atleast3d (bool, default=False): if true we ensure that the channel
            dimension exists (only relevant for 1-channel images)

    Example:
        >>> import itertools as it
        >>> wh_basis = [(5, 5), (3, 5), (5, 3), (1, 1), (1, 3), (3, 1)]
        >>> for w, h in wh_basis:
        >>>     shape_basis = [(w, h), (w, h, 1), (w, h, 3)]
        >>>     # Test all permutations of shap inputs
        >>>     for shape1, shape2 in it.product(shape_basis, shape_basis):
        >>>         print('*    input shapes: %r, %r' % (shape1, shape2))
        >>>         img1 = np.empty(shape1)
        >>>         img2 = np.empty(shape2)
        >>>         img1, img2 = make_channels_comparable(img1, img2)
        >>>         print('... output shapes: %r, %r' % (img1.shape, img2.shape))
        >>>         elem = (img1 + img2)
        >>>         print('... elem(+) shape: %r' % (elem.shape,))
        >>>         assert elem.size == img1.size, 'outputs should have same size'
        >>>         assert img1.size == img2.size, 'new imgs should have same size'
        >>>         print('--------')
    """
    # if img1.shape[0:2] != img2.shape[0:2]:
    #     raise ValueError(
    #         'Spatial sizes of {!r} and {!r} are not compatible'.format(
    #             img1.shape, img2.shape))
    if img1.shape != img2.shape:
        c1 = num_channels(img1)
        c2 = num_channels(img2)
        if len(img1.shape) == 2 and len(img2.shape) == 2:
            # Both images are 2d grayscale
            if atleast3d:
                # add the third dim with 1 channel
                img1 = img1[:, None]
                img2 = img2[:, None]
        elif len(img1.shape) == 3 and len(img2.shape) == 2:
            # Image 2 is grayscale
            if c1 == 3:
                img2 = np.tile(img2[..., None], 3)
            else:
                img2 = img2[..., None]
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Image 1 is grayscale
            if c2 == 3:
                img1 = np.tile(img1[..., None], 3)
            else:
                img1 = img1[..., None]
        elif len(img1.shape) == 3 and len(img2.shape) == 3:
            # Both images have 3 dims.
            # Check if either have color, then check for alpha
            if c1 == 1 and c2 == 1:
                # raise AssertionError('UNREACHABLE: Both are 3-grayscale')
                pass
            elif c1 == 4 and c2 == 4:
                # raise AssertionError('UNREACHABLE: Both are 3-alpha')
                pass
            elif c1 == 3 and c2 == 3:
                # raise AssertionError('UNREACHABLE: Both are 3-color')
                pass
            elif c1 == 1 and c2 == 3:
                img1 = np.tile(img1, 3)
            elif c1 == 3 and c2 == 1:
                img2 = np.tile(img2, 3)
            elif c1 == 1 and c2  == 4:
                img1 = np.dstack((np.tile(img1, 3), _alpha_fill_for(img1)))
            elif c1 == 4 and c2  == 1:
                img2 = np.dstack((np.tile(img2, 3), _alpha_fill_for(img2)))
            elif c1 == 3 and c2  == 4:
                img1 = np.dstack((img1, _alpha_fill_for(img1)))
            elif c1 == 4 and c2  == 3:
                img2 = np.dstack((img2, _alpha_fill_for(img2)))
            else:
                raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
        else:
            raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
    return img1, img2


def _alpha_fill_for(img):
    """ helper for make_channels_comparable """
    fill_value = (255 if img.dtype.kind in ('i', 'u') else 1)
    alpha_chan = np.full(img.shape[0:2], dtype=img.dtype,
                         fill_value=fill_value)
    return alpha_chan


def atleast_3channels(arr, copy=True):
    r"""
    Ensures that there are 3 channels in the image

    Args:
        arr (ndarray[N, M, ...]): the image
        copy (bool): Always copies if True, if False, then copies only when the
            size of the array must change.

    Returns:
        ndarray: with shape (N, M, C), where C in {3, 4}

    Doctest:
        >>> assert atleast_3channels(np.zeros((10, 10))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 1))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 3))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 4))).shape[-1] == 4
    """
    ndims = len(arr.shape)
    if ndims == 2:
        res = np.tile(arr[:, :, None], 3)
        return res
    elif ndims == 3:
        h, w, c = arr.shape
        if c == 1:
            res = np.tile(arr, 3)
        elif c in [3, 4]:
            res = arr.copy() if copy else arr
        else:
            raise ValueError('Cannot handle ndims={}'.format(ndims))
    else:
        raise ValueError('Cannot handle arr.shape={}'.format(arr.shape))
    return res


def normalize(arr, mode='linear', alpha=None, beta=None, out=None):
    """
    Rebalance pixel intensities via contrast stretching.

    By default linearly stretches pixel intensities to minimum and maximum
    values.

    Args:
        arr (ndarray): array to normalize, usually an image

        out (ndarray | None): output array. Note, that we will create an
            internal floating point copy for integer computations.

        mode (str): either linear or sigmoid.

        alpha (float): Only used if mode=sigmoid.  Division factor
            (pre-sigmoid). If unspecified computed as:
            ``max(abs(old_min - beta), abs(old_max - beta)) / 6.212606``.
            Note this parameter is sensitive to if the input is a float or
            uint8 image.

        beta (float): subtractive factor (pre-sigmoid). This should be the
            intensity of the most interesting bits of the image, i.e. bring
            them to the center (0) of the distribution.
            Defaults to ``(max - min) / 2``.  Note this parameter is sensitive
            to if the input is a float or uint8 image.

    References:
        https://en.wikipedia.org/wiki/Normalization_(image_processing)

    Example:
        >>> raw_f = np.random.rand(8, 8)
        >>> norm_f = normalize(raw_f)

        >>> raw_f = np.random.rand(8, 8) * 100
        >>> norm_f = normalize(raw_f)
        >>> assert np.isclose(norm_f.min(), 0)
        >>> assert np.isclose(norm_f.max(), 1)

        >>> raw_u = (np.random.rand(8, 8) * 255).astype(np.uint8)
        >>> norm_u = normalize(raw_u)

    Example:
        >>> from kwimage.im_core import *  # NOQA
        >>> import kwimage
        >>> arr = kwimage.grab_test_image('lowcontrast')
        >>> arr = kwimage.ensure_float01(arr)
        >>> norms = {}
        >>> norms['arr'] = arr.copy()
        >>> norms['linear'] = normalize(arr, mode='linear')
        >>> norms['sigmoid'] = normalize(arr, mode='sigmoid')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(norms))
        >>> for key, img in norms.items():
        >>>     kwplot.imshow(img, pnum=pnum_(), title=key)

    Benchmark:
        # Our method is faster than standard in-line implementations.

        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2, unit='ms')
        arr = kwimage.grab_test_image('lowcontrast', dsize=(512, 512))

        print('--- uint8 ---')
        arr = ensure_float01(arr)
        out = arr.copy()
        for timer in ti.reset('naive1-float'):
            with timer:
                (arr - arr.min()) / (arr.max() - arr.min())

        import timerit
        for timer in ti.reset('simple-float'):
            with timer:
                max_ = arr.max()
                min_ = arr.min()
                result = (arr - min_) / (max_ - min_)

        for timer in ti.reset('normalize-float'):
            with timer:
                normalize(arr)

        for timer in ti.reset('normalize-float-inplace'):
            with timer:
                normalize(arr, out=out)

        print('--- float ---')
        arr = ensure_uint255(arr)
        out = arr.copy()
        for timer in ti.reset('naive1-uint8'):
            with timer:
                (arr - arr.min()) / (arr.max() - arr.min())

        import timerit
        for timer in ti.reset('simple-uint8'):
            with timer:
                max_ = arr.max()
                min_ = arr.min()
                result = (arr - min_) / (max_ - min_)

        for timer in ti.reset('normalize-uint8'):
            with timer:
                normalize(arr)

        for timer in ti.reset('normalize-uint8-inplace'):
            with timer:
                normalize(arr, out=out)

    Ignore:
        globals().update(xdev.get_func_kwargs(normalize))
    """
    if out is None:
        out = arr.copy()

    if arr.dtype.kind in ('i', 'u'):
        # Need a floating point workspace
        float_out = out.astype(np.float32)
        new_max = float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind == 'f':
        float_out = out
        new_max = 1.0
    else:
        raise NotImplementedError

    new_min = 0.0

    old_min = float_out.min()
    old_max = float_out.max()
    old_span = old_max - old_min
    new_span = new_max - new_min

    if mode == 'linear':
        # linear case
        # out = (arr - old_min) * (new_span / old_span) + new_min
        if old_span == 0:
            factor = 1.0
        else:
            factor = (new_span / old_span)
        if old_min != 0:
            float_out -= old_min
    elif mode == 'sigmoid':
        # nonlinear case
        # out = new_span * sigmoid((arr - beta) / alpha) + new_min
        from scipy.special import expit as sigmoid
        if beta is None:
            # should center the desired distribution to visualize on zero
            beta = old_max - old_min

        if alpha is None:
            # division factor
            # from scipy.special import logit
            # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
            # This chooses alpha such the original min/max value will be pushed
            # towards -1 / +1.
            alpha = max(abs(old_min - beta), abs(old_max - beta)) / 6.212606
        energy = float_out
        energy -= beta
        energy /= alpha
        # Ideally the data of interest is roughly in the range (-6, +6)
        float_out = sigmoid(energy, out=float_out)
        factor = new_span
    else:
        raise KeyError(mode)

    # Stretch / shift to the desired output range
    if factor != 1:
        float_out *= factor

    if new_min != 0:
        float_out += new_min

    if float_out is not out:
        out[:] = float_out.astype(out.dtype)
    return out
