# -*- coding: utf-8 -*-
"""
Not sure how to best classify these functions
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
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
        if img.dtype.kind != 'u' or img.dtype.itemsize != 1:
            # Only check min/max if the image is not a uint8
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
            raise ValueError('Cannot handle ndims={} with c={}'.format(ndims, c))
    else:
        raise ValueError('Cannot handle arr.shape={}'.format(arr.shape))
    return res


def normalize(arr, mode='linear', alpha=None, beta=None, out=None):
    """
    Rebalance pixel intensities via contrast stretching.

    By default linearly stretches pixel intensities to minimum and maximum
    values.

    Notes:
        This function has been MOVED to kwarray

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

    # TODO:
    # - [ ] Parametarize new_min / new_max values
    #     - [ ] infer from datatype
    #     - [ ] explicitly given
    new_min = 0.0
    if arr.dtype.kind in ('i', 'u'):
        # Need a floating point workspace
        float_out = out.astype(np.float32)
        new_max = float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind == 'f':
        float_out = out
        new_max = 1.0
    else:
        raise NotImplementedError

    # TODO:
    # - [ ] Parametarize old_min / old_max strategies
    #     - [ ] explicitly given min and max
    #     - [ ] raw-naive min and max inference
    #     - [ ] outlier-aware min and max inference
    old_min = float_out.min()
    old_max = float_out.max()

    old_span = old_max - old_min
    new_span = new_max - new_min

    if mode == 'linear':
        # linear case
        # out = (arr - old_min) * (new_span / old_span) + new_min
        factor = 1.0 if old_span == 0 else (new_span / old_span)
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


def padded_slice(data, in_slice, pad=None, padkw=None, return_info=False):
    """
    Allows slices with out-of-bound coordinates.  Any out of bounds coordinate
    will be sampled via padding.

    Note:
        Negative slices have a different meaning here then they usually do.
        Normally, they indicate a wrap-around or a reversed stride, but here
        they index into out-of-bounds space (which depends on the pad mode).
        For example a slice of -2:1 literally samples two pixels to the left of
        the data and one pixel from the data, so you get two padded values and
        one data value.

    Args:
        data (Sliceable[T]): data to slice into. Any channels must be the last dimension.
        in_slice (slice | Tuple[slice, ...]): slice for each dimensions
        ndim (int): number of spatial dimensions
        pad (List[int|Tuple]): additional padding of the slice
        padkw (Dict): if unspecified defaults to ``{'mode': 'constant'}``
        return_info (bool, default=False): if True, return extra information
            about the transform.

    Returns:

        Sliceable:
            data_sliced: subregion of the input data (possibly with padding,
                depending on if the original slice went out of bounds)


        Tuple[Sliceable, Dict] :
            data_sliced : as above

            transform : information on how to return to the original coordinates

                Currently a dict containing:
                    st_dims: a list indicating the low and high space-time
                        coordinate values of the returned data slice.

    Example:
        >>> data = np.arange(5)
        >>> in_slice = [slice(-2, 7)]

        >>> data_sliced = padded_slice(data, in_slice)
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 1, 2, 3, 4, 0, 0])

        >>> data_sliced = padded_slice(data, in_slice, pad=(3, 3))
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])

        >>> data_sliced = padded_slice(data, slice(3, 4), pad=[(1, 0)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([2, 3])

    """
    if isinstance(in_slice, slice):
        in_slice = [in_slice]

    ndim = len(in_slice)

    data_dims = data.shape[:ndim]

    data_slice, extra_padding = _pad_slice(in_slice, data_dims, pad=pad)

    in_slice_clipped = tuple(slice(*d) for d in data_slice)
    # Get the parts of the image that are in bounds
    data_clipped = data[in_slice_clipped]

    # Add any padding that is needed to behave like negative dims exist
    if sum(map(sum, extra_padding)) == 0:
        # The slice was completely in bounds
        data_sliced = data_clipped
    else:
        if padkw is None:
            padkw = {
                'mode': 'constant',
            }
        if len(data.shape) != len(extra_padding):
            extra_padding = extra_padding + [(0, 0)]
        data_sliced = np.pad(data_clipped, extra_padding, **padkw)

    st_dims = data_slice[0:ndim]
    pad_dims = extra_padding[0:ndim]

    st_dims = [(s - pad_[0], t + pad_[1])
               for (s, t), pad_ in zip(st_dims, pad_dims)]

    # TODO: return a better transform back to the original space
    if return_info:
        transform = {
            'st_dims': st_dims,
            'st_offset': [d[0] for d in st_dims]
        }
        return data_sliced, transform
    else:
        return data_sliced


def _pad_slice(in_slice, data_dims, pad=None):
    """
    Given a slices for each dimension, image dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    Args:
        in_slice (Tuple[slice]):
            a tuple of slices for to apply to data data dimension.
        data_dims (Tuple[int]):
            n-dimension data sizes (e.g. 2d height, width)
        pad (tuple): (List[int|Tuple]):
            extra pad applied to (left and right) / (both) sides of each slice
            dim

    Returns:
        Tuple:
            data_slice - low and high values of a fancy slice corresponding to
                the image with shape `data_dims`. This slice may not correspond
                to the full window size if the requested bounding box goes out
                of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where slice is inside the data dims on left edge
        >>> in_slice = (slice(0, 10), slice(0, 10))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = _pad_slice(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(0, 20), (0, 15)]
        extra_padding = [(10, 0), (5, 0)]

    Example:
        >>> # Case where slice is bigger than the image
        >>> in_slice = (slice(-10, 400), slice(-10, 400))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = _pad_slice(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(0, 300), (0, 300)]
        extra_padding = [(20, 110), (15, 105)]

    Example:
        >>> # Case where slice is inside than the image
        >>> in_slice = (slice(10, 40), slice(10, 40))
        >>> data_dims  = [300, 300]
        >>> pad        = None
        >>> a, b = _pad_slice(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(10, 40), (10, 40)]
        extra_padding = [(0, 0), (0, 0)]
    """
    low_dims = [sl.start for sl in in_slice]
    high_dims = [sl.stop for sl in in_slice]

    # Determine the real part of the image that can be sliced out
    data_slice = []
    extra_padding = []
    if pad is None:
        pad = 0
    if isinstance(pad, int):
        pad = [pad] * len(data_dims)
    # Normalize to left/right pad value for each dim
    pad_slice = [p if ub.iterable(p) else [p, p] for p in pad]

    # Determine the real part of the image that can be sliced out
    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad_slice):
        if d_low > d_high:
            raise ValueError('d_low > d_high: {} > {}'.format(d_low, d_high))
        # Determine where the bounds would be if the image size was inf
        raw_low = d_low - d_pad[0]
        raw_high = d_high + d_pad[1]
        # Clip the slice positions to the real part of the image
        sl_low = min(D_img, max(0, raw_low))
        sl_high = min(D_img, max(0, raw_high))
        data_slice.append((sl_low, sl_high))

        # Add extra padding when the window extends past the real part
        low_diff = sl_low - raw_low
        high_diff = raw_high - sl_high

        # Hand the case where both raw coordinates are out of bounds
        extra_low = max(0, low_diff + min(0, high_diff))
        extra_high = max(0, high_diff + min(0, low_diff))
        extra = (extra_low, extra_high)
        extra_padding.append(extra)
    return data_slice, extra_padding
