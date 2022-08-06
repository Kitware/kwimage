"""
Not sure how to best classify these functions
"""
import ubelt as ub
import numpy as np
import math


def num_channels(img):
    """
    Returns the number of color channels in an image.

    Assumes images are 2D and the the channels are the trailing dimension.
    Returns 1 in the case with no trailing channel dimension, otherwise simply
    returns ``img.shape[2]``.

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
        >>> assert num_channels(np.empty((W, H, 2))) == 2
    """
    ndims = img.ndim
    if ndims == 2:
        n_channels = 1
    elif ndims == 3:
        # Previously threw an error when n_channels was not 1, 3, or 4
        n_channels = img.shape[2]
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

        dtype (type): a numpy floating type defaults to np.float32

        copy (bool):
            Always copy if True, else copy if needed. Defaults to True.

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

        copy (bool):
            always copy if True, else copy if needed. Defaults to True.

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
        atleast3d (bool):
            if true we ensure that the channel dimension exists (only relevant
            for 1-channel images). Defaults to False.

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
        arr (ndarray): an image with 2 or 3 dims.

        copy (bool):
            Always copies if True, if False, then copies only when the
            size of the array must change. Defaults to True.

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


def padded_slice(data, in_slice, pad=None, padkw=None, return_info=False):
    """
    Allows slices with out-of-bound coordinates.  Any out of bounds coordinate
    will be sampled via padding.

    DEPRECATED FOR THE VERSION IN KWARRAY (slices are more array-ish than
    image-ish)

    Note:
        Negative slices have a different meaning here then they usually do.
        Normally, they indicate a wrap-around or a reversed stride, but here
        they index into out-of-bounds space (which depends on the pad mode).
        For example a slice of -2:1 literally samples two pixels to the left of
        the data and one pixel from the data, so you get two padded values and
        one data value.

    Args:
        data (Sliceable):
            data to slice into. Any channels must be the last dimension.

        in_slice (slice | Tuple[slice, ...]):
            slice for each dimensions

        ndim (int): number of spatial dimensions

        pad (List[int|Tuple]): additional padding of the slice

        padkw (Dict): if unspecified defaults to ``{'mode': 'constant'}``

        return_info (bool): if True, return extra information
            about the transform. Defaults to False.

    SeeAlso:
        _padded_slice_embed - finds the embedded slice and padding
        _padded_slice_apply - applies padding to sliced data

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

                The structure of this dictionary mach change in the future

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
    import warnings
    warnings.warn('Deprecated use kwarray.padded_slice instead',
                  DeprecationWarning)

    if isinstance(in_slice, slice):
        in_slice = [in_slice]

    ndim = len(in_slice)
    data_dims = data.shape[:ndim]

    # separate requested slice into an in-bounds part and a padding part
    data_slice, extra_padding = _padded_slice_embed(in_slice, data_dims,
                                                    pad=pad)

    # Get the parts of the image that are in-bounds
    data_clipped = data[data_slice]

    # Apply the padding part
    data_sliced, transform = _padded_slice_apply(
        data_clipped, data_slice, extra_padding, padkw=padkw)

    if return_info:
        return data_sliced, transform
    else:
        return data_sliced


def _padded_slice_apply(data_clipped, data_slice, extra_padding, padkw=None):
    """
    Applies requested padding to an extracted data slice.
    """
    # Add any padding that is needed to behave like negative dims exist
    if sum(map(sum, extra_padding)) == 0:
        # The slice was completely in bounds
        data_sliced = data_clipped
    else:
        if padkw is None:
            padkw = {
                'mode': 'constant',
            }
        trailing_dims = len(data_clipped.shape) - len(extra_padding)
        if trailing_dims > 0:
            extra_padding = extra_padding + ([(0, 0)] * trailing_dims)
        data_sliced = np.pad(data_clipped, extra_padding, **padkw)

    st_dims = [(sl.start - pad_[0], sl.stop + pad_[1])
               for sl, pad_ in zip(data_slice, extra_padding)]

    # TODO: return a better transform back to the original space
    transform = {
        'st_dims': st_dims,
        'st_offset': [d[0] for d in st_dims]
    }
    return data_sliced, transform


def _padded_slice_embed(in_slice, data_dims, pad=None):
    """
    Embeds a "padded-slice" inside known data dimension.

    Returns the valid data portion of the slice with extra padding for regions
    outside of the available dimension.

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
            data_slice - Tuple[slice] a slice that can be applied to an array
                with with shape `data_dims`. This slice will not correspond to
                the full window size if the requested slice is out of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where slice is inside the data dims on left edge
        >>> from kwimage.im_core import *  # NOQA
        >>> in_slice = (slice(0, 10), slice(0, 10))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = _padded_slice_embed(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(0, 20, None), slice(0, 15, None))
        extra_padding = [(10, 0), (5, 0)]

    Example:
        >>> # Case where slice is bigger than the image
        >>> in_slice = (slice(-10, 400), slice(-10, 400))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = _padded_slice_embed(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(0, 300, None), slice(0, 300, None))
        extra_padding = [(20, 110), (15, 105)]

    Example:
        >>> # Case where slice is inside than the image
        >>> in_slice = (slice(10, 40), slice(10, 40))
        >>> data_dims  = [300, 300]
        >>> pad        = None
        >>> a, b = _padded_slice_embed(in_slice, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(10, 40, None), slice(10, 40, None))
        extra_padding = [(0, 0), (0, 0)]
    """
    low_dims = [sl.start for sl in in_slice]
    high_dims = [sl.stop for sl in in_slice]

    # Determine the real part of the image that can be sliced out
    data_slice_st = []
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
        data_slice_st.append((sl_low, sl_high))

        # Add extra padding when the window extends past the real part
        low_diff = sl_low - raw_low
        high_diff = raw_high - sl_high

        # Hand the case where both raw coordinates are out of bounds
        extra_low = max(0, low_diff + min(0, high_diff))
        extra_high = max(0, high_diff + min(0, low_diff))
        extra = (extra_low, extra_high)
        extra_padding.append(extra)

    data_slice = tuple(slice(s, t) for s, t in data_slice_st)
    return data_slice, extra_padding


def normalize(arr, mode='linear', alpha=None, beta=None, out=None):
    """
    Rebalance pixel intensities via contrast stretching.

    By default linearly stretches pixel intensities to minimum and maximum
    values.

    Note:
        DEPRECATED: this function has been MOVED to ``kwarray.normalize``
    """
    import kwarray
    return kwarray.normalize(arr, mode=mode, alpha=alpha, beta=beta, out=out)


def find_robust_normalizers(data, params='auto'):
    """
    Finds robust normalization statistics for a single observation

    Args:
        data (ndarray): a 1D numpy array where invalid data has already been removed

        params (str | dict): normalization params

    Returns:
        Dict[str, str | float]: normalization parameters

    TODO:
        - [ ] No Magic Numbers! Use first principles to deterimine defaults.
        - [ ] Probably a lot of literature on the subject.
        - [ ] Is this a kwarray function in general?

    Example:
        >>> from kwimage.im_core import *  # NOQA
        >>> data = np.random.rand(100)
        >>> norm_params1 = find_robust_normalizers(data, params='auto')
        >>> norm_params2 = find_robust_normalizers(data, params={'low': 0, 'high': 1.0})
        >>> norm_params3 = find_robust_normalizers(np.empty(0), params='auto')
        >>> print('norm_params1 = {}'.format(ub.repr2(norm_params1, nl=1)))
        >>> print('norm_params2 = {}'.format(ub.repr2(norm_params2, nl=1)))
        >>> print('norm_params3 = {}'.format(ub.repr2(norm_params3, nl=1)))
    """
    if data.size == 0:
        normalizer = {
            'type': None,
            'min_val': np.nan,
            'max_val': np.nan,
        }
    else:
        # should center the desired distribution to visualize on zero
        # beta = np.median(imdata)
        default_params = {
            'low': 0.01,
            'mid': 0.5,
            'high': 0.9,
            'mode': 'sigmoid',
        }
        if isinstance(params, str):
            if params == 'auto':
                params = {}
            else:
                raise KeyError(params)

        params = ub.dict_union(default_params, params)
        quant_low = params['low']
        quant_mid = params['mid']
        quant_high = params['high']
        qvals = [0, quant_low, quant_mid, quant_high, 1]
        quantile_vals = np.quantile(data, qvals)

        (quant_low_abs, quant_low_val, quant_mid_val, quant_high_val,
         quant_high_abs) = quantile_vals

        # TODO: we could implement a hueristic where we do a numerical inspection
        # of the intensity distribution. We could apply a normalization that is
        # known to work for data with that sort of histogram distribution.
        # This might involve fitting several parametarized distributions to the
        # data and choosing the one with the best fit. (check how many modes there
        # are).

        # inner_range = quant_high_val - quant_low_val
        # upper_inner_range = quant_high_val - quant_mid_val
        # upper_lower_range = quant_mid_val - quant_low_val
        # http://mathcenter.oxford.emory.edu/site/math117/shapeCenterAndSpread/

        # Compute amount of weight in each quantile
        quant_center_amount = (quant_high_val - quant_low_val)
        quant_low_amount = (quant_mid_val - quant_low_val)
        quant_high_amount = (quant_high_val - quant_mid_val)

        if math.isclose(quant_center_amount, 0):
            high_weight = 0.5
            low_weight = 0.5
        else:
            high_weight = quant_high_amount / quant_center_amount
            low_weight = quant_low_amount / quant_center_amount

        quant_high_residual = (1.0 - quant_high)
        quant_low_residual = (quant_low - 0.0)
        # todo: verify, having slight head fog, not 100% sure
        low_pad_val = quant_low_residual * (low_weight * quant_center_amount)
        high_pad_val = quant_high_residual * (high_weight * quant_center_amount)

        min_val = max(quant_low_abs, quant_low_val - low_pad_val)
        max_val = max(quant_high_abs, quant_high_val - high_pad_val)

        beta = quant_mid_val
        # division factor
        # from scipy.special import logit
        # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
        # This chooses alpha such the original min/max value will be pushed
        # towards -1 / +1.
        alpha = max(abs(min_val - beta), abs(max_val - beta)) / 6.212606

        normalizer = {
            'type': 'normalize',
            'mode': params['mode'],
            'min_val': min_val,
            'max_val': max_val,
            'beta': beta,
            'alpha': alpha,
        }
    return normalizer


def normalize_intensity(imdata, return_info=False, nodata=None, axis=None,
                        dtype=np.float32, params='auto', mask=None):
    """
    Normalize data intensities using heuristics to help put sensor data with
    extremely high or low contrast into a visible range.

    This function is designed with an emphasis on getting something that is
    reasonable for visualization.

    TODO:
        - [ ] Move to kwarray and renamed to robust_normalize?
        - [ ] Support for M-estimators?

    Args:
        imdata (ndarray): raw intensity data

        return_info (bool):
            if True, return information about the chosen normalization
            heuristic.

        params (str | dict):
            can contain keys, low, high, or center
            e.g. {'low': 0.1, 'center': 0.8, 'high': 0.9}

        axis (None | int):
            The axis to normalize over, if unspecified, normalize jointly

        nodata (None | int):
            A value representing nodata to leave unchanged during
            normalization, for example 0

        dtype (type) : can be float32 or float64

        mask (ndarray | None):
            A mask indicating what pixels are valid and what pixels should be
            considered nodata.  Mutually exclusive with ``nodata`` argument.
            A mask value of 1 indicates a VALID pixel. A mask value of 0
            indicates an INVALID pixel.

    Returns:
        ndarray: a floating point array with values between 0 and 1.

    Example:
        >>> from kwimage.im_core import *  # NOQA
        >>> import ubelt as ub
        >>> import kwimage
        >>> import kwarray
        >>> s = 512
        >>> bit_depth = 11
        >>> dtype = np.uint16
        >>> max_val = int(2 ** bit_depth)
        >>> min_val = int(0)
        >>> rng = kwarray.ensure_rng(0)
        >>> background = np.random.randint(min_val, max_val, size=(s, s), dtype=dtype)
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(s / 2)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(s / 2).translate(s / 2)
        >>> forground = np.zeros_like(background, dtype=np.uint8)
        >>> forground = poly1.fill(forground, value=255)
        >>> forground = poly2.fill(forground, value=122)
        >>> forground = (kwimage.ensure_float01(forground) * max_val).astype(dtype)
        >>> imdata = background + forground
        >>> normed, info = normalize_intensity(imdata, return_info=True)
        >>> print('info = {}'.format(ub.repr2(info, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(imdata, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(normed, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> from kwimage.im_core import *  # NOQA
        >>> import ubelt as ub
        >>> import kwimage
        >>> # Test on an image that is already normalized to test how it
        >>> # degrades
        >>> imdata = kwimage.grab_test_image() / 255

        >>> quantile_basis = {
        >>>     'mode': ['linear', 'sigmoid'],
        >>>     'high': [0.8, 0.9, 1.0],
        >>> }
        >>> quantile_grid = list(ub.named_product(quantile_basis))
        >>> quantile_grid += ['auto']
        >>> rows = []
        >>> rows.append({'key': 'orig', 'result': imdata})
        >>> for params in quantile_grid:
        >>>     key = ub.repr2(params, compact=1)
        >>>     result, info = normalize_intensity(imdata, return_info=True, params=params)
        >>>     print('key = {}'.format(key))
        >>>     print('info = {}'.format(ub.repr2(info, nl=1)))
        >>>     rows.append({'key': key, 'info': info, 'result': result})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(rows))
        >>> for row in rows:
        >>>     _, ax = kwplot.imshow(row['result'], fnum=1, pnum=pnum_())
        >>>     ax.set_title(row['key'])
    """
    import kwarray

    if axis is not None:
        # Hack, normalize each channel individually. This could
        # be implementd more effciently.
        assert not return_info
        reorg = imdata.swapaxes(0, axis)
        if mask is None:
            parts = []
            for item in reorg:
                part = normalize_intensity(item, nodata=nodata, axis=None)
                parts.append(part[None, :])
        else:
            reorg_mask = mask.swapaxes(0, axis)
            parts = []
            for item, item_mask in zip(reorg, reorg_mask):
                part = normalize_intensity(item, nodata=nodata, axis=None,
                                           mask=item_mask)
                parts.append(part[None, :])
        recomb = np.concatenate(parts, axis=0)
        final = recomb.swapaxes(0, axis)
        return final

    if imdata.dtype.kind == 'f':
        if mask is None:
            mask = ~np.isnan(imdata)

    if mask is None:
        if nodata is not None:
            mask = imdata != nodata

    if mask is None:
        imdata_valid = imdata
    else:
        imdata_valid = imdata[mask]

    assert not np.any(np.isnan(imdata_valid))

    normalizer = find_robust_normalizers(imdata_valid, params=params)

    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        imdata_valid_normalized = kwarray.normalize(
            imdata_valid.astype(dtype), mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
        if mask is None:
            imdata_normalized = imdata_valid_normalized
        else:
            imdata_normalized = imdata.copy()
            imdata_normalized[mask] = imdata_valid_normalized
    else:
        raise KeyError(normalizer['type'])

    if mask is not None:
        result = np.where(mask, imdata_normalized, imdata)
    else:
        result = imdata_normalized

    if return_info:
        return result, normalizer
    else:
        return result
