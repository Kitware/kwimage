"""
Wrappers around cv2 functions

Note: all functions in kwimage work with RGB input by default instead of BGR,
which is what the underlying cv2 functions use.
"""
import cv2
import numpy as np
import ubelt as ub
import numbers
from functools import lru_cache
from kwimage import im_core

__all__ = [
    'connected_components',
    'convert_colorspace',
    'gaussian_blur',
    'gaussian_patch',
    'imcrop',
    'imscale',
    'morphology',
]


_CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}


# https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
_CV2_BORDER_MODES = {
    'constant':    cv2.BORDER_CONSTANT,
    'replicate':   cv2.BORDER_REPLICATE,
    'reflect':     cv2.BORDER_REFLECT,
    'wrap':        cv2.BORDER_WRAP,
    'reflect101':  cv2.BORDER_REFLECT101,
    'transparent': cv2.BORDER_TRANSPARENT,
    # 'isolated':    cv2.BORDER_ISOLATED,
}

# https://www.projectpro.io/recipes/what-are-types-of-borders-which-can-be-made-opencv
# https://answers.opencv.org/question/50706/border_reflect-vs-border_reflect_101/
# cv2.BORDER_REPLICATE creates a border by replicating the last element of the image like this: PPPPPPP|ProjectPro|ooooooo
# cv2.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
# cv2.BORDER_WRAP - forms a border like this: ojectPro|ProjectPro|ProjectP
# cv2.BORDER_REFLECT101 - Same as BORDER_REFLECT, but with a slight change in which the outter-most pixels (a or h) are not repeated : gfedcb|abcdefgh|gfedcba


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
        >>> # xdoctest: +REQUIRES(module:cv2)
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
    if isinstance(interpolation, str):
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


def _coerce_border_mode(border_mode, default=cv2.BORDER_CONSTANT):
    """
    Converts border_mode into flags suitable cv2 functions

    Args:
        border_mode (int | str | None):
            string or cv2-style interpolation type

        default (int):
            the value to use if the previous argument is None

    Returns:
        int: flag specifying borderMode type that can be passed to
           functions like cv2.warpAffine, etc...

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> flag = _coerce_border_mode('constant')
        >>> assert flag == cv2.BORDER_CONSTANT
        >>> flag = _coerce_border_mode(cv2.BORDER_CONSTANT)
        >>> assert flag == cv2.BORDER_CONSTANT
        >>> flag = _coerce_border_mode(None, default='reflect')
        >>> assert flag == cv2.BORDER_REFLECT
        >>> # xdoctest: +REQUIRES(module:pytest)
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     _coerce_border_mode(3.4)
        >>> import pytest
        >>> with pytest.raises(KeyError):
        >>>     _coerce_border_mode('foobar')
    """
    if border_mode is None:
        border_mode = default

    # Handle coercion from string to cv2 integer flag
    if isinstance(border_mode, str):
        try:
            return _CV2_BORDER_MODES[border_mode]
        except KeyError:
            raise KeyError(
                'Invalid border_mode value={!r}. '
                'Valid strings for border_mode are {}'.format(
                    border_mode, list(_CV2_BORDER_MODES.keys())))
    elif isinstance(border_mode, numbers.Integral):
        return int(border_mode)
    else:
        raise TypeError(
            'Invalid border_mode value={!r}. '
            'Type must be int or string but got {!r}'.format(
                border_mode, type(border_mode)))


def _coerce_border_value(border_value, default=0, image=None):
    """
    Handles cv2 border values

    Args:
        border_value (int | float | Iterable[int | float]):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.

        default (int):
            the value to use if the previous argument is None

        image (None | ndarray):
            The image image the operation will be applied to.

    Returns:
        ...
    """
    borderValue = border_value
    if borderValue is None:
        borderValue = default

    if isinstance(borderValue, str):
        from kwimage import im_color
        borderValue = im_color.Color(borderValue).forimage(image)
    elif not ub.iterable(borderValue):
        # convert scalar border value to a tuple to ensure the user always
        # fully defines the output. (and to have conciseness)
        num_chan = im_core.num_channels(image)
        # More than 4 channels will start to wrap around, so this is fine.
        borderValue = (borderValue,) * min(4, num_chan)

    if len(borderValue) > 4:
        # FIXME; opencv bug
        # https://github.com/opencv/opencv/issues/22283
        raise ValueError('borderValue cannot have more than 4 components. '
                         'OpenCV #22283 describes why')
    return borderValue


def _coerce_border_mode_value(border_mode, border_value, image):
    """
    Common code for warp_affine and warp_persepctive
    """

    borderMode = _coerce_border_mode(border_mode)
    if isinstance(border_value, str):
        # it is annoying to have borer value / border mode be different
        # if it is a string assume the border mode is a strategy
        # todo: cleanup
        assert border_mode is None
        border_mode = border_value
        border_value = 0
    borderValue = _coerce_border_value(border_value, image=image)
    return borderMode, borderValue


def imscale(img, scale, interpolation=None, return_scale=False):
    """
    DEPRECATED and removed: use imresize instead
    """
    ub.schedule_deprecation(
        modname='kwimage',
        name='imscale',
        type='function',
        migration='Use imresize instead.',
        deprecate='0.9.0',
        error='0.9.5',
        remove='1.0.0',
    )


def imcrop(img, dsize, about=None, origin=None, border_value=None,
           interpolation='nearest'):
    """
    Crop an image about a specified point, padding if necessary.

    This is like :func:`PIL.Image.Image.crop` with more convenient arguments,
    or :func:`cv2.getRectSubPix` without the baked-in bilinear interpolation.

    Args:
        img (ndarray): image to crop

        dsize (Tuple[None | int, None | int]): the desired width and height
            of the new image. If a dimension is None, then it is automatically
            computed to preserve aspect ratio. This can be larger than the
            original dims; if so, the cropped image is padded with
            border_value.

        about (Tuple[str | int, str | int]): the location to crop about.
            Mutually exclusive with origin. Defaults to top left.
            If ints (w,h) are provided, that will be the center of the cropped
            image.
            There are also string codes available:
            'lt': make the top left point of the image the top left point of
            the cropped image.  This is equivalent to
            ``img[:dsize[1], :dsize[0]]``, plus padding.
            'rb': make the bottom right point of the image the bottom right
            point of the cropped image.  This is equivalent to
            ``img[-dsize[1]:, -dsize[0]:]``, plus padding.
            'cc': make the center of the image the center of the cropped image.
            Any combination of these codes can be used, ex. 'lb', 'ct', ('r',
            200), ...

        origin (Tuple[int, int] | None):
            the origin of the crop in (x,y) order (same order as dsize/about).
            Mutually exclusive with about. Defaults to top left.

        border_value (Number | Tuple | str):
            any border border_value accepted by cv2.copyMakeBorder,
            ex. [255, 0, 0] (blue). Default is 0.

        interpolation (str):
            Can be 'nearest', in which case integral cropping is used.
            Can also be 'linear', in which case cv2.getRectSubPix is used.
            Defaults to 'nearest'.

    Returns:
        ndarray: the cropped image

    SeeAlso:
        :func:`kwarray.padded_slice` - a similar function for working with
            "negative slices".

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import numpy as np
        >>> #
        >>> img = kwimage.grab_test_image('astro', dsize=(32, 32))[..., 0:3]
        >>> #
        >>> # regular crop
        >>> new_img1 = kwimage.imcrop(img, dsize=(5,6))
        >>> assert new_img1.shape[0:2] == (6, 5)
        >>> #
        >>> # padding for coords outside the image bounds
        >>> new_img2 = kwimage.imcrop(img, dsize=(5,6),
        >>>             origin=(-1,0), border_value=[1, 0, 0])
        >>> assert np.all(new_img2[:, 0, 0:3] == [1, 0, 0])
        >>> #
        >>> # codes for corner- and edge-centered cropping
        >>> new_img3 = kwimage.imcrop(img, dsize=(5,6),
        >>>             about='cb')
        >>> #
        >>> # special code for bilinear interpolation
        >>> # with floating-point coordinates
        >>> new_img4 = kwimage.imcrop(img, dsize=(5,6),
        >>>             about=(5.5, 8.5), interpolation='linear')
        >>> #
        >>> # use with bounding boxes
        >>> bbox = kwimage.Boxes.random(scale=5, rng=132).to_xywh().quantize()
        >>> origin, dsize = np.split(bbox.data[0], 2)
        >>> new_img5 = kwimage.imcrop(img, dsize=dsize,
        >>>             origin=origin)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=6)
        >>> kwplot.imshow(img, pnum=pnum_())
        >>> kwplot.imshow(new_img1, pnum=pnum_())
        >>> kwplot.imshow(new_img2, pnum=pnum_())
        >>> kwplot.imshow(new_img3, pnum=pnum_())
        >>> kwplot.imshow(new_img4, pnum=pnum_())
        >>> kwplot.imshow(new_img5, pnum=pnum_())
        >>> kwplot.show_if_requested()
    """
    import numbers

    old_h, old_w = img.shape[0:2]

    assert len(dsize) == 2
    new_w, new_h = dsize
    assert isinstance(new_w, numbers.Integral)
    assert isinstance(new_h, numbers.Integral)

    if new_w is None:
        assert new_h is not None
        new_w = int(np.round(new_h * old_w / old_h))
    elif new_h is None:
        assert new_w is not None
        new_h = int(np.round(new_w * old_h / old_w))

    old_h, old_w = img.shape[0:2]

    if origin is not None:

        if about is not None:
            raise AssertionError('provide at most one of "about" or "origin"')

        assert len(origin) == 2
        new_x, new_y = origin
        assert isinstance(new_x, numbers.Integral)
        assert isinstance(new_y, numbers.Integral)
        cen_w = new_x + new_w // 2
        cen_h = new_y + new_h // 2

    elif about is not None:

        if origin is not None:
            raise AssertionError('provide at most one of "about" or "origin"')

        assert len(about) == 2

        if about[0] == 'l':
            cen_w = new_w // 2
        elif about[0] == 'r':
            cen_w = old_w - (new_w - new_w // 2)
        elif about[0] == 'c':
            cen_w = old_w // 2
        elif isinstance(about[0], numbers.Integral):
            cen_w = about[0]
        elif isinstance(about[0], numbers.Real):
            if interpolation != 'linear':
                raise ValueError('interpolation must be linear when about is real valued')
            cen_w = about[0]
        else:
            raise ValueError('Invalid about code {}. Must be [l | c | r | int][t | c | b | int]'.format(about))

        if about[1] == 't':
            cen_h = new_h // 2
        elif about[1] == 'b':
            cen_h = old_h - (new_h - new_h // 2)
        elif about[1] == 'c':
            cen_h = old_h // 2
        elif isinstance(about[1], numbers.Integral):
            cen_h = about[1]
        elif isinstance(about[1], numbers.Real):
            if interpolation != 'linear':
                raise ValueError('interpolation must be linear when about is real valued')
            cen_h = about[1]
        else:
            raise ValueError('Invalid about code {}. Must be [l | c | r | int][t | c | b | int]'.format(about))

    else:
        # take top left as the origin
        cen_w = new_w // 2
        cen_h = new_h // 2

    if interpolation == 'linear':
        new_img = cv2.getRectSubPix(img, dsize, (cen_h, cen_w))
    elif interpolation == 'nearest':
        # build a patch that may go outside the image bounds
        ymin, ymax = cen_w - new_w // 2, cen_w + (new_w - new_w // 2)
        xmin, xmax = cen_h - new_h // 2, cen_h + (new_h - new_h // 2)

        # subtract out portions that leave the image bounds
        lft, ymin = - min(0, ymin), max(0, ymin)
        rgt, ymax = max(0, ymax - old_w), min(old_w, ymax)
        top, xmin = - min(0, xmin), max(0, xmin)
        bot, xmax = max(0, xmax - old_h), min(old_h, xmax)

        # slice the image using the corrected bounds and append the rest as a border
        new_img = cv2.copyMakeBorder(img[xmin:xmax, ymin:ymax], top, bot, lft, rgt,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=border_value)
    else:
        raise KeyError(interpolation)

    if len(new_img.shape) == 2 and len(img.shape) == 3:
        # Make the output shape consistent with the input
        assert img.shape[2] == 1, 'have 1 input channel in this case'
        new_img = new_img[..., None]

    return new_img


def _cv2_input_fixer(img):
    """
    OpenCV is very particular about its inputs, we would like to loosen those
    requirements by seemlessly detecting and fixing dtypes when possible
    """
    # Notes:
    # CV2 Data Types
    # https://udayawijenayake.com/2021/06/07/opencv-data-types/
    # sorted([{'cv_code': k, 'cv_value': getattr(cv2, k)} for k in dir(cv2) if k.startswith('CV_')], key=lambda x: x['cv_value'])
    # [
    #     {'cv_code': 'CV_8U', 'cv_value': cv2.CV_8U},
    #     {'cv_code': 'CV_8S', 'cv_value': cv2.CV_8S},
    #     {'cv_code': 'CV_16U', 'cv_value': cv2.CV_16U},
    #     {'cv_code': 'CV_16S', 'cv_value': cv2.CV_16S},
    #     {'cv_code': 'CV_32SC1', 'cv_value': cv2.CV_32SC1},
    #     {'cv_code': 'CV_32F', 'cv_value': cv2.CV_32F},
    #     {'cv_code': 'CV_64FC1', 'cv_value': cv2.CV_64FC1},
    # ]
    final_dtype = None
    if img.dtype.kind == 'b':
        final_dtype = img.dtype
        img = img.astype(np.uint8)

    # TODO: can we be flexible with more data types?
    # # Allow upcast
    # if img.dtype.kind == 'i' and img.dtype.itemsize == 1:
    #     final_dtype = img.dtype
    #     img = img.astype(np.int16)
    # if img.dtype.kind == 'i' and img.dtype.itemsize == 4:
    #     # Allowing memory increase for the sake of easy of use
    #     final_dtype = img.dtype
    #     img = img.astype(np.int64)
    #     ...
    return img, final_dtype


DTYPE_KEY_TO_DTYPE = {
    ('b', 1): bool,

    ('u', 1): np.uint8,
    ('u', 2): np.uint16,
    ('u', 4): np.uint32,
    ('u', 8): np.uint64,

    ('i', 1): np.int8,
    ('i', 2): np.int16,
    ('i', 4): np.int32,
    ('i', 8): np.int64,

    ('f', 2): np.float16,
    ('f', 4): np.float32,
    ('f', 8): np.float64,
}
_HAS_FLOAT128 = hasattr(np, 'float128')

if _HAS_FLOAT128:
    DTYPE_KEY_TO_DTYPE[('f', 16)] = np.float128

DTYPE_TO_DTYPE_KEY = ub.invert_dict(DTYPE_KEY_TO_DTYPE)


def __build_cv2_allowed_dtypes():
    default = ub.udict({
        DTYPE_TO_DTYPE_KEY[bool]: np.uint8,
        DTYPE_TO_DTYPE_KEY[np.uint8]: np.uint8,

        DTYPE_TO_DTYPE_KEY[np.uint16]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.uint32]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.uint64]: np.float32,

        DTYPE_TO_DTYPE_KEY[np.int8]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.int16]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.int32]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.int64]: np.float32,

        DTYPE_TO_DTYPE_KEY[np.int8]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.float16]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.float32]: np.float32,
        DTYPE_TO_DTYPE_KEY[np.float64]: np.float32,
    })
    if _HAS_FLOAT128:
        default[DTYPE_TO_DTYPE_KEY[np.float128]] = np.float32

    CV2_ALLOWED_DTYPE_MAPPINGS = {}
    CV2_ALLOWED_DTYPE_MAPPINGS['uint8,float32'] = default
    CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,float32'] = CV2_ALLOWED_DTYPE_MAPPINGS['uint8,float32'] | {
        DTYPE_TO_DTYPE_KEY[np.int8]: np.int16,
        DTYPE_TO_DTYPE_KEY[np.int16]: np.int16,
    }
    CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,int32,float32'] = CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,float32']  | {
        DTYPE_TO_DTYPE_KEY[np.uint16]: np.int32,
        DTYPE_TO_DTYPE_KEY[np.int32]: np.int32,
    }
    CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,int32,float32,float64'] = CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,int32,float32'] | {
        DTYPE_TO_DTYPE_KEY[np.int64]: np.float64,
        DTYPE_TO_DTYPE_KEY[np.float64]: np.float64,
    }
    if _HAS_FLOAT128:
        CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,int32,float32,float64'][DTYPE_TO_DTYPE_KEY[np.float128]] = np.float64

    CV2_ALLOWED_DTYPE_MAPPINGS['uint8,uint16,int8,int16,int32,float32'] = CV2_ALLOWED_DTYPE_MAPPINGS['uint8,int16,int32,float32'] | {
        DTYPE_TO_DTYPE_KEY[np.int64]: np.float64,
        DTYPE_TO_DTYPE_KEY[np.int8]: np.int8,
        DTYPE_TO_DTYPE_KEY[np.int32]: np.int32,
        DTYPE_TO_DTYPE_KEY[np.uint16]: np.uint16,
        DTYPE_TO_DTYPE_KEY[np.float64]: np.float32,
    }
    if _HAS_FLOAT128:
        CV2_ALLOWED_DTYPE_MAPPINGS['uint8,uint16,int8,int16,int32,float32'][DTYPE_TO_DTYPE_KEY[np.float128]] = np.float32

    for k in CV2_ALLOWED_DTYPE_MAPPINGS.keys():
        CV2_ALLOWED_DTYPE_MAPPINGS[k] = CV2_ALLOWED_DTYPE_MAPPINGS[k].map_values(np.dtype)
    return CV2_ALLOWED_DTYPE_MAPPINGS


CV2_ALLOWED_DTYPE_MAPPINGS = __build_cv2_allowed_dtypes()


def _cv2_input_fixer_v2(img, allowed_types='uint8,int16,int32,float32,float64',
                        contiguous=True, owndata=False):
    """
    OpenCV is very particular about its inputs, we would like to loosen those
    requirements by seemlessly detecting and fixing dtypes when possible

    Example:
        from kwimage.im_cv2 import _cv2_input_fixer_v2  # NOQA
        img = np.random.rand(32, 32).astype(np.int64)
        fixed_img, final_dtype = _cv2_input_fixer_v2(img)
        print(f'img.dtype={img.dtype}')
        print(f'fixed_img.dtype={fixed_img.dtype}')
        print(f'final_dtype={final_dtype}')
    """
    # Notes:
    # CV2 Data Types
    # https://udayawijenayake.com/2021/06/07/opencv-data-types/
    in_dtype = img.dtype
    in_dtype_key = in_dtype.kind, in_dtype.itemsize
    out_dtype = CV2_ALLOWED_DTYPE_MAPPINGS[allowed_types][in_dtype_key]
    out_dtype_key = out_dtype.kind, out_dtype.itemsize

    if out_dtype_key == in_dtype_key:
        final_dtype = None
    else:
        final_dtype = img.dtype
        img = img.astype(out_dtype, copy=False)

    if contiguous and not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    if owndata and not img.flags['OWNDATA']:
        img = img.copy()

    return img, final_dtype


def _cv2_imresize(img, scale=None, dsize=None, max_dim=None, min_dim=None,
                  interpolation=None, grow_interpolation=None, letterbox=False,
                  return_info=False, antialias=False, border_value=0):
    """
    Resize an image via a scale factor, final size, or size and aspect ratio.

    Wraps and generalizes cv2.resize, allows for specification of either a
    scale factor, a final size, or the final size for a particular dimension.

    Note:
        As described in [ResizeConfusion]_, this each entry in the image array
        as representing the center of a pixel. This is the pixels_are='area'
        approach, or align_corners=False in pytorch. It is equivalent to a
        shift and scale in warp_affine (which by default uses align corners).

    Note:
        The border mode cannot be specified here and seems to always be reflect
        in the underlying cv2 implementation.

    Args:
        img (ndarray): image to resize

        scale (float | Tuple[float, float]):
            Desired floating point scale factor. If a tuple, the dimension
            ordering is x,y.  Mutually exclusive with dsize, min_dim, max_dim.

        dsize (Tuple[int | None, int | None] | None):
            The desired width and height of the new image. If a dimension is
            None, then it is automatically computed to preserve aspect ratio.
            Mutually exclusive with scale, min_dim, max_dim.

        max_dim (int):
            New size of the maximum dimension, the other dimension is scaled to
            maintain aspect ratio. Mutually exclusive with scale, dsize,
            min_dim.

        min_dim (int):
            New size of the minimum dimension, the other dimension is scaled to
            maintain aspect ratio. Mutually exclusive with scale, dsize,
            max_dim.

        interpolation (str | int):
            The interpolation key or code (e.g. linear lanczos). By default
            "area" is used if the image is shrinking and "lanczos" is used if
            the image is growing. Note, if this is explicitly set, then it will
            be used regardless of if the image is growing or shrinking. Set
            ``grow_interpolation`` to change the default for an enlarging
            interpolation.

        grow_interpolation (str | int):
            The interpolation key or code to use when the image is being
            enlarged. Does nothing if "interpolation" is explicitly given. If
            "interpolation" is not specified "area" is used when shrinking.
            Defaults to "lanczos".

        letterbox (bool):
            If used in conjunction with dsize, then the image is scaled and
            translated to fit in the center of the new image while maintaining
            aspect ratio. Border padding is added if necessary.  Defaults to
            False.

        return_info (bool):
            if True returns information about the final transformation in a
            dictionary. If there is an offset, the scale is applied before the
            offset when transforming to the new resized space.  Defaults to
            False.

        antialias (bool):
            if True blurs to anti-alias before downsampling.
            Defaults to False.

        border_value (int | float | Iterable[int | float]):
            if letterbox is True, this is used as the constant fill value.

    Returns:
        ndarray | Tuple[ndarray, Dict] :
            the new image and optionally an info dictionary if
            `return_info=True`

    References:
        .. [ResizeConfusion] https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
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
        >>> # xdoctest: +REQUIRES(module:cv2)
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

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Check aliasing
        >>> import kwimage
        >>> #img = kwimage.grab_test_image('checkerboard')
        >>> img = kwimage.grab_test_image('pm5644')
        >>> # test with nans
        >>> img = kwimage.ensure_float01(img)
        >>> img[100:200, 400:700] = np.nan
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

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Test single pixel resize
        >>> import kwimage
        >>> import numpy as np
        >>> assert kwimage.imresize(np.random.rand(1, 1, 3), scale=3).shape == (3, 3, 3)
        >>> assert kwimage.imresize(np.random.rand(1, 1), scale=3).shape == (3, 3)

        # cv2.resize(np.random.rand(1, 1, 3), (3, 3))


    TODO:
        - [X] When interpolation is area and the number of channels > 4 cv2.resize will error but it is fine for linear interpolation
        - [ ] TODO: add padding options when letterbox=True

        - [ ] Allow for pre-clipping when letterbox=True
    """
    _mutex_args = [scale, dsize, max_dim, min_dim]
    _num_mutex_args = sum(a is not None for a in _mutex_args)
    if _num_mutex_args > 1:
        raise ValueError(ub.paragraph(
            '''
            May only specify EXACTLY one of scale, dsize, max_dim, xor min_dim'
            Got scale={}, dsize={}, max_dim={}, min_dim={}
            ''').format(*_mutex_args))
    elif _num_mutex_args == 0:
        # None of the scale params were specified, return the image as-is
        return img

    def _aa_resize(a, scale, dsize, interpolation):
        sx, sy = scale
        if sx < 1 or sy < 1:
            a, sx, sy = _cv2_prepare_downscale(a, sx, sy)
        return cv2.resize(a, dsize=dsize, interpolation=interpolation)

    def _regular_resize(a, scale, dsize, interpolation):
        return cv2.resize(a, dsize=dsize, interpolation=interpolation)

    def _patched_resize(img, scale, dsize, interpolation):
        if old_w == 1 and old_h == 1:
            # Best we can do in this case.
            new_w, new_h = dsize
            if len(img.shape) == 2:
                return np.tile(img, (new_w, new_h))
            else:
                return np.tile(img, (new_w, new_h, 1))
        img = _cv2_imputation(img)
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

    old_w, old_h = img.shape[0:2][::-1]

    # Compute the new size based on the input arguments
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
        if new_h is None:
            new_h = old_h
            new_w = old_w
        else:
            new_w = new_h * old_w / old_h
    elif new_h is None:
        assert new_w is not None
        new_h = new_w * old_h / old_w

    # Determine other settings and backends
    grow_interpolation = _coerce_interpolation(grow_interpolation)

    if antialias:
        _chosen_resize = _aa_resize
    else:
        _chosen_resize = _regular_resize

    img, final_dtype = _cv2_input_fixer(img)
    # if not img.flags['C_CONTIGUOUS']:
    #     img = np.ascontiguousarray(img)

    if letterbox:
        if dsize is None:
            raise ValueError('letterbox can only be used with dsize')
        orig_size = np.array(img.shape[0:2][::-1])
        w, h = dsize
        if w is None:
            w = orig_size[0]
        if h is None:
            h = orig_size[1]
        dsize = (w, h)
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

        borderValue = _coerce_border_value(border_value, image=embed_img)
        new_img = cv2.copyMakeBorder(
            embed_img, top, bot, left, right, borderType=cv2.BORDER_CONSTANT,
            value=borderValue)
        if return_info:
            info = {
                'offset': offset,
                'scale': scale,
                'dsize': dsize,
                'embed_size': embed_size,
            }
    else:
        # Use np.round over python round, which has incompatible behavior
        old_dsize = (old_w, old_h)
        new_w_ = max(1, int(np.round(new_w)))
        new_h_ = max(1, int(np.round(new_h)))
        new_dsize = (new_w_, new_h_)
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

    if final_dtype is not None:
        # The output type and input type should match
        new_img = new_img.astype(final_dtype)

    if len(new_img.shape) == 2 and len(img.shape) == 3:
        # Make the output shape consistent with the input
        assert img.shape[2] == 1, 'have 1 input channel in this case'
        new_img = new_img[..., None]

    if return_info:
        return new_img, info
    else:
        return new_img


def convert_colorspace(img, src_space, dst_space, copy=False,
                       implicit=False, dst=None):
    """
    Converts colorspace of img.

    Convenience function around :func:`cv2.cvtColor`

    Args:
        img (ndarray): image data with float32 or uint8 precision

        src_space (str): input image colorspace. (e.g. BGR, GRAY)

        dst_space (str): desired output colorspace. (e.g. RGB, HSV, LAB)

        implicit (bool):
            if False, the user must correctly specify if the input/output
                colorspaces contain alpha channels.
            If True and the input image has an alpha channel, we modify
                src_space and dst_space to ensure they both end with "A".

        dst (ndarray[Any, UInt8]): inplace-output array.

    Returns:
        ndarray: img - image data

    Note:
        Note the LAB and HSV colorspaces in float do not go into the 0-1 range.

        For HSV the floating point range is:
            0:360, 0:1, 0:1
        For LAB the floating point range is:
            0:100, -86.1875:98.234375, -107.859375:94.46875
            (Note, that some extreme combinations of a and b are not valid)

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
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
        print('minvals = {}'.format(ub.urepr(minvals, nl=0)))
        print('maxvals = {}'.format(ub.urepr(maxvals, nl=0)))
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

        sigma (float | Tuple[float, float] | None):
            Gaussian standard deviation. If unspecified, it is derived using
            the formula ``0.3 * ((s - 1) * 0.5 - 1) + 0.8`` as described in
            [Cv2GaussKern]_.

    Returns:
        ndarray

    References:
        .. [Cv2GaussKern] http://docs.opencv.org/modules/imgproc/doc/filtering.html#getgaussiankernel

    TODO:
        - [ ] Look into this C-implementation https://kwgitlab.kitware.com/computer-vision/heatmap/blob/master/heatmap/heatmap.c

    CommandLine:
        xdoctest -m kwimage.im_cv2 gaussian_patch --show

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import numpy as np
        >>> shape = (88, 24)
        >>> sigma = None  # 1.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> assert np.all(np.isclose(sum_, 1.0))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> norm = (gausspatch - gausspatch.min()) / (gausspatch.max() - gausspatch.min())
        >>> kwplot.imshow(norm)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import numpy as np
        >>> shape = (24, 24)
        >>> sigma = 3.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> assert np.all(np.isclose(sum_, 1.0))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> norm = (gausspatch - gausspatch.min()) / (gausspatch.max() - gausspatch.min())
        >>> kwplot.imshow(norm)
        >>> kwplot.show_if_requested()
    """
    if sigma is None:
        sigma_x = 0.3 * ((shape[0] - 1) * 0.5 - 1) + 0.8
        sigma_y = 0.3 * ((shape[1] - 1) * 0.5 - 1) + 0.8
    elif isinstance(sigma, (float, int)):
        sigma_x = sigma_y = sigma
    else:
        sigma_x, sigma_y = sigma
    # see hesaff/src/helpers.cpp : computeCircularGaussMask
    kernel_d0 = cv2.getGaussianKernel(shape[0], sigma_x)
    if shape[0] == shape[1] and sigma_y == sigma_x:
        kernel_d1 = kernel_d0
    else:
        kernel_d1 = cv2.getGaussianKernel(shape[1], sigma_y)
    gausspatch = kernel_d0.dot(kernel_d1.T)
    return gausspatch


def _auto_kernel_sigma(kernel=None, sigma=None, autokernel_mode='ours'):
    """
    Attempt to determine sigma and kernel size from heuristics

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> _auto_kernel_sigma(None, None)
        >>> _auto_kernel_sigma(3, None)
        >>> _auto_kernel_sigma(None, 0.8)
        >>> _auto_kernel_sigma(7, None)
        >>> _auto_kernel_sigma(None, 1.4)

    Ignore:
        >>> # xdoctest: +REQUIRES(--demo)
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> rows = []
        >>> for k in np.arange(3, 101, 2):
        >>>     s = _auto_kernel_sigma(k, None)[1][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_sigma'})
        >>> #
        >>> sigmas = np.array([r['s'] for r in rows])
        >>> other = np.linspace(0, sigmas.max() + 1, 100)
        >>> sigmas = np.unique(np.hstack([sigmas, other]))
        >>> sigmas.sort()
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='cv2')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (cv2)'})
        >>> #
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='ours')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (ours)'})
        >>> import pandas as pd
        >>> df = pd.DataFrame(rows)
        >>> p = df.pivot(['s'], ['type'], ['k'])
        >>> print(p[~p.droplevel(0, axis=1).auto_sigma.isnull()])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> sns.lineplot(data=df, x='s', y='k', hue='type')
    """
    import numbers
    if kernel is None and sigma is None:
        kernel = 3

    if kernel is not None:
        if isinstance(kernel, numbers.Integral):
            k_x = k_y = kernel
        else:
            k_x, k_y = kernel

    if sigma is None:
        # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L344
        sigma_x = 0.3 * ((k_x - 1) * 0.5 - 1) + 0.8
        sigma_y = 0.3 * ((k_y - 1) * 0.5 - 1) + 0.8
    else:
        if isinstance(sigma, numbers.Number):
            sigma_x = sigma_y = sigma
        else:
            sigma_x, sigma_y = sigma

    if kernel is None:
        if autokernel_mode == 'zero':
            # When 0 computed internally via cv2
            k_x = k_y = 0
        elif autokernel_mode == 'cv2':
            # This is the CV2 definition
            # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L387
            depth_factor = 3  # or 4 for non-uint8
            k_x = int(round(sigma_x * depth_factor * 2 + 1)) | 1
            k_y = int(round(sigma_y * depth_factor * 2 + 1)) | 1
        elif autokernel_mode == 'ours':
            # But I think this definition makes more sense because it keeps
            # sigma and the kernel in agreement more often
            """
            # Our hueristic is computed via solving the sigma heuristic for k
            import sympy as sym
            s, k = sym.symbols('s, k', rational=True)
            sa = sym.Rational('3 / 10') * ((k - 1) / 2 - 1) + sym.Rational('8 / 10')
            sym.solve(sym.Eq(s, sa), k)
            """
            k_x = max(3, round(20 * sigma_x / 3 - 7 / 3)) | 1
            k_y = max(3, round(20 * sigma_y / 3 - 7 / 3)) | 1
        else:
            raise KeyError(autokernel_mode)
    sigma = (sigma_x, sigma_y)
    kernel = (k_x, k_y)
    return kernel, sigma


def gaussian_blur(image, kernel=None, sigma=None, border_mode=None, dst=None):
    """
    Apply a gausian blur to an image.

    This is a simple wrapper around :func:`cv2.GaussianBlur` with concise
    parametarization and sane defaults.

    Args:
        image (ndarray):
            the input image

        kernel (int | Tuple[int, int]):
            The kernel size in x and y directions.

        sigma (float | Tuple[float, float]):
            The gaussian spread in x and y directions.

        border_mode (str | int | None):
            Border text code or cv2 integer. Border codes are 'constant'
            (default), 'replicate', 'reflect', 'reflect101', and 'transparent'.

        dst (ndarray | None): optional inplace-output array.

    Returns:
        ndarray: the blurred image

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> image = kwimage.ensure_float01(kwimage.grab_test_image('astro'))
        >>> blurred1 = kwimage.gaussian_blur(image)
        >>> blurred2 = kwimage.gaussian_blur(image, kernel=9)
        >>> blurred3 = kwimage.gaussian_blur(image, sigma=2)
        >>> blurred4 = kwimage.gaussian_blur(image, sigma=(2, 5), kernel=5)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=4, nCols=1)
        >>> blurs = [blurred1, blurred2, blurred3, blurred4]
        >>> for blurred in blurs:
        >>>     diff = np.abs(image - blurred)
        >>>     stack = kwimage.stack_images([image, blurred, diff], pad=10, axis=1)
        >>>     kwplot.imshow(stack, pnum=pnum_())
        >>> kwplot.show_if_requested()
    """
    if kernel is None and sigma is None:
        kernel = 3

    if kernel is not None:
        if isinstance(kernel, int):
            k_x = k_y = kernel
        else:
            k_x, k_y = kernel

    if sigma is not None:
        if isinstance(sigma, (float, int)):
            sigma_x = sigma_y = sigma
        else:
            sigma_x, sigma_y = sigma

    if sigma is None:
        # https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
        # sigma_x = 0.3 * ((k_x - 1) * 0.5 - 1) + 0.8
        # sigma_y = 0.3 * ((k_y - 1) * 0.5 - 1) + 0.8
        # When 0 computed via cv2 from kernel
        sigma_x = sigma_y = 0

    if kernel is None:
        # When 0 computed via cv2 from sigma
        k_x, k_y = 0, 0

    borderType = _coerce_border_mode(border_mode)
    image = _cv2_imputation(image)
    blurred = cv2.GaussianBlur(
        image, (k_x, k_y), sigmaX=sigma_x, sigmaY=sigma_y,
        borderType=borderType, dst=dst
    )
    return blurred


def _cv2_warp_affine(image, transform, dsize=None, antialias=False,
                     interpolation='linear', border_mode=None, border_value=0,
                     large_warp_dim=None, return_info=False,
                     origin_convention='center'):
    """
    Applies an affine transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image as a numpy array.
            Note: this is passed directly to cv2, so it is best to ensure that
            it is contiguous and using a dtype that cv2 can handle.

        transform (ndarray | dict | kwimage.Affine): a coercable affine matrix.
            See :class:`kwimage.Affine` for details on what can be coerced.

        dsize (Tuple[int, int] | None | str):
            A integer width and height tuple of the resulting "canvas" image.

            If None, then the input image size is used.

            If specified as a string, dsize is computed based on the given
            heuristic.

            If 'positive' (or 'auto'), dsize is computed such that the positive
            coordinates of the warped image will fit in the new canvas. In this
            case, any pixel that maps to a negative coordinate will be clipped.
            This has the property that the input transformation is not
            modified. NOTE: there are issues with this when the transformation
            includes rotation of reflections.

            If 'content' (or 'max'), the transform is modified with an extra
            translation such that both the positive and negative coordinates of
            the warped image will fit in the new canvas.

        antialias (bool)
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur. Defaults to False

        interpolation (str | int):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area. Defaults to "linear".

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float] | str):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.
            If a string, it indicates a border_mode that is not constant.
            In this case border mode should not be given.

        large_warp_dim (int | None | str):
            If specified, perform the warp piecewise in chunks of the specified
            size. If "auto", it is set to the maximum "short" value in numpy.
            This works around a limitation of cv2.warpAffine, which must have
            image dimensions < SHRT_MAX (=32767 in version 4.5.3)

        return_info (bool):
            if True, returns information about the operation. In the case
            where dsize="content", this includes the modified transformation.

        origin_convention (str):
            Controls the interpretation of the underlying raster.
            Can be "center" (default opencv behavior), or "corner" (matches
            torchvision / detectron2 behavior).
            If "center", then center of the top left pixel is at (0, 0), and
            the top left corner is at (-0.5, -0.5).
            If "center", then center of the top left pixel is at (0.5, 0.5), and
            the top left corner is at (0, 0).
            Currently defaults to "center", but in the future we may change the
            default to "corner".  For more info see [WhereArePixels]_.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            the warped image, or if return info is True, the warped image and
            the info dictionary.

    TODO:
        - [ ] When dsize='positive' but the transform contains an axis flip,
              the width / height of the box will become negative.  Should we
              adjust for this?

    References:
        .. [WhereArePixels] https://ppwwyyxx.com/blog/2021/Where-are-Pixels/

    SeeAlso:

        :func:`kwimage.im_transform.warp_affine` -
            for doctests that cover this function.

    Ignore:
        transform_ = transform
        params = transform_.decompose()
        recon = transform_.affine(**params)

        theta = np.linspace(-np.pi, np.pi)[:, None]
        pts = np.c_[np.sin(theta), np.cos(theta), np.ones(len(theta))]

        poly = kwimage.Polygon.random()
        poly1 = poly.warp(transform_)
        poly2 = poly.warp(recon)
        pts_warp1 = transform_.matrix @ pts.T
        pts_warp2 = transform_.matrix @ pts.T

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> transform = Affine.random() @ Affine.scale(0.5) @ Affine.translate(-300)
        >>> from kwimage.im_cv2 import _cv2_warp_affine
        >>> warped = _cv2_warp_affine(image, transform, dsize='auto', antialias=1, interpolation='nearest')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(warped)
        >>> kwplot.show_if_requested()
    """
    from kwimage.transform import Affine
    import kwimage
    from kwimage._common import _coerce_warp_dsize_inputs

    if isinstance(image, np.ma.MaskedArray):
        mask = image.mask
        orig_mask_dtype = mask.dtype
        mask = mask.astype(np.uint8)
    else:
        mask = None

    is_masked = mask is not None

    transform = Affine.coerce(transform)
    flags = _coerce_interpolation(interpolation)
    borderMode, borderValue = _coerce_border_mode_value(
        border_mode, border_value, image)

    h, w = image.shape[0:2]
    input_dsize = (w, h)
    require_warped_info = large_warp_dim is not None
    dsize_inputs = _coerce_warp_dsize_inputs(
        dsize=dsize,
        input_dsize=input_dsize,
        transform=transform,
        require_warped_info=require_warped_info
    )
    affine_params = None
    max_dsize = dsize_inputs['max_dsize']
    new_origin = dsize_inputs['new_origin']
    transform_ = dsize_inputs['transform']
    dsize = dsize_inputs['dsize']
    info = {
        'transform': transform_,
        'dsize': dsize,
        'antialias_info': None,
    }

    _try_warp_tail_args = (large_warp_dim, dsize, max_dsize, new_origin, flags,
                           borderMode, borderValue)

    if origin_convention == 'corner':
        # cv2.warpAffine uses the integer-center convention, but by modifying
        # the transform we can implement the integer-corner behavior.
        offset1 = kwimage.Affine.translate((0.5, 0.5))
        offset2 = kwimage.Affine.translate((-0.5, -0.5))
        transform_ = offset2 @ transform_ @ offset1
    elif origin_convention == 'center':
        ...
    else:
        raise KeyError(origin_convention)

    try:
        if any(d == 0 for d in dsize) or any(d == 0 for d in image.shape[0:2]):
            # Handle case where the input image has no size or the destination
            # canvas has no size. In either case we just return empty data
            output_shape = (dsize[1], dsize[0]) + image.shape[2:]

            # Temporary workaround to prevent a hang due to incompatible shapes the
            # length of borderValue (when iterable) is limited to 4, but we can fix
            # this in some cases if all of the value are the same by converting to
            # a scalar.
            if not ub.iterable(borderValue):
                fill_value = borderValue
            elif len(image.shape[2:]) != len(borderValue):
                if ub.allsame(borderValue):
                    fill_value = borderValue[0]
                elif all(np.isnan(v) for v in borderValue):
                    fill_value = borderValue[0]
                else:
                    fill_value = borderValue
            else:
                fill_value = borderValue

            result = np.full(
                shape=output_shape, fill_value=fill_value, dtype=image.dtype)
            if is_masked:
                result_mask = np.full(
                    shape=output_shape, fill_value=False, dtype=mask.dtype)
        elif not antialias:
            result = _cv2_try_warp_affine(image, transform_, *_try_warp_tail_args)
            if is_masked:
                result_mask = _cv2_try_warp_affine(mask, transform_, *_try_warp_tail_args)
        else:
            # Decompose the affine matrix into its 6 core parameters
            if affine_params is None:
                affine_params = transform_.decompose()
            sx, sy = affine_params['scale']

            if sx > 1 and sy > 1:
                # No downsampling detected, no need to antialias
                result = _cv2_try_warp_affine(image, transform_, *_try_warp_tail_args)
                if is_masked:
                    result_mask = _cv2_try_warp_affine(mask, transform_, *_try_warp_tail_args)
            else:
                # At least one dimension is downsampled
                """
                Variations that could change in the future:

                    * In _gauss_params I'm not sure if we want to compute integer or
                        fractional "number of downsamples".

                    * The fudge factor bothers me, but seems necessary
                """

                # Compute the transform with all scaling removed
                noscale_warp = Affine.affine(**ub.dict_diff(affine_params, {'scale'}))

                # Execute part of the downscale with iterative pyramid downs
                downscaled, residual_sx, residual_sy = _cv2_prepare_downscale(
                    image, sx, sy)

                # Compute the transform from the downsampled image to the destination
                rest_warp = noscale_warp @ Affine.scale((residual_sx, residual_sy))

                info['antialias_info'] = {
                    'noscale_warp': noscale_warp,
                    'rest_warp': rest_warp,
                }

                result = _cv2_try_warp_affine(downscaled, rest_warp, *_try_warp_tail_args)

                if is_masked:
                    downscaled_mask, _, _ = _cv2_prepare_downscale(mask, sx, sy)
                    result_mask = _cv2_try_warp_affine(downscaled_mask, rest_warp, *_try_warp_tail_args)
    except Exception as ex:
        print('Error in warp_affine: ex = {}'.format(ub.urepr(ex, nl=1)))
        print(f'type(image) = {type(image)}')
        print(f'type(transform) = {type(transform)}')
        print(f'type(dsize) = {type(dsize)}')
        if isinstance(image, np.ndarray):
            print(f'image.shape={image.shape}')
        print(f'dsize={dsize}')
        print(f'interpolation={interpolation}')
        print(f'border_value={border_value}')
        print(f'border_mode={border_mode}')
        print(f'large_warp_dim={large_warp_dim}')
        print(f'return_info={return_info}')
        raise

    if is_masked:
        result_mask = result_mask.astype(orig_mask_dtype)
        result = np.ma.array(result, mask=result_mask)

    if return_info:
        return result, info
    else:
        return result


def _cv2_try_warp_affine(image, transform_, large_warp_dim, dsize, max_dsize,
                         new_origin, flags, borderMode, borderValue):
    """
    Helper for warp_affine
    """
    image = _cv2_imputation(image)

    if large_warp_dim == 'auto':
        # this is as close as we can get to actually discovering SHRT_MAX since
        # it's not introspectable through cv2.  numpy and cv2 could be pointing
        # to a different limits.h, but otherwise this is correct
        # https://stackoverflow.com/a/44123354
        SHRT_MAX = np.iinfo(np.short).max
        large_warp_dim = SHRT_MAX

    max_dim = max(image.shape[0:2])
    if large_warp_dim is None or max_dim < large_warp_dim:
        try:
            M = np.asarray(transform_)
            return cv2.warpAffine(image, M[0:2], dsize=dsize, flags=flags,
                                  borderMode=borderMode,
                                  borderValue=borderValue)
        except cv2.error as e:
            if e.err == 'dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX':
                print(
                    'Image too large for warp_affine. Bypass this error by setting '
                    'kwimage.warp_affine(large_warp_dim="auto")')
                raise e
            else:
                raise

    else:
        # make these pieces as large as possible for efficiency
        pieces_per_dim = 1 + max_dim // (large_warp_dim - 1)
        return _cv2_large_warp_affine(image, transform_, dsize, max_dsize,
                                      new_origin, flags, borderMode,
                                      borderValue, pieces_per_dim)


def _cv2_imputation(image):
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    return image


def _cv2_large_warp_affine(image, transform_, dsize, max_dsize, new_origin,
                           flags, borderMode, borderValue, pieces_per_dim):
    """
    Split an image into pieces smaller than cv2's limit, perform cv2.warpAffine on each piece,
    and stitch them back together with minimal artifacts.

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # xdoctest: +REQUIRES(--large_memory)
        >>> import kwimage
        >>> img = np.random.randint(255, size=(32767, 32767), dtype=np.uint8)
        >>> aff = kwimage.Affine.random()
        >>> import cv2
        >>> #
        >>> # without this function
        >>> try:
        >>>     res = kwimage.warp_affine(img, aff, large_warp_dim=None)
        >>> except cv2.error as e:
        >>>     pass
        >>> #
        >>> # with this function
        >>> res = kwimage.warp_affine(img, aff, large_warp_dim='auto')
        >>> assert res.shape == img.shape
        >>> assert res.dtype == img.dtype

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import cv2
        >>> image = kwimage.grab_test_image('astro')
        >>> # Use wrapper function
        >>> transform = kwimage.Affine.coerce(
        >>>     {'offset': (136.3946757082253, 0.0),
        >>>      'scale': (1.7740542832875767, 1.0314621286400032),
        >>>      'theta': 0.2612311452107956,
        >>>      'type': 'affine'})
        >>> res, info = kwimage.warp_affine(
        >>>     image, transform, dsize='content', return_info=True,
        >>>     large_warp_dim=128)
        >>> # Explicit args for this function
        >>> transform = info['transform']
        >>> new_origin = np.array((0, 0))
        >>> max_dsize = (1015, 745)
        >>> dsize = max_dsize
        >>> res2 = _cv2_large_warp_affine(image, transform, dsize, max_dsize, new_origin,
        >>>                   flags=cv2.INTER_LINEAR, borderMode=None,
        >>>                   borderValue=None, pieces_per_dim=2)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(res, pnum=(1, 2, 1))
        >>> kwplot.imshow(res2, pnum=(1, 2, 2))
    """
    from kwimage import Affine, Boxes
    import cv2
    import itertools as it

    def _split_2d(arr):
        # provide indexes to view arr in 2d blocks like 2 uses of
        # np.array_split() but provides the indexes, not the data
        h, w = arr.shape[0:2]
        xs, ys = zip(
            *np.linspace([0, 0], [w, h], num=pieces_per_dim + 1, dtype=int))
        ixs = [
            xx + yy for xx, yy in it.product(zip(xs[:-1], xs[1:]),
                                             zip(ys[:-1], ys[1:]))
        ]
        return Boxes(ixs, 'xxyy')  # could use to_slices() for portability

    # do the warp with dsize='max' to make sure we don't lose any pieces
    # then crop it down later if needed
    max_transform = Affine.translate(-new_origin) @ transform_

    # create an empty canvas to fill with the warped pieces
    # this is a masked version of kwarray.Stitcher
    # it mitigates but does not remove piece edge artifacts
    result = np.zeros((*max_dsize[::-1], *image.shape[2:]), dtype=np.float32)
    weight = np.zeros((*max_dsize[::-1], *image.shape[2:]), dtype=np.uint8)

    # compute each piece with dsize=max and apply it to the canvas
    # Note that this will unavoidably produce artifacts along the "seams"
    # because interpolation is not performed across them.
    for img_piece in _split_2d(image):

        # restore extra dim from looping before converting to slice
        img_piece = Boxes([img_piece.data], img_piece.format)
        img_piece_ix = img_piece.to_slices()[0]

        piece_wh = img_piece.to_xywh().data[0, 2:4]
        warped_origin = img_piece.warp(max_transform).to_xywh().data[0, 0:2]

        centered_bb = Boxes(
            np.array([[0, 0, *piece_wh]]), 'xywh').warp(max_transform)
        centered_origin = centered_bb.data[0, 0:2]

        piece_centered_matrix = (
            Affine.translate(-centered_origin) @ max_transform).matrix
        warped_bbox = img_piece.warp(
            piece_centered_matrix).to_ltrb().quantize()

        warped_dsize = tuple(map(int, warped_bbox.to_xywh().data[0, 2:4]))
        # do the quantizing manually here to avoid changing dsize
        # TODO add check for going OOB of result's shape and replace floor w/
        # round this produces shifts of up to 1 px
        result_bbox = Boxes(
            np.array([[*np.floor(warped_origin), *warped_dsize]]).astype(int),
            'xywh')
        result_ix = result_bbox.to_slices()[0]

        warped_piece = cv2.warpAffine(image[img_piece_ix],
                                      piece_centered_matrix[0:2],
                                      dsize=warped_dsize,
                                      flags=flags,
                                      borderMode=borderMode,
                                      borderValue=borderValue)

        weight_piece = cv2.warpAffine(np.ones_like(image[img_piece_ix]),
                                      piece_centered_matrix[0:2],
                                      dsize=warped_dsize,
                                      flags=flags,
                                      borderMode=borderMode,
                                      borderValue=borderValue)

        result[result_ix] += warped_piece
        weight[result_ix] += weight_piece

    result = (result / np.where(weight != 0, weight, 1)).astype(image.dtype)

    # crop and pad the canvas to the desired size
    result = imcrop(result,
                    dsize,
                    origin=np.round(-new_origin).astype(int),
                    border_value=borderValue)

    return result


def _prepare_scale_residual(sx, sy, fudge=0):
    """
    Helper to decompose a scale factor into pyramid downscales plus a residual
    scale factor.
    """
    max_scale = max(sx, sy)
    ideal_num_downs = int(np.log2(1 / max_scale))
    num_downs = max(ideal_num_downs - fudge, 0)
    pyr_scale = 1 / (2 ** num_downs)
    residual_sx = sx / pyr_scale
    residual_sy = sy / pyr_scale
    return num_downs, residual_sx, residual_sy


def _cv2_prepare_downscale(image, sx, sy):
    """
    Does a partial downscale with antialiasing and prepares for a final
    downsampling. Only downscales by factors of 2, any residual scaling to
    be done is returned.

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> s = 523
        >>> image = np.random.rand(s, s)
        >>> sx = sy = 1 / 11
        >>> downsampled, rx, ry = _cv2_prepare_downscale(image, sx, sy)
    """
    # The "fudge" factor limits the number of downsampled pyramid
    # operations. A bigger fudge factor means means that the final
    # gaussian kernel for the antialiasing operation will be bigger.
    # It essentially says that at most "fudge" downsampling ops will
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

    num_downs, residual_sx, residual_sy = _prepare_scale_residual(sx, sy, fudge)
    # Downsample iteratively with antialiasing
    downscaled = _pyrDownK(image, num_downs)

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
        print('result = {}'.format(ub.urepr(result, nl=1)))
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


"""
items = {k.split('_')[1].lower(): 'cv2.' + k for k in dir(cv2) if k.startswith('MORPH_')}
items = ub.sorted_vals(items, key=lambda x: eval(x, {'cv2': cv2}))
print('_CV2_MORPH_MODES = {}'.format(ub.urepr(items, nl=1, sv=1, align=':')))
"""
_CV2_STRUCT_ELEMENTS = {
    'rect'    : cv2.MORPH_RECT,
    'cross'   : cv2.MORPH_CROSS,
    'ellipse' : cv2.MORPH_ELLIPSE,
}


_CV2_MORPH_MODES = {
    'erode'   : cv2.MORPH_ERODE,
    'dilate'  : cv2.MORPH_DILATE,
    'open'    : cv2.MORPH_OPEN,
    'close'   : cv2.MORPH_CLOSE,
    'gradient': cv2.MORPH_GRADIENT,
    'tophat'  : cv2.MORPH_TOPHAT,
    'blackhat': cv2.MORPH_BLACKHAT,
    'hitmiss' : cv2.MORPH_HITMISS,
}


@lru_cache(128)
def _morph_kernel_core(w, h, element):
    if w == 0 or h == 0:
        return np.empty((0, 0), dtype=np.uint8)
    struct_shape = _CV2_STRUCT_ELEMENTS.get(element, element)
    element = cv2.getStructuringElement(struct_shape, (h, w))
    return element


def _morph_kernel(kernel, element='rect'):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> from kwimage.im_cv2 import _morph_kernel
        >>> from kwimage.im_cv2 import _CV2_MORPH_MODES  # NOQA
        >>> from kwimage.im_cv2 import _CV2_STRUCT_ELEMENTS  # NOQA
        >>> kernel = 20
        >>> results = {}
        >>> for element in _CV2_STRUCT_ELEMENTS.keys():
        ...     results[element] = _morph_kernel(kernel, element)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result.astype(np.float32), pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()
    """
    if isinstance(kernel, np.ndarray) and len(kernel.shape) == 2:
        # Kernel is a custom element
        return kernel
    else:
        if isinstance(kernel, int):
            w = h = kernel
        else:
            w, h = kernel
        return _morph_kernel_core(w, h, element)


def morphology(data, mode, kernel=5, element='rect', iterations=1,
               border_mode='constant', border_value=0):
    """
    Executes a morphological operation.

    Args:
        input (ndarray[dtype=uint8 | float64]): data
            (note if mode is hitmiss data must be uint8)

        mode (str) : morphology mode, can be one of:
            'erode', 'dilate', 'open', 'close', 'gradient', 'tophat',
            'blackhat', or 'hitmiss'.

        kernel (ndarray | int | Tuple[int, int]):
            size of the morphology kernel (w, h) to be constructed according to
            "element".  If the kernel size is 0, this function returns a copy
            of the data.  Can also be a 2D array which is a custom structuring
            element.  In this case "element" is ignored.

        element (str):
            structural element, can be 'rect', 'cross', or 'ellipse'.

        iterations (int):
            numer of times to repeat the operation

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float]):
            Used as the fill value if border_mode is constant.
            Otherwise this is ignored.

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> import kwimage
        >>> #image = kwimage.grab_test_image(dsize=(380, 380))
        >>> image = kwimage.Mask.demo().data * 255
        >>> basis = {
        >>>     'mode': ['dilate'],
        >>>     'kernel': [5, (3, 7)],
        >>>     'element': ['rect', 'cross', 'ellipse'],
        >>>     #'mode': ['dilate', 'erode'],
        >>> }
        >>> grid = list(ub.named_product(basis))
        >>> grid += [{'mode': 'dilate', 'kernel': 0, 'element': 'rect', }]
        >>> grid += [{'mode': 'dilate', 'kernel': 'random', 'element': 'custom'}]
        >>> results = {}
        >>> for params in grid:
        ...     key = ub.urepr(params, compact=1, si=0, nl=1)
        ...     if params['kernel'] == 'random':
        ...         params['kernel'] = np.random.rand(5, 5)
        ...     results[key] = morphology(image, **params)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> to_stack = []
        >>> canvas = image
        >>> canvas = kwimage.imresize(canvas, dsize=(380, 380), interpolation='nearest')
        >>> canvas = kwimage.draw_header_text(canvas, 'input', color='kitware_green')
        >>> to_stack.append(canvas)
        >>> for key, result in results.items():
        >>>     canvas = result
        >>>     canvas = kwimage.imresize(canvas, dsize=(380, 380), interpolation='nearest')
        >>>     canvas = kwimage.draw_header_text(canvas, key, color='kitware_green')
        >>>     to_stack.append(canvas)
        >>> canvas = kwimage.stack_images_grid(to_stack, pad=10, bg_value='kitware_blue')
        >>> canvas = kwimage.draw_header_text(canvas, '--- kwimage.morphology demo ---', color='kitware_green')
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> from kwimage.im_cv2 import _CV2_MORPH_MODES  # NOQA
        >>> from kwimage.im_cv2 import _CV2_STRUCT_ELEMENTS  # NOQA
        >>> #shape = (32, 32)
        >>> shape = (64, 64)
        >>> data = (np.random.rand(*shape) > 0.5).astype(np.uint8)
        >>> import kwimage
        >>> data = kwimage.gaussian_patch(shape)
        >>> data = data / data.max()
        >>> data = kwimage.ensure_uint255(data)
        >>> results = {}
        >>> kernel = 5
        >>> for mode in _CV2_MORPH_MODES.keys():
        ...     for element in _CV2_STRUCT_ELEMENTS.keys():
        ...         results[f'{mode}+{element}'] = morphology(data, mode, kernel=kernel, element=element, iterations=2)
        >>> results['raw'] = data
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nCols=3, nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()

    References:
        https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """
    import cv2
    kernel = _morph_kernel(kernel, element)
    if kernel.size == 0:
        return data.copy()

    data, final_dtype = _cv2_input_fixer(data)

    borderMode = _coerce_border_mode(border_mode)
    borderValue = _coerce_border_value(border_value, image=data)
    if isinstance(mode, str):
        morph_mode = _CV2_MORPH_MODES[mode]
    elif isinstance(mode, int):
        morph_mode = mode
    else:
        raise TypeError(type(mode))

    data = _cv2_imputation(data)
    new = cv2.morphologyEx(
        data, op=morph_mode, kernel=kernel, iterations=iterations,
        borderValue=borderValue, borderType=borderMode
    )
    if final_dtype is not None:
        new = new.astype(final_dtype)
    return new


def connected_components(image, connectivity=8, ltype=np.int32,
                         with_stats=True, algo='default'):
    """
    Find connected components in a binary image.

    Wrapper around :func:`cv2.connectedComponentsWithStats`.

    Args:
        image (ndarray): a binary uint8 image. Zeros denote the background, and
            non-zeros numbers are foreground regions that will be partitioned
            into connected components.

        connectivity (int): either 4 or 8

        ltype (numpy.dtype | str | int):
            The dtype for the output label array.
            Can be either 'int32' or 'uint16', and this can be specified as a
            cv2 code or a numpy dtype.

        algo (str):
            The underlying algorithm to use. See [Cv2CCAlgos]_ for details.
            Options are spaghetti, sauf, bbdt. (default is spaghetti)

    Returns:
        Tuple[ndarray, dict]:
            The label array and an information dictionary

    TODO:
        Document the details of which type of coordinates we are using.
        I.e. are pixels points or areas? (I think this uses the points
        convention?)

    Note:
        opencv 4.5.5 will segfault if connectivity=4 See: [CvIssue21366]_.

    Note:
        Based on information in [SO35854197]_.

    References:
        .. [SO35854197] https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
        .. [Cv2CCAlgos] https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga5ed7784614678adccb699c70fb841075
        .. [CvIssue21366] https://github.com/opencv/opencv/issues/21366

    CommandLine:
        xdoctest -m kwimage.im_cv2 connected_components:0 --show

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> mask = kwimage.Mask.demo()
        >>> image = mask.data
        >>> labels, info = connected_components(image)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas0 = kwimage.atleast_3channels(mask.data * 255)
        >>> canvas2 = canvas0.copy()
        >>> canvas3 = canvas0.copy()
        >>> boxes = info['label_boxes']
        >>> centroids = info['label_centroids']
        >>> label_colors = kwimage.Color.distinct(info['num_labels'])
        >>> index_to_color = np.array([kwimage.Color('black').as01()] + label_colors)
        >>> canvas2 = centroids.draw_on(canvas2, color=label_colors, radius=None)
        >>> boxes.draw_on(canvas3, color=label_colors, thickness=1)
        >>> legend = kwplot.make_legend_img(ub.dzip(range(len(index_to_color)), index_to_color))
        >>> colored_label_img = index_to_color[labels]
        >>> canvas1 = kwimage.stack_images([colored_label_img, legend], axis=1, resize='smaller')
        >>> kwplot.imshow(canvas0, pnum=(1, 4, 1), title='input image')
        >>> kwplot.imshow(canvas1, pnum=(1, 4, 2), title='label image (colored w legend)')
        >>> kwplot.imshow(canvas2, pnum=(1, 4, 3), title='component centroids')
        >>> kwplot.imshow(canvas3, pnum=(1, 4, 4), title='component bounding boxes')
    """

    if isinstance(ltype, str):
        if ltype in {'int32', 'CV2_32S'}:
            ltype = np.int32
        elif ltype in {'uint16', 'CV_16U'}:
            ltype = np.uint16
    if ltype is np.int32:
        ltype = cv2.CV_32S
    elif ltype is np.int16:
        ltype = cv2.CV_16U
    if not isinstance(ltype, int):
        raise TypeError('type(ltype) = {}'.format(type(ltype)))

    # It seems very easy for a segfault to happen here.
    image = np.ascontiguousarray(image)
    if image.dtype.kind != 'u' or image.dtype.itemsize != 1:
        raise ValueError('input image must be a uint8')

    if algo != 'default':
        if algo in {'spaghetti', 'bolelli'}:
            ccltype = cv2.CCL_SPAGHETTI
        elif algo in {'sauf', 'wu'}:
            ccltype = cv2.CCL_SAUF
        elif algo in {'bbdt', 'grana'}:
            ccltype = cv2.CCL_BBDT
        else:
            raise KeyError(algo)

        if with_stats:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
                image, connectivity=connectivity, ccltype=ccltype, ltype=ltype)
        else:
            num_labels, labels = cv2.connectedComponentsWithAlgorithm(
                image, connectivity=connectivity, ccltype=ccltype, ltype=ltype)
    else:
        if with_stats:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                image, connectivity=connectivity, ltype=ltype)
        else:
            num_labels, labels = cv2.connectedComponents(
                image, connectivity=connectivity, ltype=ltype)

    info = {
        'num_labels': num_labels,
    }

    if with_stats:
        # Transform stats into a kwimage boxes object for each label
        import kwimage
        info['label_boxes'] = kwimage.Boxes(stats[:, [
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]], 'ltwh')
        info['label_areas'] = stats[:, cv2.CC_STAT_AREA]
        info['label_centroids'] = kwimage.Points(xy=centroids)

    return labels, info


def _cv2_warp_projective(image, transform, dsize=None, antialias=False,
                         interpolation='linear', border_mode=None,
                         border_value=0, large_warp_dim=None,
                         origin_convention='center', return_info=False):
    """
    Applies an projective transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image as a numpy array.
            Note: this is passed directly to cv2, so it is best to ensure that
            it is contiguous and using a dtype that cv2 can handle.

        transform (ndarray | dict | kwimage.Projective): a coercable projective matrix.
            See :class:`kwimage.Projective` for details on what can be coerced.

        dsize (Tuple[int, int] | None | str):
            A integer width and height tuple of the resulting "canvas" image.
            If None, then the input image size is used.

            If specified as a string, dsize is computed based on the given
            heuristic.

            If 'positive' (or 'auto'), dsize is computed such that the positive
            coordinates of the warped image will fit in the new canvas. In this
            case, any pixel that maps to a negative coordinate will be clipped.
            This has the property that the input transformation is not
            modified.

            If 'content' (or 'max'), the transform is modified with an extra
            translation such that both the positive and negative coordinates of
            the warped image will fit in the new canvas.

        antialias (bool)
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur. Defaults to False

        interpolation (str | int):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area. Defaults to "linear".

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float]):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.

        large_warp_dim (int | None | str):
            If specified, perform the warp piecewise in chunks of the specified
            size. If "auto", it is set to the maximum "short" value in numpy.
            This works around a limitation of cv2.warpAffine, which must have
            image dimensions < SHRT_MAX (=32767 in version 4.5.3)

        origin_convention (str):
            Controls the interpretation of the underlying raster.
            Can be "center" (default opencv behavior), or "corner" (matches
            torchvision / detectron2 behavior).
            If "center", then center of the top left pixel is at (0, 0), and
            the top left corner is at (-0.5, -0.5).
            If "center", then center of the top left pixel is at (0.5, 0.5), and
            the top left corner is at (0, 0).
            Currently defaults to "center", but in the future we may change the
            default to "corner".  For more info see [WhereArePixels]_.

        return_info (bool):
            if True, returns information about the operation. In the case
            where dsize="content", this includes the modified transformation.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            the warped image, or if return info is True, the warped image and
            the info dictionary.
    """
    # TODO: consolidate with warp_affine and warp_image logic
    import kwimage
    from kwimage._common import _coerce_warp_dsize_inputs

    if isinstance(image, np.ma.MaskedArray):
        mask = image.mask
        orig_mask_dtype = mask.dtype
        mask = mask.astype(np.uint8)
    else:
        mask = None

    is_masked = mask is not None

    transform = kwimage.Projective.coerce(transform)
    flags = _coerce_interpolation(interpolation)
    borderMode, borderValue = _coerce_border_mode_value(
        border_mode, border_value, image)

    h, w = image.shape[0:2]
    input_dsize = (w, h)
    require_warped_info = large_warp_dim is not None
    dsize_inputs = _coerce_warp_dsize_inputs(
        dsize=dsize,
        input_dsize=input_dsize,
        transform=transform,
        require_warped_info=require_warped_info
    )
    max_dsize = dsize_inputs['max_dsize']
    new_origin = dsize_inputs['new_origin']
    transform_ = dsize_inputs['transform']
    dsize = dsize_inputs['dsize']
    info = {
        'transform': transform_,
        'dsize': dsize,
        'antialias_info': None,
    }

    if origin_convention == 'corner':
        # cv2.warpAffine uses the integer-center convention, but by modifying
        # the transform we can implement the integer-corner behavior.
        offset1 = kwimage.Affine.translate((0.5, 0.5))
        offset2 = kwimage.Affine.translate((-0.5, -0.5))
        transform_ = offset2 @ transform_ @ offset1
    elif origin_convention == 'center':
        ...
    else:
        raise KeyError(origin_convention)

    _try_warp_tail_args = (large_warp_dim, dsize, max_dsize, new_origin, flags,
                           borderMode, borderValue)

    if any(d == 0 for d in dsize) or any(d == 0 for d in image.shape[0:2]):
        # Handle case where the input image has no size or the destination
        # canvas has no size. In either case we just return empty data
        output_shape = (dsize[1], dsize[0]) + image.shape[2:]
        result = np.full(
            shape=output_shape, fill_value=borderValue, dtype=image.dtype)
        if is_masked:
            result_mask = np.full(
                shape=output_shape, fill_value=False, dtype=mask.dtype)
    elif not antialias:
        result = _cv2_try_warp_projective(image, transform_, *_try_warp_tail_args)
        if is_masked:
            result_mask = _cv2_try_warp_projective(mask, transform_, *_try_warp_tail_args)
    else:
        # TODO: Fix cases

        # Decompose the projective matrix into its 6 core parameters
        params = transform_.decompose()
        sx, sy = params['scale']

        if sx > 1 and sy > 1:
            # No downsampling detected, no need to antialias
            result = _cv2_try_warp_projective(image, transform_, *_try_warp_tail_args)
            if is_masked:
                result_mask = _cv2_try_warp_projective(mask, transform_, *_try_warp_tail_args)
        else:
            # We can't actually do this in the projective case
            raise NotImplementedError('cannot antialias in projective case yet')

    if is_masked:
        result_mask = result_mask.astype(orig_mask_dtype)
        result = np.ma.array(result, mask=result_mask)

    if return_info:
        return result, info
    else:
        return result


def _cv2_try_warp_projective(image, transform_, large_warp_dim, dsize,
                             max_dsize, new_origin, flags, borderMode,
                             borderValue):
    """
    Helper for warp_projective
    """
    image = _cv2_imputation(image)

    if large_warp_dim == 'auto':
        # this is as close as we can get to actually discovering SHRT_MAX since
        # it's not introspectable through cv2.  numpy and cv2 could be pointing
        # to a different limits.h, but otherwise this is correct
        # https://stackoverflow.com/a/44123354
        SHRT_MAX = np.iinfo(np.short).max
        large_warp_dim = SHRT_MAX

    max_dim = max(image.shape[0:2])
    if large_warp_dim is None or max_dim < large_warp_dim:
        try:
            M = np.asarray(transform_)
            return cv2.warpPerspective(image, M, dsize=dsize, flags=flags,
                                       borderMode=borderMode,
                                       borderValue=borderValue)
        except cv2.error as e:
            if e.err == 'dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX':
                print(
                    'Image too large for warp_projective. Bypass this error by setting '
                    'kwimage.warp_projective(large_warp_dim="auto")')
                raise e
            else:
                raise

    else:
        raise NotImplementedError
        # # make these pieces as large as possible for efficiency
        # pieces_per_dim = 1 + max_dim // (large_warp_dim - 1)
        # return _large_warp_projective(image, transform_, dsize, max_dsize,
        #                    new_origin, flags, borderMode,
        #                    borderValue, pieces_per_dim)
