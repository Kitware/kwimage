"""
TODO:
    - [ ] Replace internal padded slice with kwarray.padded_slice
"""
import ubelt as ub
import numpy as np
import kwarray

try:
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion

try:
    import torch
    import torch.nn.functional as F
    TORCH_GRID_SAMPLE_HAS_ALIGN = LooseVersion(torch.__version__) >= LooseVersion('1.3.0')
except Exception:
    torch = None
    F = None
    TORCH_GRID_SAMPLE_HAS_ALIGN = None


def _coordinate_grid(dims, align_corners=False):
    """
    Creates a homogenous coordinate system.

    Args:
        dims (Tuple[int, ...]): height / width or depth / height / width

        align_corners (bool):
            returns a grid where the left and right corners assigned to the
            extreme values and intermediate values are interpolated.

    Returns:
        Tensor:
            with shape=(3, *DIMS)

    References:
        https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> _coordinate_grid((2, 2))
        tensor([[[0., 1.],
                 [0., 1.]],
                [[0., 0.],
                 [1., 1.]],
                [[1., 1.],
                 [1., 1.]]])
        >>> _coordinate_grid((2, 2, 2))
        >>> _coordinate_grid((2, 2), align_corners=True)
        tensor([[[0., 2.],
                 [0., 2.]],
                [[0., 0.],
                 [2., 2.]],
                [[1., 1.],
                 [1., 1.]]])

    """
    if align_corners:
        def _corner_grid(d):
            return torch.linspace(0, d, d)
        _grid_fn = _corner_grid
    else:
        def _disc_grid(d):
            return torch.arange(0, d)
        _grid_fn = _disc_grid

    if len(dims) == 2:
        h, w = dims
        h_range = _grid_fn(h).view(h, 1).expand(h, w).float()  # [H, W]
        w_range = _grid_fn(w).view(1, w).expand(h, w).float()  # [H, W]
        ones = torch.ones(h, w)
        pixel_coords = torch.stack((w_range, h_range, ones), dim=0)  # [3, H, W]
    elif len(dims) == 3:
        d, h, w = dims
        d_range = _grid_fn(d).view(d, 1, 1).expand(d, h, w).float()  # [D, H, W]
        h_range = _grid_fn(h).view(1, h, 1).expand(d, h, w).float()  # [D, H, W]
        w_range = _grid_fn(w).view(1, 1, w).expand(d, h, w).float()  # [D, H, W]
        ones = torch.ones(d, h, w)
        pixel_coords = torch.stack((w_range, h_range, d_range, ones), dim=0)  # [4, D, H, W]
        pass
    else:
        raise NotImplementedError('Can only work with 2d and 3d dims')
    return pixel_coords


def warp_tensor(inputs, mat, output_dims, mode='bilinear',
                padding_mode='zeros', isinv=False, ishomog=None,
                align_corners=False, new_mode=False):
    r"""
    A pytorch implementation of warp affine that works similarly to
    :func:`cv2.warpAffine` and :func:`cv2.warpPerspective`.

    It is possible to use 3x3 transforms to warp 2D image data.  It is also
    possible to use 4x4 transforms to warp 3D volumetric data.

    Args:
        inputs (Tensor): tensor to warp.
            Up to 3 (determined by output_dims) of the trailing space-time
            dimensions are warped. Best practice is to use inputs with the
            shape in [B, C, *DIMS].

        mat (Tensor):
            either a 3x3 / 4x4 single transformation matrix to apply to all
            inputs or Bx3x3 or Bx4x4 tensor that specifies a transformation
            matrix for each batch item.

        output_dims (Tuple[int, ...]):
            The output space-time dimensions. This can either be in the form
            (W,), (H, W), or (D, H, W).

        mode (str):
            Can be bilinear or nearest.
            See `torch.nn.functional.grid_sample`

        padding_mode (str):
            Can be zeros, border, or reflection.
            See `torch.nn.functional.grid_sample`.

        isinv (bool):
            Set to true if `mat` is the inverse transform

        ishomog (bool):
            Set to True if the matrix is non-affine

        align_corners (bool):
            Note the default of False does not work correctly with grid_sample
            in torch <= 1.2, but using align_corners=True isnt typically what
            you want either. We will be stuck with buggy functionality until
            torch 1.3 is released.

            However, using align_corners=0 does seem to reasonably correspond
            with opencv behavior.

    Returns:
        Tensor: warped tensor

    Note:
        Also, it may be possible to speed up the code with `F.affine_grid`

        KNOWN ISSUE: There appears to some difference with cv2.warpAffine when
            rotation or shear are non-zero. I'm not sure what the cause is.
            It may just be floating point issues, but Im' not sure.

        See issues in [TorchAffineTransform]_ and [TorchIssue15386]_.

    TODO:
        - [ ] FIXME: see example in Mask.scale where this algo breaks when the matrix is `2x3`
        - [ ] Make this algo work when matrix ix 2x2

    References:
        .. [TorchAffineTransform] https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
        .. [TorchIssue15386] https://github.com/pytorch/pytorch/issues/15386

    Example:
        >>> # Create a relatively simple affine matrix
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import skimage
        >>> mat = torch.FloatTensor(skimage.transform.AffineTransform(
        >>>     translation=[1, -1], scale=[.532, 2],
        >>>     rotation=0, shear=0,
        >>> ).params)
        >>> # Create inputs and an output dimension
        >>> input_shape = [1, 1, 4, 5]
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> output_dims = (11, 7)
        >>> # Warp with our code
        >>> result1 = warp_tensor(inputs, mat, output_dims=output_dims, align_corners=0)
        >>> print('result1 =\n{}'.format(ub.repr2(result1.cpu().numpy()[0, 0], precision=2)))
        >>> # Warp with opencv
        >>> import cv2
        >>> cv2_M = mat.cpu().numpy()[0:2]
        >>> src = inputs[0, 0].cpu().numpy()
        >>> dsize = tuple(output_dims[::-1])
        >>> result2 = cv2.warpAffine(src, cv2_M, dsize=dsize, flags=cv2.INTER_LINEAR)
        >>> print('result2 =\n{}'.format(ub.repr2(result2, precision=2)))
        >>> # Ensure the results are the same (up to floating point errors)
        >>> assert np.all(np.isclose(result1[0, 0].cpu().numpy(), result2, atol=1e-2, rtol=1e-2))

    Example:
        >>> # Create a relatively simple affine matrix
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import skimage
        >>> mat = torch.FloatTensor(skimage.transform.AffineTransform(
        >>>     rotation=0.01, shear=0.1).params)
        >>> # Create inputs and an output dimension
        >>> input_shape = [1, 1, 4, 5]
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> output_dims = (11, 7)
        >>> # Warp with our code
        >>> result1 = warp_tensor(inputs, mat, output_dims=output_dims)
        >>> print('result1 =\n{}'.format(ub.repr2(result1.cpu().numpy()[0, 0], precision=2, supress_small=True)))
        >>> print('result1.shape = {}'.format(result1.shape))
        >>> # Warp with opencv
        >>> import cv2
        >>> cv2_M = mat.cpu().numpy()[0:2]
        >>> src = inputs[0, 0].cpu().numpy()
        >>> dsize = tuple(output_dims[::-1])
        >>> result2 = cv2.warpAffine(src, cv2_M, dsize=dsize, flags=cv2.INTER_LINEAR)
        >>> print('result2 =\n{}'.format(ub.repr2(result2, precision=2)))
        >>> print('result2.shape = {}'.format(result2.shape))
        >>> # Ensure the results are the same (up to floating point errors)
        >>> # NOTE: The floating point errors seem to be significant for rotation / shear
        >>> assert np.all(np.isclose(result1[0, 0].cpu().numpy(), result2, atol=1, rtol=1e-2))

    Example:
        >>> # Create a random affine matrix
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import skimage
        >>> rng = np.random.RandomState(0)
        >>> mat = torch.FloatTensor(skimage.transform.AffineTransform(
        >>>     translation=rng.randn(2), scale=1 + rng.randn(2),
        >>>     rotation=rng.randn() / 10., shear=rng.randn() / 10.,
        >>> ).params)
        >>> # Create inputs and an output dimension
        >>> input_shape = [1, 1, 5, 7]
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> output_dims = (3, 11)
        >>> # Warp with our code
        >>> result1 = warp_tensor(inputs, mat, output_dims=output_dims, align_corners=0)
        >>> print('result1 =\n{}'.format(ub.repr2(result1.cpu().numpy()[0, 0], precision=2)))
        >>> # Warp with opencv
        >>> import cv2
        >>> cv2_M = mat.cpu().numpy()[0:2]
        >>> src = inputs[0, 0].cpu().numpy()
        >>> dsize = tuple(output_dims[::-1])
        >>> result2 = cv2.warpAffine(src, cv2_M, dsize=dsize, flags=cv2.INTER_LINEAR)
        >>> print('result2 =\n{}'.format(ub.repr2(result2, precision=2)))
        >>> # Ensure the results are the same (up to floating point errors)
        >>> # NOTE: The errors seem to be significant for rotation / shear
        >>> assert np.all(np.isclose(result1[0, 0].cpu().numpy(), result2, atol=1, rtol=1e-2))

    Example:
        >>> # Test 3D warping with identity
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> mat = torch.eye(4)
        >>> input_dims = [2, 3, 3]
        >>> output_dims = (2, 3, 3)
        >>> input_shape = [1, 1] + input_dims
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> result = warp_tensor(inputs, mat, output_dims=output_dims)
        >>> print('result =\n{}'.format(ub.repr2(result.cpu().numpy()[0, 0], precision=2)))
        >>> assert torch.all(inputs == result)

    Example:
        >>> # Test 3D warping with scaling
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> mat = torch.FloatTensor([
        >>>     [0.8,   0,   0, 0],
        >>>     [  0, 1.0,   0, 0],
        >>>     [  0,   0, 1.2, 0],
        >>>     [  0,   0,   0, 1],
        >>> ])
        >>> input_dims = [2, 3, 3]
        >>> output_dims = (2, 3, 3)
        >>> input_shape = [1, 1] + input_dims
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> result = warp_tensor(inputs, mat, output_dims=output_dims, align_corners=0)
        >>> print('result =\n{}'.format(ub.repr2(result.cpu().numpy()[0, 0], precision=2)))
        result =
        np.array([[[ 0.  ,  1.25,  1.  ],
                   [ 3.  ,  4.25,  2.5 ],
                   [ 6.  ,  7.25,  4.  ]],
                  ...
                  [[ 7.5 ,  8.75,  4.75],
                   [10.5 , 11.75,  6.25],
                   [13.5 , 14.75,  7.75]]], dtype=np.float32)

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> mat = torch.eye(3)
        >>> input_dims = [5, 7]
        >>> output_dims = (11, 7)
        >>> for n_prefix_dims in [0, 1, 2, 3, 4, 5]:
        >>>      input_shape = [2] * n_prefix_dims + input_dims
        >>>      inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>>      result = warp_tensor(inputs, mat, output_dims=output_dims)
        >>>      #print('result =\n{}'.format(ub.repr2(result.cpu().numpy(), precision=2)))
        >>>      print(result.shape)

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> mat = torch.eye(4)
        >>> input_dims = [5, 5, 5]
        >>> output_dims = (6, 6, 6)
        >>> for n_prefix_dims in [0, 1, 2, 3, 4, 5]:
        >>>      input_shape = [2] * n_prefix_dims + input_dims
        >>>      inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>>      result = warp_tensor(inputs, mat, output_dims=output_dims)
        >>>      #print('result =\n{}'.format(ub.repr2(result.cpu().numpy(), precision=2)))
        >>>      print(result.shape)

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(warp_tensor))
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import cv2
        >>> inputs = torch.arange(9).view(1, 1, 3, 3).float() + 2
        >>> input_dims = inputs.shape[2:]
        >>> #output_dims = (6, 6)
        >>> def fmt(a):
        >>>     return ub.repr2(a.numpy(), precision=2)
        >>> s = 2.5
        >>> output_dims = tuple(np.round((np.array(input_dims) * s)).astype(int).tolist())
        >>> mat = torch.FloatTensor([[s, 0, 0], [0, s, 0], [0, 0, 1]])
        >>> inv = mat.inverse()
        >>> warp_tensor(inputs, mat, output_dims)
        >>> print('## INPUTS')
        >>> print(fmt(inputs))
        >>> print('\nalign_corners=True')
        >>> print('----')
        >>> print('## warp_tensor, align_corners=True')
        >>> print(fmt(warp_tensor(inputs, inv, output_dims, isinv=True, align_corners=True)))
        >>> print('## interpolate, align_corners=True')
        >>> print(fmt(F.interpolate(inputs, output_dims, mode='bilinear', align_corners=True)))
        >>> print('\nalign_corners=False')
        >>> print('----')
        >>> print('## warp_tensor, align_corners=False, new_mode=False')
        >>> print(fmt(warp_tensor(inputs, inv, output_dims, isinv=True, align_corners=False)))
        >>> print('## warp_tensor, align_corners=False, new_mode=True')
        >>> print(fmt(warp_tensor(inputs, inv, output_dims, isinv=True, align_corners=False, new_mode=True)))
        >>> print('## interpolate, align_corners=False')
        >>> print(fmt(F.interpolate(inputs, output_dims, mode='bilinear', align_corners=False)))
        >>> print('## interpolate (scale), align_corners=False')
        >>> print(ub.repr2(F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False).numpy(), precision=2))
        >>> cv2_M = mat.cpu().numpy()[0:2]
        >>> src = inputs[0, 0].cpu().numpy()
        >>> dsize = tuple(output_dims[::-1])
        >>> print('\nOpen CV warp Result')
        >>> result2 = (cv2.warpAffine(src, cv2_M, dsize=dsize, flags=cv2.INTER_LINEAR))
        >>> print('result2 =\n{}'.format(ub.repr2(result2, precision=2)))
    """

    if mode == 'linear':
        mode = 'bilinear'

    output_dims = tuple(map(int, output_dims))

    # Determine the number of space-time dimensions
    ndims = len(output_dims)

    # https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
    input_dims = inputs.shape[-ndims:]
    prefix_dims = inputs.shape[:-ndims]

    # Normalize the inputs so they are in 4D or 5D standard form
    # I.e. either [B, C, H, W] or [B, C, D, H, W]
    # We need exactly two non-spacetime (prefix) dims
    if len(prefix_dims) < 2:
        # Create a dummy batch / channel dimension
        _part1 = [1] * (2 - len(prefix_dims))
        _part2 = [-1] * len(inputs.shape)
        _input_expander = _part1 + _part2
        inputs_ = inputs.expand(*_input_expander)
    elif len(prefix_dims) > 2:
        fake_b = np.prod(prefix_dims[:-1])
        fake_c = prefix_dims[-1]
        # Consolodate leading dimensions into the batch dim
        inputs_ = inputs.view(fake_b, fake_c, *input_dims)
    else:
        inputs_ = inputs

    device = inputs.device

    input_size = torch.Tensor(np.array(input_dims[::-1]))[None, :, None]
    input_size = input_size.to(device)  # [1, ndims, 1]

    if len(mat.shape) not in [2, 3]:
        raise ValueError('Invalid mat shape')

    if mat.shape[-1] not in [3, 4] or mat.shape[-1] not in [2, 3, 4]:
        # if tuple(mat.shape) != (2, 2):
        raise ValueError(
            'mat must have shape: '
            # '(..., 2, 2) or '
            '(..., 2, 3) or (..., 3, 3)'
            ' or (..., 3, 4) or (..., 4, 4)'
        )

    # Ensure that mat is a 3x3 matrix, and check if it is affine or projective
    if mat.shape[-2] != mat.shape[-1]:
        _homog_row = [0] * (mat.shape[-1] - 1) + [1]
        homog_row = torch.Tensor(_homog_row).to(mat.device)
        homog_row = homog_row.expand_as(mat[..., 0:1, :])
        mat = torch.cat([homog_row, mat], dim=len(mat.shape) - 2)
        ishomog = False

    if ishomog is None:
        ishomog = False  # set to true for non-affine
        if mat.shape[-2] == 3:
            if not torch.all(mat[-2] != torch.Tensor([0, 0, 1])):
                ishomog = True

    inv = mat if isinv else mat.inverse()
    if len(inv.shape) == 2:
        inv = inv[None, :]

    if inv.device != device:
        inv = inv.to(device)

    # Construct a homogenous coordinate system in the output frame where the
    # input is aligned with the top left corner.
    # X = ndims + 1 if ishomog else ndims
    X = ndims + 1

    if not TORCH_GRID_SAMPLE_HAS_ALIGN:
        import warnings
        warnings.warn('cannot use new mode in warp_tensor when torch < 1.3')
        new_mode = False

    # NOTE: grid_sample in torch<1.3 does not support align_corners=False correctly
    unwarped_coords = _coordinate_grid(output_dims, align_corners=align_corners)     # [X, *DIMS]
    unwarped_coords = unwarped_coords.to(device)

    unwarped_coords_ = unwarped_coords.view(1, X, -1)  # [1, X, prod(DIMS)]
    warped_coords = inv.matmul(unwarped_coords_)
    if ishomog:
        # If we had a projective projective transform we unhomogenize
        warped_coords = warped_coords[:, 0:ndims] / warped_coords[:, ndims]
    else:
        # For affine we can simply discard the homogenous component
        warped_coords = warped_coords[:, 0:ndims]

    # Normalized the warped coordinates that align with the input to [-1, +1]
    # Anything outside of the input range is mapped outside of [-1, +1]
    if align_corners:
        grid_coords = warped_coords * (2.0 / (input_size))  # normalize from [0, 2]
        grid_coords -= 1.0                                  # normalize from [-1, +1]
    else:
        grid_coords = warped_coords * (2.0 / (input_size - 1.0))  # normalize from [0, 2]
        grid_coords -= 1.0                                        # normalize from [-1, +1]
        if new_mode:
            # HACK: For whatever reason the -1,+1 extremes doesn't point to the
            # extreme pixels, but applying this squish factor seems to help.
            # The idea seems to be that if the input dims are D x D the
            # ((D - 1) / D)-th value is what points to the middle of the bottom
            # right input pixel and not (+1, +1).
            # Need to figure out what's going on in a more principled way.
            input_dims_ = torch.FloatTensor(list(input_dims))
            squish = ((input_dims_ - 1.0) / (input_dims_))
            grid_coords = grid_coords * squish[None, :, None]

    if False:
        # Debug output coords
        print('### unwarped')
        print(unwarped_coords[0:2])
        print('### warped')
        print(warped_coords.view(2, *output_dims))
        print('### grid')
        print(grid_coords.view(2, *output_dims))

        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[-1.0, -1.0]]]]), mode='bilinear', align_corners=False)
        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[-2 / 3, -2 / 3]]]]), mode='bilinear', align_corners=False)
        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[0.0, 0.0]]]]), mode='bilinear', align_corners=False)
        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[2 / 3, 2 / 3]]]]), mode='bilinear', align_corners=False)
        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[1.0, 1.0]]]]), mode='bilinear', align_corners=False)

        F.grid_sample(inputs_[:, :, 0:2, 0:2], torch.FloatTensor(
            [[[[-1 / 2, -1 / 2]]]]), mode='bilinear', align_corners=False)
        inputs_ = torch.arange(16).view(1, 1, 4, 4).float() + 1
        F.grid_sample(inputs_, torch.FloatTensor(
            [[[[-3 / 4, -3 / 4]]]]), mode='bilinear', align_corners=False)

        for f in np.linspace(0.5, 1.0, 10):
            print('f = {!r}'.format(f))
            print(F.grid_sample(inputs_, torch.FloatTensor(
                [[[[f, f]]]]), mode='bilinear', align_corners=False))

    # The warped coordinate [-1, -1] will references to the left-top pixel of
    # the input, analgously [+1, +1] references the right-bottom pixel of the
    # input.
    # Note: that -1, -1 refers to the center of the first pixel, not the edge.
    # See:
    # https://github.com/pytorch/pytorch/issues/20785
    # https://github.com/pytorch/pytorch/pull/23923
    # https://github.com/pytorch/pytorch/pull/24929
    # https://user-images.githubusercontent.com/9757500/58150486-c5315900-7c34-11e9-9466-24f2bd431fa4.png

    # # Note: Was unable to quite figure out how to use F.affine_grid
    # gride_shape = torch.Size((B, C,) + tuple(output_dims))
    # grid = F.affine_grid(inv[None, 0:2], gride_shape)
    # outputs = F.grid_sample(inputs, grid)
    # return outputs

    # Reshape to dimensions compatible with grid_sample
    grid_coords = grid_coords.transpose(1, 2)  # swap space/coord dims
    _reshaper = [1] + list(output_dims) + [ndims]
    grid_coords = grid_coords.reshape(*_reshaper)  # Unpack dims
    _expander = [inputs_.shape[0]] + list(output_dims) + [ndims]
    grid_coords = grid_coords.expand(*_expander)

    # grid_coords = grid_coords.to(device)
    # TODO: pass align_corners when supported in torch 1.3
    # Note: enabling this breaks tests and backwards compat, so
    # verify there are no problems before enabling this.
    if new_mode and TORCH_GRID_SAMPLE_HAS_ALIGN:
        # the new grid sample allows you to set align_corners, but I don't
        # remember if the previous logic depends on the old behavior.
        outputs_ = F.grid_sample(inputs_, grid_coords, mode=mode,
                                 padding_mode=padding_mode,
                                 align_corners=bool(align_corners))
    else:
        # The old grid sample always had align_corners=True
        outputs_ = F.grid_sample(inputs_, grid_coords, mode=mode,
                                 padding_mode=padding_mode,
                                 align_corners=True)

    # Unpack outputs to match original input shape
    final_dims = list(prefix_dims) + list(output_dims)
    outputs = outputs_.view(*final_dims)
    return outputs


def subpixel_align(dst, src, index, interp_axes=None):
    """
    Returns an aligned version of the source tensor and destination index.

    Used as the backend to implement other subpixel functions like:
        subpixel_accum, subpixel_maximum.


    """
    if interp_axes is None:
        # Assume spatial dimensions are trailing
        interp_axes = len(dst.shape) + np.arange(-min(2, len(index)), 0)

    raw_subpixel_starts = np.array([0 if sl.start is None else sl.start
                                    for sl in index])
    raw_subpixel_stops = np.array([dst.shape[i] if sl.stop is None else sl.stop
                                   for i, sl in enumerate(index)])
    raw_extent = raw_subpixel_stops - raw_subpixel_starts

    if not ub.iterable(src):
        # Broadcast scalars
        impl = kwarray.ArrayAPI.impl(dst)
        shape = tuple(raw_extent.astype(int).tolist())
        src = impl.full(shape, dtype=dst.dtype, fill_value=src)

    if not np.all(np.isclose(src.shape, raw_extent, atol=0.3)):
        raise ValueError(
            'Got src.shape = {}, but the raw slice extent was {}'.format(
                tuple(src.shape), tuple(raw_extent)))
    if True:
        # check that all non interp slices are integral
        noninterp_axes = np.where(~kwarray.boolmask(interp_axes, len(dst.shape)))[0]
        for i in noninterp_axes:
            assert raw_subpixel_starts[i] % 1 == 0
            assert raw_subpixel_stops[i] % 1 == 0

    # Clip off any out of bounds
    subpixel_st, extra_padding = _rectify_slice(
        dst.shape, raw_subpixel_starts, raw_subpixel_stops)

    subpixel_starts = np.array([a[0] for a in subpixel_st])
    subpixel_stops = np.array([a[1] for a in subpixel_st])

    subpixel_pad_left = np.array([a[0] for a in extra_padding])
    # subpixel_pad_right = np.array([a[1] for a in extra_padding])

    # Any fractional start dimension will be a positive translate
    translation = np.zeros_like(subpixel_starts, dtype=float)
    translation += (subpixel_starts % 1)
    # Any value that is cutoff on the left is a negative translate
    translation -= subpixel_pad_left

    # Construct the slice in dst that will correspond to the aligned src
    aligned_index = tuple([
        slice(s, t) for s, t in zip(np.floor(subpixel_starts).astype(int),
                                    np.ceil(subpixel_stops).astype(int))])
    # Align the source coordinates with the destination coordinates
    output_shape = [sl.stop - sl.start for sl in aligned_index]

    translation_ = [translation[i] for i in interp_axes]
    aligned_src = subpixel_translate(src, translation_,
                                     output_shape=output_shape,
                                     interp_axes=interp_axes)
    return aligned_src, aligned_index


def subpixel_set(dst, src, index, interp_axes=None):
    """
    Add the source values array into the destination array at a particular
    subpixel index.

    Args:
        dst (ArrayLike): destination accumulation array
        src (ArrayLike): source array containing values to add
        index (Tuple[slice]): subpixel slice into dst that corresponds with src
        interp_axes (tuple): specify which axes should be spatially interpolated

    TODO:
        - [ ]: allow index to be a sequence indices

    Example:
        >>> import kwimage
        >>> dst = np.zeros(5) + .1
        >>> src = np.ones(2)
        >>> index = [slice(1.5, 3.5)]
        >>> kwimage.util_warp.subpixel_set(dst, src, index)
        >>> print(ub.repr2(dst, precision=2, with_dtype=0))
        np.array([0.1, 0.5, 1. , 0.5, 0.1])
    """
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    # accumulate the newly aligned source array
    try:
        dst[aligned_index] = aligned_src
    except RuntimeError:
        try:
            print('dst.shape = {!r}'.format(dst.shape))
            print('dst.dtype = {!r}'.format(dst.dtype))
            print('dst.device = {!r}'.format(dst.device))

            print('aligned_src.shape = {!r}'.format(aligned_src.shape))
            print('aligned_src.dtype = {!r}'.format(aligned_src.dtype))
            print('aligned_src.device = {!r}'.format(aligned_src.device))

            print('src.shape = {!r}'.format(src.shape))
            print('src.dtype = {!r}'.format(src.dtype))
            print('src.device = {!r}'.format(src.device))
        except Exception:
            print('unexpected numpy')
        raise
    return dst


def subpixel_accum(dst, src, index, interp_axes=None):
    """
    Add the source values array into the destination array at a particular
    subpixel index.

    Args:
        dst (ArrayLike): destination accumulation array
        src (ArrayLike): source array containing values to add
        index (Tuple[slice]): subpixel slice into dst that corresponds with src
        interp_axes (tuple): specify which axes should be spatially interpolated

    TextArt:
        Inputs:
            +---+---+---+---+---+  dst.shape = (5,)
                  +---+---+        src.shape = (2,)
                  |=======|        index = 1.5:3.5

        Subpixel shift the source by -0.5.
        When the index is non-integral, pad the aligned src with an extra value
        to ensure all dst pixels that would be influenced by the smaller
        subpixel shape are influenced by the aligned src. Note that we are not
        scaling.

                +---+---+---+      aligned_src.shape = (3,)
                |===========|      aligned_index = 1:4

    Example:
        >>> dst = np.zeros(5)
        >>> src = np.ones(2)
        >>> index = [slice(1.5, 3.5)]
        >>> subpixel_accum(dst, src, index)
        >>> print(ub.repr2(dst, precision=2, with_dtype=0))
        np.array([0. , 0.5, 1. , 0.5, 0. ])

    Example:
        >>> dst = np.zeros((6, 6))
        >>> src = np.ones((3, 3))
        >>> index = (slice(1.5, 4.5), slice(1, 4))
        >>> subpixel_accum(dst, src, index)
        >>> print(ub.repr2(dst, precision=2, with_dtype=0))
        np.array([[0. , 0. , 0. , 0. , 0. , 0. ],
                  [0. , 0.5, 0.5, 0.5, 0. , 0. ],
                  [0. , 1. , 1. , 1. , 0. , 0. ],
                  [0. , 1. , 1. , 1. , 0. , 0. ],
                  [0. , 0.5, 0.5, 0.5, 0. , 0. ],
                  [0. , 0. , 0. , 0. , 0. , 0. ]])
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> dst = torch.zeros((1, 3, 6, 6))
        >>> src = torch.ones((1, 3, 3, 3))
        >>> index = (slice(None), slice(None), slice(1.5, 4.5), slice(1.25, 4.25))
        >>> subpixel_accum(dst, src, index)
        >>> print(ub.repr2(dst.numpy()[0, 0], precision=2, with_dtype=0))
        np.array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
                  [0.  , 0.38, 0.5 , 0.5 , 0.12, 0.  ],
                  [0.  , 0.75, 1.  , 1.  , 0.25, 0.  ],
                  [0.  , 0.75, 1.  , 1.  , 0.25, 0.  ],
                  [0.  , 0.38, 0.5 , 0.5 , 0.12, 0.  ],
                  [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

    Doctest:
        >>> # TODO: move to a unit test file
        >>> subpixel_accum(np.zeros(5), np.ones(2), [slice(1.5, 3.5)]).tolist()
        [0.0, 0.5, 1.0, 0.5, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(2), [slice(0, 2)]).tolist()
        [1.0, 1.0, 0.0, 0.0, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(.5, 3.5)]).tolist()
        [0.5, 1.0, 1.0, 0.5, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(-1, 2)]).tolist()
        [1.0, 1.0, 0.0, 0.0, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(-1.5, 1.5)]).tolist()
        [1.0, 0.5, 0.0, 0.0, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(10, 13)]).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(3.25, 6.25)]).tolist()
        [0.0, 0.0, 0.0, 0.75, 1.0]
        >>> subpixel_accum(np.zeros(5), np.ones(3), [slice(4.9, 7.9)]).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.099...]
        >>> subpixel_accum(np.zeros(5), np.ones(9), [slice(-1.5, 7.5)]).tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0]
        >>> subpixel_accum(np.zeros(5), np.ones(9), [slice(2.625, 11.625)]).tolist()
        [0.0, 0.0, 0.375, 1.0, 1.0]
        >>> subpixel_accum(np.zeros(5), 1, [slice(2.625, 11.625)]).tolist()
        [0.0, 0.0, 0.375, 1.0, 1.0]
    """
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    # accumulate the newly aligned source array
    try:
        dst[aligned_index] += aligned_src
    except RuntimeError:
        try:
            print('dst.shape = {!r}'.format(dst.shape))
            print('dst.dtype = {!r}'.format(dst.dtype))
            print('dst.device = {!r}'.format(dst.device))

            print('aligned_src.shape = {!r}'.format(aligned_src.shape))
            print('aligned_src.dtype = {!r}'.format(aligned_src.dtype))
            print('aligned_src.device = {!r}'.format(aligned_src.device))

            print('src.shape = {!r}'.format(src.shape))
            print('src.dtype = {!r}'.format(src.dtype))
            print('src.device = {!r}'.format(src.device))
        except Exception:
            print('unexpected numpy')
        raise
    return dst


def subpixel_maximum(dst, src, index, interp_axes=None):
    """
    Take the max of the source values array into and the destination array at a
    particular subpixel index. Modifies the destination array.

    Args:
        dst (ArrayLike): destination array to index into
        src (ArrayLike): source array that agrees with the index
        index (Tuple[slice]): subpixel slice into dst that corresponds with src
        interp_axes (tuple): specify which axes should be spatially interpolated

    Example:
        >>> dst = np.array([0, 1.0, 1.0, 1.0, 0])
        >>> src = np.array([2.0, 2.0])
        >>> index = [slice(1.6, 3.6)]
        >>> subpixel_maximum(dst, src, index)
        >>> print(ub.repr2(dst, precision=2, with_dtype=0))
        np.array([0. , 1. , 2. , 1.2, 0. ])

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> dst = torch.zeros((1, 3, 5, 5)) + .5
        >>> src = torch.ones((1, 3, 3, 3))
        >>> index = (slice(None), slice(None), slice(1.4, 4.4), slice(1.25, 4.25))
        >>> subpixel_maximum(dst, src, index)
        >>> print(ub.repr2(dst.numpy()[0, 0], precision=2, with_dtype=0))
        np.array([[0.5 , 0.5 , 0.5 , 0.5 , 0.5 ],
                  [0.5 , 0.5 , 0.6 , 0.6 , 0.5 ],
                  [0.5 , 0.75, 1.  , 1.  , 0.5 ],
                  [0.5 , 0.75, 1.  , 1.  , 0.5 ],
                  [0.5 , 0.5 , 0.5 , 0.5 , 0.5 ]])
    """
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    impl = kwarray.ArrayAPI.impl(dst)
    impl.maximum(dst[aligned_index], aligned_src, out=dst[aligned_index])
    return dst


def subpixel_minimum(dst, src, index, interp_axes=None):
    """
    Take the min of the source values array into and the destination array at a
    particular subpixel index. Modifies the destination array.

    Args:
        dst (ArrayLike): destination array to index into
        src (ArrayLike): source array that agrees with the index
        index (Tuple[slice]): subpixel slice into dst that corresponds with src
        interp_axes (tuple): specify which axes should be spatially interpolated

    Example:
        >>> dst = np.array([0, 1.0, 1.0, 1.0, 0])
        >>> src = np.array([2.0, 2.0])
        >>> index = [slice(1.6, 3.6)]
        >>> subpixel_minimum(dst, src, index)
        >>> print(ub.repr2(dst, precision=2, with_dtype=0))
        np.array([0. , 0.8, 1. , 1. , 0. ])

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> dst = torch.zeros((1, 3, 5, 5)) + .5
        >>> src = torch.ones((1, 3, 3, 3))
        >>> index = (slice(None), slice(None), slice(1.4, 4.4), slice(1.25, 4.25))
        >>> subpixel_minimum(dst, src, index)
        >>> print(ub.repr2(dst.numpy()[0, 0], precision=2, with_dtype=0))
        np.array([[0.5 , 0.5 , 0.5 , 0.5 , 0.5 ],
                  [0.5 , 0.45, 0.5 , 0.5 , 0.15],
                  [0.5 , 0.5 , 0.5 , 0.5 , 0.25],
                  [0.5 , 0.5 , 0.5 , 0.5 , 0.25],
                  [0.5 , 0.3 , 0.4 , 0.4 , 0.1 ]])
    """
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    impl = kwarray.ArrayAPI.impl(dst)
    impl.minimum(dst[aligned_index], aligned_src, out=dst[aligned_index])
    return dst


def subpixel_slice(inputs, index):
    """
    Take a subpixel slice from a larger image.  The returned output is
    left-aligned with the requested slice.

    Args:
        inputs (ArrayLike): data
        index (Tuple[slice]): a slice to subpixel accuracy

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import kwimage
        >>> import torch
        >>> # say we have a (576, 576) input space
        >>> # and a (9, 9) output space downsampled by 64x
        >>> ospc_feats = np.tile(np.arange(9 * 9).reshape(1, 9, 9), (1024, 1, 1))
        >>> inputs = torch.from_numpy(ospc_feats)
        >>> # We detected a box in the input space
        >>> ispc_bbox = kwimage.Boxes([[64,  65, 100, 120]], 'ltrb')
        >>> # Get coordinates in the output space
        >>> ospc_bbox = ispc_bbox.scale(1 / 64)
        >>> tl_x, tl_y, br_x, br_y = ospc_bbox.data[0]
        >>> # Convert the box to a slice
        >>> index = [slice(None), slice(tl_y, br_y), slice(tl_x, br_x)]
        >>> # Note: I'm not 100% sure this work right with non-intergral slices
        >>> outputs = kwimage.subpixel_slice(inputs, index)

    Example:
        >>> inputs = np.arange(5 * 5 * 3).reshape(5, 5, 3)
        >>> index = [slice(0, 3), slice(0, 3)]
        >>> outputs = subpixel_slice(inputs, index)
        >>> index = [slice(0.5, 3.5), slice(-0.5, 2.5)]
        >>> outputs = subpixel_slice(inputs, index)

        >>> inputs = np.arange(5 * 5).reshape(1, 5, 5).astype(float)
        >>> index = [slice(None), slice(3, 6), slice(3, 6)]
        >>> outputs = subpixel_slice(inputs, index)
        >>> print(outputs)
        [[[18. 19.  0.]
          [23. 24.  0.]
          [ 0.  0.  0.]]]
        >>> index = [slice(None), slice(3.5, 6.5), slice(2.5, 5.5)]
        >>> outputs = subpixel_slice(inputs, index)
        >>> print(outputs)
        [[[20.   21.   10.75]
          [11.25 11.75  6.  ]
          [ 0.    0.    0.  ]]]
    """
    subpixel_starts = np.array(
        [0 if sl.start is None else sl.start for sl in index])
    subpixel_stops = np.array(
        [inputs.shape[i] if sl.stop is None else sl.stop
         for i, sl in enumerate(index)])
    is_fractional = ((subpixel_starts % 1) + (subpixel_stops % 1)) > 0
    if not np.any(is_fractional):
        # If none of the slices are fractional just do the simple thing
        int_index = [slice(int(s), int(t)) for s, t in
                     zip(subpixel_starts, subpixel_stops)]
        outputs, _ = _padded_slice(inputs, int_index)
    else:
        interp_axes = np.where(is_fractional)[0]
        shift = -subpixel_starts[interp_axes]
        output_shape = subpixel_stops - subpixel_starts
        if np.any(output_shape % 1 > 0):
            output_shape = np.ceil(output_shape)
            # raise ValueError('the slice length must be integral')
        output_shape = output_shape.astype(int)
        outputs = subpixel_translate(inputs, shift, interp_axes=interp_axes,
                                     output_shape=output_shape)
    return outputs


def subpixel_translate(inputs, shift, interp_axes=None, output_shape=None):
    """
    Translates an image by a subpixel shift value using bilinear interpolation

    Args:
        inputs (ArrayLike): data to translate

        shift (Sequence):
            amount to translate each dimension specified by `interp_axes`.
            Note: if inputs contains more than one "image" then all "images" are
            translated by the same amount. This function contains  no mechanism
            for translating each image differently. Note that by default
            this is a y,x shift for 2 dimensions.

        interp_axes (Sequence):
            axes to perform interpolation on, if not specified the final
            `n` axes are interpolated, where `n=len(shift)`

        output_shape (tuple):
            if specified the output is returned with this shape, otherwise

    Note:
        This function powers most other functions in this file.
        Speedups here can go a long way.

    Example:
        >>> inputs = np.arange(5) + 1
        >>> print(inputs.tolist())
        [1, 2, 3, 4, 5]
        >>> outputs = subpixel_translate(inputs, 1.5)
        >>> print(outputs.tolist())
        [0.0, 0.5, 1.5, 2.5, 3.5]

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> inputs = torch.arange(9).view(1, 1, 3, 3).float()
        >>> print(inputs.long())
        tensor([[[[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]]]])
        >>> outputs = subpixel_translate(inputs, (-.4, .5), output_shape=(1, 1, 2, 5))
        >>> print(outputs)
        tensor([[[[0.6000, 1.7000, 2.7000, 1.6000, 0.0000],
                  [2.1000, 4.7000, 5.7000, 3.1000, 0.0000]]]])

    Ignore:
        >>> inputs = np.arange(5)
        >>> shift = -.6
        >>> interp_axes = None
        >>> subpixel_translate(inputs, -.6)
        >>> subpixel_translate(inputs[None, None, None, :], -.6)
        >>> inputs = np.arange(25).reshape(5, 5)
        >>> shift = (-1.6, 2.3)
        >>> interp_axes = (0, 1)
        >>> subpixel_translate(inputs, shift, interp_axes, output_shape=(9, 9))
        >>> subpixel_translate(inputs, shift, interp_axes, output_shape=(3, 4))
    """
    impl = kwarray.ArrayAPI.impl(inputs)

    if output_shape is None:
        output_shape = inputs.shape

    if interp_axes is None:
        shift = _ensure_arraylike(shift)
        interp_axes = np.arange(-len(shift), 0)
    else:
        interp_axes = _ensure_arraylike(interp_axes)
        shift = _ensure_arraylike(shift, len(interp_axes))

    ndims = len(inputs.shape)  # number of inputs dimensions
    interp_order = len(interp_axes)  # number of interpolated dimensions
    output_dims = [output_shape[i] for i in interp_axes]
    # The negative shift defines the new start coordinate
    start = -shift

    # Sample values (using padded slice to deal with borders)
    # border_mode = 'zeros'
    # if border_mode == 'zeros':
    #     padkw = dict(pad_mode='constant', constant_value=0)
    # if border_mode == 'edge':
    #     padkw = dict(pad_mode='edge')
    padkw = {}

    if np.all(start % 1 == 0):
        # short circuit common simple cases where no interpolation is needed
        relevant_slice = [slice(None)] * ndims
        for i, x, d in zip(interp_axes, map(int, start), output_dims):
            relevant_slice[i] = slice(x, x + d)
        subpxl_vals, _ = _padded_slice(inputs, relevant_slice, **padkw)
    elif interp_order == 1:
        i, = interp_axes
        width, = output_dims
        x, = start
        # Get quantized pixel locations near subpixel pts
        x0 = int(np.floor(x))
        x1 = x0 + 1
        # Find linear weights
        wa = (x1 - x)
        wb = (x - x0)

        # Create a (potentially negative) slice containing the relvant area
        relevant_slice = [slice(None)] * ndims
        relevant_slice[i] = slice(x0, x1 + width)
        relevant, _ = _padded_slice(inputs, relevant_slice, **padkw)

        if impl.dtype_kind(relevant) != 'f':
            relevant = impl.astype(relevant, 'float32')

        # Take subslices of the relevant area
        sl_a = [slice(None)] * ndims
        sl_b = [slice(None)] * ndims

        # Sample values (using padded slice to deal with borders)
        sl_a[i] = slice(0, width)
        sl_b[i] = slice(1, width + 1)

        Ia = relevant[tuple(sl_a)]
        Ib = relevant[tuple(sl_b)]

        # Perform the linear interpolation
        subpxl_vals = (wa * Ia) + (wb * Ib)

    elif interp_order == 2:
        j, i = interp_axes
        height, width = output_dims
        y, x = start

        # Get quantized pixel locations near subpixel pts
        start0 = kwarray.ArrayAPI.ifloor(start)
        start1 = start0 + 1
        alpha = start1 - start
        beta  = start  - start0
        # Find bilinear weights
        wa = alpha[1] * alpha[0]
        wb = alpha[1] *  beta[0]
        wc =  beta[1] * alpha[0]
        wd  = beta[1] *  beta[0]

        # Create a (potentially negative) slice containing the relvant area
        relevant_slice = [slice(None)] * ndims
        y0, x0 = start0
        y1, x1 = start1
        relevant_slice[j] = slice(y0, y1 + height)
        relevant_slice[i] = slice(x0, x1 + width)
        relevant, _ = _padded_slice(inputs, relevant_slice, **padkw)

        if impl.dtype_kind(relevant) != 'f':
            relevant = impl.astype(relevant, 'float32')

        # Take subslices of the relevant area
        sl_a = [slice(None)] * ndims
        sl_b = [slice(None)] * ndims
        sl_c = [slice(None)] * ndims
        sl_d = [slice(None)] * ndims

        # Sample values (using padded slice to deal with borders)
        sl_a[j] = slice(0, height)
        sl_a[i] = slice(0, width)

        sl_b[j] = slice(1, height + 1)
        sl_b[i] = slice(0, width)

        sl_c[j] = slice(0, height)
        sl_c[i] = slice(1, width + 1)

        sl_d[j] = slice(1, height + 1)
        sl_d[i] = slice(1, width + 1)

        Ia = relevant[tuple(sl_a)]
        Ib = relevant[tuple(sl_b)]
        Ic = relevant[tuple(sl_c)]
        Id = relevant[tuple(sl_d)]

        # Perform the bilinear interpolation
        subpxl_vals = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    else:
        raise NotImplementedError('trilinear interpolation is not implemented')
    return subpxl_vals


def _padded_slice(data, in_slice, ndim=None, pad_slice=None,
                  pad_mode='constant', **padkw):
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
        in_slice (Tuple[slice, ...]): slice for each dimensions
        ndim (int): number of spatial dimensions
        pad_slice (List[int|Tuple]): additional padding of the slice

    Returns:
        Tuple[Sliceable, List] :

            data_sliced: subregion of the input data (possibly with padding,
                depending on if the original slice went out of bounds)

            st_dims : a list indicating the low and high space-time coordinate
                values of the returned data slice.

    Example:
        >>> data = np.arange(5)
        >>> in_slice = [slice(-2, 7)]

        >>> data_sliced, st_dims = _padded_slice(data, in_slice)
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        >>> print(st_dims)
        np.array([0, 0, 0, 1, 2, 3, 4, 0, 0])
        [(-2, 7)]

        >>> data_sliced, st_dims = _padded_slice(data, in_slice, pad_slice=(3, 3))
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        >>> print(st_dims)
        np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])
        [(-5, 10)]

        >>> data_sliced, st_dims = _padded_slice(data, slice(3, 4), pad_slice=[(1, 0)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        >>> print(st_dims)
        np.array([2, 3])
        [(2, 4)]
    """
    # TODO: use kwarray instead
    if isinstance(in_slice, slice):
        in_slice = [in_slice]

    ndim = len(in_slice)

    data_dims = data.shape[:ndim]

    low_dims = [sl.start for sl in in_slice]
    high_dims = [sl.stop for sl in in_slice]

    data_slice, extra_padding = _rectify_slice(data_dims, low_dims, high_dims,
                                               pad_slice=pad_slice)

    in_slice_clipped = tuple(slice(*d) for d in data_slice)
    # Get the parts of the image that are in bounds
    data_clipped = data[in_slice_clipped]

    # Add any padding that is needed to behave like negative dims exist
    if sum(map(sum, extra_padding)) == 0:
        # The slice was completely in bounds
        data_sliced = data_clipped
    else:
        if len(data.shape) != len(extra_padding):
            extra_padding = extra_padding + [(0, 0)]

        impl = kwarray.ArrayAPI.impl(data_clipped)
        data_sliced = impl.pad(data_clipped, extra_padding, mode=pad_mode,
                               **padkw)

    st_dims = data_slice[0:ndim]
    pad_dims = extra_padding[0:ndim]

    st_dims = [(s - pad[0], t + pad[1])
               for (s, t), pad in zip(st_dims, pad_dims)]
    return data_sliced, st_dims


def _ensure_arraylike(data, n=None):
    if not ub.iterable(data):
        if n is None:
            return np.array([data])
        else:
            return np.array([data] * n)
    else:
        if n is None or len(data) == n:
            return np.array(data)
        elif len(data) == 1:
            return np.repeat(data, n, axis=0)
        else:
            raise ValueError('cant fix shape')


def _rectify_slice(data_dims, low_dims, high_dims, pad_slice=None):
    """
    Given image dimensions, bounding box dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    Args:
        data_dims (tuple): n-dimension data sizes (e.g. 2d height, width)
        low_dims (tuple): bounding box low values (e.g. 2d ymin, xmin)
        high_dims (tuple): bounding box high values (e.g. 2d ymax, xmax)

        pad_slice (List[int|Tuple]):
            pad applied to (left and right) / (both) sides of each slice dim

    Returns:
        Tuple:
            data_slice - low and high values of a fancy slice corresponding to
                the image with shape `data_dims`. This slice may not correspond
                to the full window size if the requested bounding box goes out
                of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where 2D-bbox is inside the data dims on left edge
        >>> # Comprehensive 1D-cases are in the unit-test file
        >>> data_dims  = [300, 300]
        >>> low_dims   = [0, 0]
        >>> high_dims  = [10, 10]
        >>> pad_slice  = [(10, 10), (5, 5)]
        >>> a, b = _rectify_slice(data_dims, low_dims, high_dims, pad_slice)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = [(0, 20), (0, 15)]
        extra_padding = [(10, 0), (5, 0)]
    """
    # Determine the real part of the image that can be sliced out
    data_slice = []
    extra_padding = []

    if pad_slice is None:
        pad_slice = 0
    if isinstance(pad_slice, int):
        pad_slice = [pad_slice] * len(data_dims)
    # Normalize to left/right pad value for each dim
    pad_slice = [p if ub.iterable(p) else [p, p] for p in pad_slice]

    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad_slice):
        if d_low is None:
            d_low = 0
        if d_high is None:
            d_high = D_img
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


def _warp_tensor_cv2(inputs, mat, output_dims, mode='linear', ishomog=None):
    """
    implementation with cv2.warpAffine for speed / correctness comparison

    On GPU: torch is faster in both modes
    On CPU: torch is faster for homog, but cv2 is faster for affine

    Benchmark:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> from kwimage.util.util_warp import *
        >>> from kwimage.util.util_warp import _warp_tensor_cv2
        >>> from kwimage.util.util_warp import warp_tensor
        >>> import numpy as np
        >>> ti = ub.Timerit(10, bestof=3, verbose=2, unit='ms')
        >>> mode = 'linear'
        >>> rng = np.random.RandomState(0)
        >>> inputs = torch.Tensor(rng.rand(16, 10, 32, 32)).to('cpu')
        >>> mat = torch.FloatTensor([[2.5, 0, 10.5], [0, 3, 0], [0, 0, 1]])
        >>> mat[2, 0] = .009
        >>> mat[2, 2] = 2
        >>> output_dims = (64, 64)
        >>> results = ub.odict()
        >>> # -------------
        >>> for timer in ti.reset('warp_tensor(torch)'):
        >>>     with timer:
        >>>         outputs = warp_tensor(inputs, mat, output_dims, mode=mode)
        >>>         torch.cuda.synchronize()
        >>> results[ti.label] = outputs
        >>> # -------------
        >>> inputs = inputs.cpu().numpy()
        >>> mat = mat.cpu().numpy()
        >>> for timer in ti.reset('warp_tensor(cv2)'):
        >>>     with timer:
        >>>         outputs = _warp_tensor_cv2(inputs, mat, output_dims, mode=mode)
        >>> results[ti.label] = outputs
        >>> import itertools as it
        >>> for k1, k2 in it.combinations(results, 2):
        >>>     a = kwarray.ArrayAPI.numpy(results[k1])
        >>>     b = kwarray.ArrayAPI.numpy(results[k2])
        >>>     diff = np.abs(a - b)
        >>>     diff_stats = kwarray.stats_dict(diff, n_extreme=1, extreme=1)
        >>>     print('{} - {}: {}'.format(k1, k2, ub.repr2(diff_stats, nl=0, precision=4)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(results['warp_tensor(torch)'][0, 0], fnum=1, pnum=(1, 2, 1), title='torch')
        >>> kwplot.imshow(results['warp_tensor(cv2)'][0, 0], fnum=1, pnum=(1, 2, 2), title='cv2')
    """
    import cv2
    import kwimage
    impl = kwarray.ArrayAPI.impl(inputs)

    inputs = impl.numpy(inputs)
    mat = kwarray.ArrayAPI.numpy(mat)
    dsize = tuple(map(int, output_dims[::-1]))

    if mode == 'bilinear':
        mode = 'linear'
    flags = kwimage.im_cv2._coerce_interpolation(mode)
    input_shape = inputs.shape
    if len(input_shape) < 2:
        raise ValueError('height x width must be last two dims')
    inputs_ = inputs.reshape(-1, *input_shape[-2:])
    output_shape_ = tuple(inputs_.shape[0:1]) + tuple(output_dims)
    output_shape = tuple(inputs.shape[:-2]) + tuple(output_dims)
    outputs_ = np.empty(output_shape_, dtype=inputs.dtype)

    if len(mat.shape) not in [2, 3]:
        raise ValueError('Invalid mat shape')

    if ishomog is None:
        ishomog = False
        if mat.shape[-2] == 3:
            if not np.all(mat[-2] != [0, 0, 1]):
                ishomog = True

    if ishomog:
        M = mat
        for src, dst in zip(inputs_, outputs_):
            cv2.warpPerspective(src, M, dsize=dsize, dst=dst, flags=flags)
    else:
        M = mat[0:2]
        for src, dst in zip(inputs_, outputs_):
            cv2.warpAffine(src, M, dsize=dsize, dst=dst, flags=flags)
    outputs = outputs_.reshape(output_shape)

    outputs = impl.ensure(outputs)
    return outputs


def warp_points(matrix, pts, homog_mode='divide'):
    """
    Warp ND points / coordinates using a transformation matrix.

    Homogoenous coordinates are added on the fly if needed. Works with both
    numpy and torch.

    Args:
        matrix (ArrayLike): [D1 x D2] transformation matrix.
            if using homogenous coordinates D2=D + 1, otherwise D2=D.
            if using homogenous coordinates and the matrix represents an Affine
            transformation, then either D1=D or D1=D2, i.e. the last row of
            zeros and a one is optional.

        pts (ArrayLike): [N1 x ... x D] points (usually x, y).
            If points are already in homogenous space, then the output will be
            returned in homogenous space. D is the dimensionality of the
            points.  The leading axis may take any shape, but usually, shape
            will be [N x D] where N is the number of points.

        homog_mode (str):
            what to do for homogenous coordinates. Can either divide, keep, or
            drop. Defaults do 'divide'.

    Retrns:
        new_pts (ArrayLike): the points after being transformed by the matrix

    Example:
        >>> from kwimage.util_warp import *  # NOQA
        >>> # --- with numpy
        >>> rng = np.random.RandomState(0)
        >>> pts = rng.rand(10, 2)
        >>> matrix = rng.rand(2, 2)
        >>> warp_points(matrix, pts)
        >>> # --- with torch
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> pts = torch.Tensor(pts)
        >>> matrix = torch.Tensor(matrix)
        >>> warp_points(matrix, pts)

    Example:
        >>> from kwimage.util_warp import *  # NOQA
        >>> # --- with numpy
        >>> pts = np.ones((10, 2))
        >>> matrix = np.diag([2, 3, 1])
        >>> ra = warp_points(matrix, pts)
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> rb = warp_points(torch.Tensor(matrix), torch.Tensor(pts))
        >>> assert np.allclose(ra, rb.numpy())

    Example:
        >>> from kwimage.util_warp import *  # NOQA
        >>> # test different cases
        >>> rng = np.random.RandomState(0)
        >>> # Test 3x3 style projective matrices
        >>> pts = rng.rand(1000, 2)
        >>> matrix = rng.rand(3, 3)
        >>> ra33 = warp_points(matrix, pts)
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> rb33 = warp_points(torch.Tensor(matrix), torch.Tensor(pts))
        >>> assert np.allclose(ra33, rb33.numpy())
        >>> # Test opencv style affine matrices
        >>> pts = rng.rand(10, 2)
        >>> matrix = rng.rand(2, 3)
        >>> ra23 = warp_points(matrix, pts)
        >>> rb23 = warp_points(torch.Tensor(matrix), torch.Tensor(pts))
        >>> assert np.allclose(ra33, rb33.numpy())
    """
    impl = kwarray.ArrayAPI.coerce(pts)

    if len(matrix.shape) != 2:
        raise ValueError('matrix must have 2 dimensions')

    new_shape = pts.shape
    D = pts.shape[-1]  # the trailing axis is the point dimensionality
    D1, D2 = matrix.shape

    # Reshape points into a NxD, and transpose into a
    pts_T = impl.T(impl.view(pts, (-1, D)))

    if D != D2:
        assert D + 1 == D2, 'only can have one homog coord'
        # Add homogenous coordinate
        new_pts_T = impl.cat([pts_T, impl.ones_like(pts_T[0:1])], axis=0)
    else:
        # new_pts_T = impl.contiguous(pts_T)
        new_pts_T = pts_T

    # TODO: we could be more memory efficient (and possibly faster) by using
    # the `out` kwarg and resuing memory in `new_pts_T`, we just need to ensure
    # it doesn't share memory with `pts`, which is probably doable by just
    # using imp.contiguous, but this needs testing.
    new_pts_T = impl.matmul(matrix, new_pts_T)

    if D != D1:
        if homog_mode == 'divide':
            # remove homogenous coordinates (unless the matrix was affine with
            # the last row was ommitted)
            new_pts_T = new_pts_T[0:D] / new_pts_T[-1:]
        elif homog_mode == 'drop':
            # FIXME: the drop mode probably doesn't correspond to anything real
            # and thus should be removed
            new_pts_T = new_pts_T[0:D]
        elif homog_mode == 'keep':
            new_pts_T = new_pts_T
            new_shape = pts.shape[0:-1] + (D1,)
        else:
            raise KeyError(homog_mode)

    # Return the warped points with the same shape as the input
    new_pts = impl.T(new_pts_T)
    new_pts = impl.view(new_pts, new_shape)
    return new_pts


def remove_homog(pts, mode='divide'):
    """
    Remove homogenous coordinate to a point array.

    This is a convinience function, it is not particularly efficient.

    SeeAlso:
        cv2.convertPointsFromHomogeneous

    Example:
        >>> homog_pts = np.random.rand(10, 3)
        >>> remove_homog(homog_pts, 'divide')
        >>> remove_homog(homog_pts, 'drop')
    """
    impl = kwarray.ArrayAPI.coerce(pts)
    D = pts.shape[-1]  # the trailing axis is the point dimensionality
    new_D = D - 1
    pts_T = impl.T(impl.view(pts, (-1, D)))
    if mode == 'divide':
        new_pts_T = pts_T[0:new_D] / pts_T[-1:]
    elif mode == 'drop':
        # FIXME: the drop mode probably doesn't correspond to anything real
        # and thus should be removed
        new_pts_T = pts_T[0:new_D]
    else:
        raise KeyError(mode)
    new_pts = impl.T(new_pts_T)
    return new_pts


def add_homog(pts):
    """
    Add a homogenous coordinate to a point array

    This is a convinience function, it is not particularly efficient.

    SeeAlso:
        cv2.convertPointsToHomogeneous

    Example:
        >>> pts = np.random.rand(10, 2)
        >>> add_homog(pts)

    Benchmark:
        >>> import timerit
        >>> ti = timerit.Timerit(1000, bestof=10, verbose=2)
        >>> pts = np.random.rand(1000, 2)
        >>> for timer in ti.reset('kwimage'):
        >>>     with timer:
        >>>         kwimage.add_homog(pts)
        >>> for timer in ti.reset('cv2'):
        >>>     with timer:
        >>>         cv2.convertPointsToHomogeneous(pts)
        >>> # cv2 is 4x faster, but has more restrictive inputs
    """
    import kwarray
    impl = kwarray.ArrayAPI.coerce(pts)
    new_pts = impl.cat([
        pts, impl.ones(pts.shape[0:-1] + (1,), dtype=pts.dtype)], axis=-1)
    return new_pts


def subpixel_getvalue(img, pts, coord_axes=None, interp='bilinear',
                      bordermode='edge'):
    """
    Get values at subpixel locations

    Args:
        img (ArrayLike): image to sample from
        pts (ArrayLike): subpixel rc-coordinates to sample
        coord_axes (Sequence):
            axes to perform interpolation on, if not specified the first `d`
            axes are interpolated, where `d=pts.shape[-1]`.
            IE: this indicates which axes each coordinate dimension corresponds to.
        interp (str): interpolation mode
        bordermode (str): how locations outside the image are handled

    Example:
        >>> from kwimage.util_warp import *  # NOQA
        >>> img = np.arange(3 * 3).reshape(3, 3)
        >>> pts = np.array([[1, 1], [1.5, 1.5], [1.9, 1.1]])
        >>> subpixel_getvalue(img, pts)
        array([4. , 6. , 6.8])
        >>> subpixel_getvalue(img, pts, coord_axes=(1, 0))
        array([4. , 6. , 5.2])
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> img = torch.Tensor(img)
        >>> pts = torch.Tensor(pts)
        >>> subpixel_getvalue(img, pts)
        tensor([4.0000, 6.0000, 6.8000])
        >>> subpixel_getvalue(img.numpy(), pts.numpy(), interp='nearest')
        array([4., 8., 7.], dtype=float32)
        >>> subpixel_getvalue(img.numpy(), pts.numpy(), interp='nearest', coord_axes=[1, 0])
        array([4., 8., 5.], dtype=float32)
        >>> subpixel_getvalue(img, pts, interp='nearest')
        tensor([4., 8., 7.])

    References:
        stackoverflow.com/uestions/12729228/simple-binlin-interp-images-numpy

    SeeAlso:
        cv2.getRectSubPix(image, patchSize, center[, patch[, patchType]])
    """
    # Image info
    impl = kwarray.ArrayAPI.coerce(img)
    ptsT = impl.T(pts)
    assert bordermode == 'edge'

    if coord_axes is None:
        coord_axes = list(range(len(ptsT)))

    if interp == 'nearest':
        r, c = impl.iround(ptsT, dtype=int)
        ndims = len(img.shape)
        index_a = [slice(None)] * ndims
        i, j = coord_axes
        index_a[i] = r
        index_a[j] = c
        subpxl_vals = img[tuple(index_a)]
    elif interp == 'bilinear':
        # Subpixel locations to sample
        indices, weights = _bilinear_coords(ptsT, impl, img, coord_axes)
        index_a, index_b, index_c, index_d = indices
        wa, wb, wc, wd = weights

        # Sample values
        Ia = img[index_a]
        Ib = img[index_b]
        Ic = img[index_c]
        Id = img[index_d]

        Iwa = wa * Ia
        Iwb = wb * Ib
        Iwc = wc * Ic
        Iwd = wd * Id

        # Perform the bilinear interpolation
        subpxl_vals = Iwa
        subpxl_vals += Iwb
        subpxl_vals += Iwc
        subpxl_vals += Iwd
    else:
        raise KeyError(interp)

    return subpxl_vals


def subpixel_setvalue(img, pts, value, coord_axes=None,
                      interp='bilinear', bordermode='edge'):
    """
    Set values at subpixel locations

    Args:
        img (ArrayLike): image to set values in
        pts (ArrayLike): subpixel rc-coordinates to set
        value (ArrayLike): value to place in the image
        coord_axes (Sequence):
            axes to perform interpolation on, if not specified the first `d`
            axes are interpolated, where `d=pts.shape[-1]`.
            IE: this indicates which axes each coordinate dimension corresponds to.
        interp (str): interpolation mode
        bordermode (str): how locations outside the image are handled

    Example:
        >>> from kwimage.util_warp import *  # NOQA
        >>> img = np.arange(3 * 3).reshape(3, 3).astype(float)
        >>> pts = np.array([[1, 1], [1.5, 1.5], [1.9, 1.1]])
        >>> interp = 'bilinear'
        >>> value = 0
        >>> print('img = {!r}'.format(img))
        >>> pts = np.array([[1.5, 1.5]])
        >>> img2 = subpixel_setvalue(img.copy(), pts, value)
        >>> print('img2 = {!r}'.format(img2))
        >>> pts = np.array([[1.0, 1.0]])
        >>> img2 = subpixel_setvalue(img.copy(), pts, value)
        >>> print('img2 = {!r}'.format(img2))
        >>> pts = np.array([[1.1, 1.9]])
        >>> img2 = subpixel_setvalue(img.copy(), pts, value)
        >>> print('img2 = {!r}'.format(img2))
        >>> img2 = subpixel_setvalue(img.copy(), pts, value, coord_axes=[1, 0])
        >>> print('img2 = {!r}'.format(img2))
    """
    assert bordermode == 'edge'
    # Image info
    impl = kwarray.ArrayAPI.coerce(img)
    ptsT = impl.T(pts)
    ndims = len(img.shape)

    if len(pts) == 0:
        return img

    if coord_axes is None:
        # TODO: cleanup
        coord_axes = list(range(len(ptsT)))

    if interp == 'nearest':
        r, c = impl.iround(ptsT, dtype=int)
        index_a = [slice(None)] * ndims
        i, j = coord_axes
        index_a[i] = r
        index_a[j] = c
        img[tuple(index_a)] = value
    elif interp == 'bilinear':
        # Get quantized pixel locations near subpixel pts
        # TODO: Figure out an efficient way to do this.
        indices, weights = _bilinear_coords(ptsT, impl, img, coord_axes)
        index_a, index_b, index_c, index_d = indices
        wa, wb, wc, wd = weights

        # set values (blend old values with new values at subpixel locs)
        # when location is an integer, the value is exactly overwritten.
        # I'm unsure if this is correct
        Ia = (1 - wa) * img[index_a] + (wa * value)
        Ib = (1 - wb) * img[index_b] + (wb * value)
        Ic = (1 - wc) * img[index_c] + (wc * value)
        Id = (1 - wd) * img[index_d] + (wd * value)

        img[index_a] = Ia
        img[index_b] = Ib
        img[index_c] = Ic
        img[index_d] = Id
    else:
        raise KeyError(interp)
    return img


def _bilinear_coords(ptsT, impl, img, coord_axes):
    i, j = coord_axes
    height, width = img.shape[0:2]
    ndims = len(img.shape)

    r, c = ptsT
    # Get quantized pixel locations near subpixel pts
    r0, c0 = impl.floor(ptsT)
    c1 = c0 + 1
    r1 = r0 + 1

    # Make sure the values do not go past the boundary
    # Note: this is equivalent to bordermode=edge
    c0 = impl.clip(c0, 0, width - 1, out=c0)
    c1 = impl.clip(c1, 0, width - 1, out=c1)
    r0 = impl.clip(r0, 0, height - 1, out=r0)
    r1 = impl.clip(r1, 0, height - 1, out=r1)

    # Find bilinear weights
    alpha0 = (c - c0)
    alpha1 = (c1 - c)
    beta0 = (r - r0)
    beta1 = (r1 - r)
    wa = alpha1 * beta1
    wb = alpha1 * beta0
    wc = alpha0 * beta1
    wd = alpha0 * beta0

    nChannels = 1 if len(img.shape) == 2 else img.shape[2]
    if nChannels != 1:
        wa = impl.T(impl.asarray([wa] *  nChannels))
        wb = impl.T(impl.asarray([wb] *  nChannels))
        wc = impl.T(impl.asarray([wc] *  nChannels))
        wd = impl.T(impl.asarray([wd] *  nChannels))

    r0 = impl.astype(r0, int)
    r1 = impl.astype(r1, int)
    c0 = impl.astype(c0, int)
    c1 = impl.astype(c1, int)

    index_a = [slice(None)] * ndims
    index_b = [slice(None)] * ndims
    index_c = [slice(None)] * ndims
    index_d = [slice(None)] * ndims

    index_a[i] = r0
    index_b[i] = r1
    index_c[i] = r0
    index_d[i] = r1

    index_a[j] = c0
    index_b[j] = c0
    index_c[j] = c1
    index_d[j] = c1

    indices = (
        tuple(index_a),
        tuple(index_b),
        tuple(index_c),
        tuple(index_d),
    )
    weights = (wa, wb, wc, wd)
    return indices, weights
