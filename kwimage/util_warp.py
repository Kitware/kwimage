# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.functional as F
import ubelt as ub
import torch
import numpy as np


def _coordinate_grid(dims):
    """
    Creates a homogenous coordinate system.

    Args:
        dims (Tuple[int*]): height / width or depth / height / width

    Returns:
        Tensor[shape=(3, *DIMS)]

    References:
        https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> _coordinate_grid((2, 2))
        tensor([[[0., 1.],
                 [0., 1.]],
                [[0., 0.],
                 [1., 1.]],
                [[1., 1.],
                 [1., 1.]]])
        >>> _coordinate_grid((2, 2, 2))
    """
    if len(dims) == 2:
        h, w = dims
        h_range = torch.arange(0, h).view(h, 1).expand(h, w).float()  # [H, W]
        w_range = torch.arange(0, w).view(1, w).expand(h, w).float()  # [H, W]
        ones = torch.ones(h, w)
        pixel_coords = torch.stack((w_range, h_range, ones), dim=0)  # [3, H, W]
    elif len(dims) == 3:
        d, h, w = dims
        d_range = torch.arange(0, d).view(d, 1, 1).expand(d, h, w).float()  # [D, H, W]
        h_range = torch.arange(0, h).view(1, h, 1).expand(d, h, w).float()  # [D, H, W]
        w_range = torch.arange(0, w).view(1, 1, w).expand(d, h, w).float()  # [D, H, W]
        ones = torch.ones(d, h, w)
        pixel_coords = torch.stack((w_range, h_range, d_range, ones), dim=0)  # [4, D, H, W]
        pass
    else:
        raise NotImplementedError('Can only work with 2d and 3d dims')
    return pixel_coords


def warp_tensor(inputs, mat, output_dims, mode='bilinear',
                padding_mode='zeros', isinv=False, ishomog=None):
    r"""
    A pytorch implementation of warp affine that works similarly to
    cv2.warpAffine / cv2.warpPerspective.

    It is possible to use 3x3 transforms to warp 2D image data.
    It is also possible to use 4x4 transforms to warp 3D volumetric data.

    Args:
        inputs (Tensor[..., *DIMS]): tensor to warp.
            Up to 3 (determined by output_dims) of the trailing space-time
            dimensions are warped. Best practice is to use inputs with the
            shape in [B, C, *DIMS].

        mat (Tensor):
            either a 3x3 / 4x4 single transformation matrix to apply to all
            inputs or Bx3x3 or Bx4x4 tensor that specifies a transformation
            matrix for each batch item.

        output_dims (Tuple[int*]):
            The output space-time dimensions. This can either be in the form
                (W,), (H, W), or (D, H, W).

        mode (str):
            see `torch.nn.functional.grid_sample`

        padding_mode (str):
            see `torch.nn.functional.grid_sample`

        isinv (bool, default=False):
            Set to true if `mat` is the inverse transform

        ishomog (bool, default=None):
            Set to True if the matrix is non-affine

    Notes:
        Also, it may be possible to speed up the code with `F.affine_grid`

        KNOWN ISSUE: There appears to some difference with cv2.warpAffine when
            rotation or shear are non-zero. I'm not sure what the cause is.
            It may just be floating point issues, but Im' not sure.

    References:
        https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
        https://github.com/pytorch/pytorch/issues/15386

    Example:
        >>> # Create a relatively simple affine matrix
        >>> import skimage
        >>> mat = torch.FloatTensor(skimage.transform.AffineTransform(
        >>>     translation=[1, -1], scale=[.5, 2],
        >>>     rotation=0, shear=0,
        >>> ).params)
        >>> # Create inputs and an output dimension
        >>> input_shape = [1, 1, 4, 5]
        >>> inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>> output_dims = (11, 7)
        >>> # Warp with our code
        >>> result1 = warp_tensor(inputs, mat, output_dims=output_dims)
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
        >>> result1 = warp_tensor(inputs, mat, output_dims=output_dims)
        >>> print('result1 =\n{}'.format(ub.repr2(result1.cpu().numpy()[0, 0], precision=2)))
        >>> # Warp with opencv
        >>> import cv2
        >>> cv2_M = mat.cpu().numpy()[0:2]
        >>> src = inputs[0, 0].cpu().numpy()
        >>> dsize = tuple(output_dims[::-1])
        >>> result2 = cv2.warpAffine(src, cv2_M, dsize=dsize, flags=cv2.INTER_LINEAR)
        >>> print('result2 =\n{}'.format(ub.repr2(result2, precision=2)))
        >>> # Ensure the results are the same (up to floating point errors)
        >>> # NOTE: The floating point errors seem to be significant for rotation / shear
        >>> assert np.all(np.isclose(result1[0, 0].cpu().numpy(), result2, atol=1, rtol=1e-2))

    Example:
        >>> # Test 3D warping with identity
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
        >>> result = warp_tensor(inputs, mat, output_dims=output_dims)
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
        >>> mat = torch.eye(4)
        >>> input_dims = [5, 5, 5]
        >>> output_dims = (6, 6, 6)
        >>> for n_prefix_dims in [0, 1, 2, 3, 4, 5]:
        >>>      input_shape = [2] * n_prefix_dims + input_dims
        >>>      inputs = torch.arange(int(np.prod(input_shape))).reshape(*input_shape).float()
        >>>      result = warp_tensor(inputs, mat, output_dims=output_dims)
        >>>      #print('result =\n{}'.format(ub.repr2(result.cpu().numpy(), precision=2)))
        >>>      print(result.shape)
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

    input_size = torch.Tensor(np.array(input_dims[::-1]))[None, :, None]
    input_size = input_size.to(inputs.device)  # [1, ndims, 1]

    if len(mat.shape) not in [2, 3]:
        raise ValueError('Invalid mat shape')

    if mat.shape[-1] not in [3, 4] or mat.shape[-1] not in [2, 3, 4]:
        raise ValueError(
            'mat must have shape: '
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

    if inv.device != inputs.device:
        inv = inv.to(inputs.device)

    # Construct a homogenous coordinate system in the output frame where the
    # input is aligned with the top left corner.
    X = ndims + 1
    unwarped_coords = _coordinate_grid(output_dims)     # [X, *DIMS]
    unwarped_coords = unwarped_coords.to(inputs.device)

    unwarped_coords_ = unwarped_coords.view(1, X, -1)  # [1, X, prod(DIMS)]
    warped_coords = inv.matmul(unwarped_coords_)
    if ishomog:
        # If we had a projective projective transform we unhomogenize
        warped_coords = warped_coords[:, 0:ndims] / warped_coords[:, ndims]
    else:
        # For affine we can simply discard the homogenous component
        warped_coords = warped_coords[:, 0:ndims]

    # Normalized the warped coordinates that align with the input to [-1, +1]
    # Anything outside of the input range is mappd outside of [-1, +1]
    warped_coords *= 2.0 / (input_size - 1.0)  # normalize from [0, 2]
    warped_coords -= 1.0                       # normalize from [-1, +1]

    # # Note: Was unable to quite figure out how to use F.affine_grid
    # gride_shape = torch.Size((B, C,) + tuple(output_dims))
    # grid = F.affine_grid(inv[None, 0:2], gride_shape)
    # outputs = F.grid_sample(inputs, grid)
    # return outputs

    # Reshape to dimensions compatible with grid_sample
    warped_coords = warped_coords.transpose(1, 2)  # swap space/coord dims
    _reshaper = [1] + list(output_dims) + [ndims]
    warped_coords = warped_coords.reshape(*_reshaper)  # Unpack dims
    _expander = [inputs_.shape[0]] + list(output_dims) + [ndims]
    warped_coords = warped_coords.expand(*_expander)

    # warped_coords = warped_coords.to(inputs.device)
    outputs_ = F.grid_sample(inputs_, warped_coords, mode=mode,
                             padding_mode=padding_mode)

    # Unpack outputs to match original input shape
    final_dims = list(prefix_dims) + list(output_dims)
    outputs = outputs_.view(*final_dims)
    return outputs


def subpixel_align(dst, src, index, interp_axes=None):
    """
    Returns an aligned version of the source tensor and destination index.
    """
    import kwil
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
        impl = kwil.ArrayAPI.impl(dst)
        shape = tuple(raw_extent.astype(np.int).tolist())
        src = impl.full(shape, dtype=dst.dtype, fill_value=src)

    if not np.all(np.isclose(src.shape, raw_extent, atol=0.3)):
        raise ValueError(
            'Got src.shape = {}, but the raw slice extent was {}'.format(
                tuple(src.shape), tuple(raw_extent)))
    if True:
        # check that all non interp slices are integral
        noninterp_axes = np.where(~kwil.boolmask(interp_axes, len(dst.shape)))[0]
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
        slice(s, t) for s, t in zip(np.floor(subpixel_starts).astype(np.int),
                                    np.ceil(subpixel_stops).astype(np.int))])
    # Align the source coordinates with the destination coordinates
    output_shape = [sl.stop - sl.start for sl in aligned_index]

    translation_ = [translation[i] for i in interp_axes]
    aligned_src = subpixel_translate(src, translation_,
                                     output_shape=output_shape,
                                     interp_axes=interp_axes)
    return aligned_src, aligned_index


def subpixel_accum(dst, src, index, interp_axes=None):
    """
    Add the source values array into the destination array at a particular
    subpixel index.

    Args:
        dst (ArrayLike): destination accumulation array
        src (ArrayLike): source array containing values to add
        index (Tuple[slice]): subpixel slice into dst that corresponds with src
        interp_axes (tuple): specify which axes should be spatially interpolated

    Notes:
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
    import kwil
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    impl = kwil.ArrayAPI.impl(dst)
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
    import kwil
    aligned_src, aligned_index = subpixel_align(dst, src, index, interp_axes)
    impl = kwil.ArrayAPI.impl(dst)
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
        >>> inputs = np.arange(5 * 5 * 3).reshape(5, 5, 3)
        >>> index = [slice(0, 3), slice(0, 3)]
        >>> outputs = subpixel_slice(inputs, index)
        >>> index = [slice(0.5, 3.5), slice(-0.5, 2.5)]
        >>> outputs = subpixel_slice(inputs, index)

        >>> inputs = np.arange(5 * 5).reshape(1, 5, 5).astype(np.float)
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
            raise ValueError('the slice length must be integral')
        output_shape = output_shape.astype(np.int)
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

        interp_axes (Sequence, default=None):
            axes to perform interpolation on, if not specified the final
            `n` axes are interpolated, where `n=len(shift)`

        output_shape (tuple, default=None):
            if specified the output is returned with this shape, otherwise

    Example:
        >>> inputs = np.arange(5) + 1
        >>> print(inputs.tolist())
        [1, 2, 3, 4, 5]
        >>> outputs = subpixel_translate(inputs, 1.5)
        >>> print(outputs.tolist())
        [0.0, 0.5, 1.5, 2.5, 3.5]

    Example:
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
    import kwil
    impl = kwil.ArrayAPI.impl(inputs)

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
        start0 = kwil.ArrayAPI.ifloor(start)
        start1 = start0 + 1
        alpha = start1 - start
        beta  = start  - start0
        # Find bilinear weights
        wa = alpha[1] * alpha[0]
        wb = alpha[1] *  beta[0]
        wc =  beta[1] * alpha[0]
        wd  = beta[1] *  beta[0]

        # y0 = int(np.floor(y))
        # x0 = int(np.floor(x))
        # y1 = y0 + 1
        # x1 = x0 + 1
        # # Find bilinear weights
        # wa = (x1 - x) * (y1 - y)
        # wb = (x1 - x) * (y - y0)
        # wc = (x - x0) * (y1 - y)
        # wd = (x - x0) * (y - y0)

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
    import kwil
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

        impl = kwil.ArrayAPI.impl(data_clipped)
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
        >>> from kwil.util.util_warp import *
        >>> from kwil.util.util_warp import _warp_tensor_cv2
        >>> from kwil.util.util_warp import warp_tensor
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
        >>>     a = kwil.ArrayAPI.numpy(results[k1])
        >>>     b = kwil.ArrayAPI.numpy(results[k2])
        >>>     diff = np.abs(a - b)
        >>>     diff_stats = kwil.stats_dict(diff, n_extreme=1, extreme=1)
        >>>     print('{} - {}: {}'.format(k1, k2, ub.repr2(diff_stats, nl=0, precision=4)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwil.autompl()
        >>> kwil.imshow(results['warp_tensor(torch)'][0, 0], fnum=1, pnum=(1, 2, 1), title='torch')
        >>> kwil.imshow(results['warp_tensor(cv2)'][0, 0], fnum=1, pnum=(1, 2, 2), title='cv2')
    """
    import cv2
    import kwil
    impl = kwil.ArrayAPI.impl(inputs)

    inputs = impl.numpy(inputs)
    mat = kwil.ArrayAPI.numpy(mat)
    dsize = tuple(map(int, output_dims[::-1]))

    if mode == 'bilinear':
        mode = 'linear'
    flags = kwil.imutil.im_cv2._rectify_interpolation(mode)
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
