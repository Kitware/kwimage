# -*- coding: utf-8 -*-
"""
Logic pertaining to run-length encodings

SeeAlso:
    kwimage.structs.mask - stores binary segmentation masks, using RLEs as a
        backend representation. Also contains cython logic for handling
        the coco-rle format.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def encode_run_length(img, binary=False, order='C'):
    """
    Construct the run length encoding (RLE) of an image.

    Args:
        img (ndarray): 2D image
        binary (bool, default=True): set to True for compatibility with COCO
        order ({'C', 'F'}, default='C'): row-major (C) or column-major (F)

    Returns:
        Dict[str, object]: encoding: dictionary items are:
            counts (ndarray): the run length encoding
            shape (Tuple): the original image shape
            binary (bool): if the counts encoding is binary or multiple values are ok
            order ({'C', 'F'}, default='C'): encoding order

    SeeAlso:
        * kwimage.Mask - a cython-backed data structure to handle coco-style RLEs

    Example:
        >>> import ubelt as ub
        >>> lines = ub.codeblock(
        >>>     '''
        >>>     ..........
        >>>     ......111.
        >>>     ..2...111.
        >>>     .222..111.
        >>>     22222.....
        >>>     .222......
        >>>     ..2.......
        >>>     ''').replace('.', '0').splitlines()
        >>> img = np.array([list(map(int, line)) for line in lines])
        >>> encoding = encode_run_length(img)
        >>> target = np.array([0,16,1,3,0,3,2,1,0,3,1,3,0,2,2,3,0,2,1,3,0,1,2,5,0,6,2,3,0,8,2,1,0,7])
        >>> assert np.all(target == encoding['counts'])

    Example:
        >>> binary = True
        >>> img = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0]])
        >>> encoding = encode_run_length(img, binary=True)
        >>> assert encoding['counts'].tolist() == [0, 1, 1, 3, 2, 1, 1]
    """
    flat = img.ravel(order=order)
    diff_idxs = np.flatnonzero(np.abs(np.diff(flat)) > 0)
    pos = np.hstack([[0], diff_idxs + 1])

    lengths = np.hstack([np.diff(pos), [len(flat) - pos[-1]]])
    values = flat[pos]

    if binary:
        # Assume there are only zeros and ones
        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError('more than 0 and 1')

        if len(values) and values[0] != 0:
            # the binary RLE always starts with zero
            counts = np.hstack([[0], lengths])
        else:
            counts = lengths
    else:
        counts = np.hstack([values[:, None], lengths[:, None]]).ravel()

    encoding = {
        'shape': img.shape,
        'counts': counts,
        'binary': binary,
        'order': order,
    }
    return encoding


def decode_run_length(counts, shape, binary=False, dtype=np.uint8, order='C'):
    """
    Decode run length encoding back into an image.

    Args:
        counts (ndarray): the run-length encoding
        shape (Tuple[int, int]), the height / width of the mask
        binary (bool): if the RLU is binary or non-binary.
            Set to True for compatibility with COCO.
        dtype (dtype, default=np.uint8): data type for decoded image
        order ({'C', 'F'}, default='C'): row-major (C) or column-major (F)

    Returns:
        ndarray: the reconstructed image

    Example:
        >>> from kwimage.im_runlen import *  # NOQA
        >>> img = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0]])
        >>> encoded = encode_run_length(img, binary=True)
        >>> recon = decode_run_length(**encoded)
        >>> assert np.all(recon == img)

        >>> import ubelt as ub
        >>> lines = ub.codeblock(
        >>>     '''
        >>>     ..........
        >>>     ......111.
        >>>     ..2...111.
        >>>     .222..111.
        >>>     22222.....
        >>>     .222......
        >>>     ..2.......
        >>>     ''').replace('.', '0').splitlines()
        >>> img = np.array([list(map(int, line)) for line in lines])
        >>> encoded = encode_run_length(img)
        >>> recon = decode_run_length(**encoded)
        >>> assert np.all(recon == img)
    """
    recon = np.zeros(shape, dtype=dtype, order=order)
    flat = recon.ravel(order)
    if binary:
        value = 0
        start = 0
        for num in counts:
            stop = start + num
            flat[start:stop] = value
            # print('value, start, start = {}, {}, {}'.format(value, start, stop))
            start = stop
            value = 1 - value
    else:
        start = 0
        for value, num in zip(counts[::2], counts[1::2]):
            stop = start + num
            flat[start:stop] = value
            start = stop
    return recon


def rle_translate(rle, offset, output_shape=None):
    """
    Translates a run-length encoded image in RLE-space.

    Args:
        rle (dict): an enconding dict returned by `encode_run_length`
        offset (Tuple): x,y offset,
            CAREFUL, this can only accept integers
        output_shape (Tuple, optional): h,w of transformed mask.
            If unspecified the input rle shape is used.

    SeeAlso:
        # ITK has some RLE code that looks like it can perform translations
        https://github.com/KitwareMedical/ITKRLEImage/blob/master/include/itkRLERegionOfInterestImageFilter.h

    Doctest:
        >>> # test that translate works on all zero images
        >>> img = np.zeros((7, 8), dtype=np.uint8)
        >>> rle = encode_run_length(img, binary=True, order='F')
        >>> new_rle = rle_translate(rle, (1, 2), (6, 9))
        >>> assert np.all(new_rle['counts'] == [54])

    Example:
        >>> from kwimage.im_runlen import *  # NOQA
        >>> img = np.array([
        >>>     [1, 1, 1, 1],
        >>>     [0, 1, 0, 0],
        >>>     [0, 1, 0, 1],
        >>>     [1, 1, 1, 1],], dtype=np.uint8)
        >>> rle = encode_run_length(img, binary=True, order='C')
        >>> offset = (1, -1)
        >>> output_shape = (3, 5)
        >>> new_rle = rle_translate(rle, offset, output_shape)
        >>> decoded = decode_run_length(**new_rle)
        >>> print(decoded)
        [[0 0 1 0 0]
         [0 0 1 0 1]
         [0 1 1 1 1]]

    Example

        >>> from kwimage.im_runlen import *  # NOQA
        >>> img = np.array([
        >>>     [0, 0, 0],
        >>>     [0, 1, 0],
        >>>     [0, 0, 0]], dtype=np.uint8)
        >>> rle = encode_run_length(img, binary=True, order='C')
        >>> new_rle = rle_translate(rle, (1, 0))
        >>> decoded = decode_run_length(**new_rle)
        >>> print(decoded)
        [[0 0 0]
         [0 0 1]
         [0 0 0]]
        >>> new_rle = rle_translate(rle, (0, 1))
        >>> decoded = decode_run_length(**new_rle)
        >>> print(decoded)
        [[0 0 0]
         [0 0 0]
         [0 1 0]]
    """
    if set(rle.keys()) == {'size', 'counts'}:
        # Handle coco rle's
        rle = rle.copy()
        rle['shape'] = rle['size']
        rle['order'] = 'F'
        rle['binary'] = True

    if not rle['binary']:
        raise NotImplementedError(
            'only binary rle translation is implemented')

    # Careful of residuals
    orig_offset = np.array(offset)
    offset = np.round(orig_offset).astype(int)
    residual = orig_offset - offset.astype(orig_offset.dtype)

    if not np.all(np.abs(residual) < 1e-6):
        import warnings
        warnings.warn('translating by rle, but offset is non-integer')

    # These are the flat indices where the value changes:
    #  * even locs are stop-indices for zeros and start indices for ones
    #  * odd locs are stop-indices for ones and start indices for zeros
    try:
        indices = rle['counts'].cumsum()
    except AttributeError:
        indices = np.array(rle['counts']).cumsum()

    if len(indices) % 2 == 1:
        indices = indices[:-1]

    # Transform indices to be start-stop inclusive indices for ones
    indices[1::2] -= 1

    # Find yx points where the binary mask changes value
    old_shape = np.array(rle['shape'])
    if output_shape is None:
        output_shape = old_shape
    new_shape = np.array(output_shape)
    rc_offset = np.array(offset[::-1])

    pts = np.unravel_index(indices, old_shape, order=rle['order'])
    major_axis = 1 if rle['order'] == 'F' else 0
    minor_axis = 1 - major_axis

    major_idxs = pts[major_axis]

    # Find locations in the major axis points where a non-zero count
    # crosses into the next minor dimension.
    pair_major_index = major_idxs.reshape(-1, 2)
    num_major_crossings = pair_major_index.T[1] - pair_major_index.T[0]
    flat_cross_idxs = np.where(num_major_crossings > 0)[0] * 2

    # Insert breaks to runs in locations that cross the major-axis.
    # This will force all runs to exist only within a single major-axis.
    # IE: Ensure points don't span multiple scanlines
    scanline_pts = [x.tolist() for x in pts]
    for idx in flat_cross_idxs[::-1]:
        prev_pt = [x[idx] for x in scanline_pts]
        next_pt = [x[idx + 1] for x in scanline_pts]
        for break_d in reversed(range(prev_pt[major_axis], next_pt[major_axis])):
            # Insert a breakpoint over every major axis crossing
            if major_axis == 1:
                new_stop = [old_shape[0] - 1, break_d]
                new_start = [0, break_d + 1]
            elif major_axis == 0:
                new_stop = [break_d, old_shape[1] - 1]
                new_start = [break_d + 1, 0]
            else:
                raise AssertionError(major_axis)
            scanline_pts[0].insert(idx + 1, new_start[0])
            scanline_pts[1].insert(idx + 1, new_start[1])
            scanline_pts[0].insert(idx + 1, new_stop[0])
            scanline_pts[1].insert(idx + 1, new_stop[1])

    # Now that new start-stop locations have been added,
    # translate the points that indices where non-zero data should go.
    new_pts = np.array(scanline_pts, dtype=int) + rc_offset[:, None]

    # <handle_out_of_bounds>
    # Note: all of the following logic relies on the fact that each run can
    # only span one major axis. This condition is true because we inserted
    # breakpoints whenever a run spanned more than one major axis.

    new_major_dim = new_shape[major_axis]
    new_minor_dim = new_shape[minor_axis]

    # Only keep points where the major axis is in bounds
    _new_major_pts = new_pts[major_axis]
    is_major_ib = ((_new_major_pts >= 0) &
                   (_new_major_pts < new_major_dim))
    # assert np.all(is_major_ib[0::2] == is_major_ib[1::2]), (
    #     'all pairs should be both in-bounds or both out-of-bounds')
    new_pts = new_pts.T[is_major_ib].T
    new_pts = np.ascontiguousarray(new_pts)

    # Now remove any points where the minor axis is OOB in the same direction.
    # (i.e. remove pairs of points that are both left-oob or both right-oob,
    # but dont remove pairs where only one is left-oob or right-oob, because
    # these still create structure in our new image.)
    _new_minor_pts = new_pts[minor_axis]
    is_left_oob = (_new_minor_pts < 0)
    is_right_oob = (_new_minor_pts >= new_minor_dim)
    is_pair_left_oob = (is_left_oob[0::2] & is_left_oob[1::2])
    is_pair_right_oob = (is_right_oob[0::2] & is_right_oob[1::2])
    is_pair_removable = (is_pair_left_oob | is_pair_right_oob)
    new_pts_pairs = new_pts.T.reshape(-1, 2, 2)
    new_pts = new_pts_pairs[~is_pair_removable].reshape(-1, 2).T
    new_pts = np.ascontiguousarray(new_pts)

    # Finally, all new points are strictly within the existing major dims and
    # we have removed any point pair where both pairs were oob in the same
    # direction, we can simply clip any regions along the minor axis that go
    # out of bounds.
    _new_minor_pts = new_pts[minor_axis]
    _new_minor_pts.clip(0, new_minor_dim - 1, out=_new_minor_pts)
    # </handle_out_of_bounds>

    # Now we have translated flat-indices in the new canvas shape
    new_indices = np.ravel_multi_index(new_pts, new_shape, order=rle['order'])
    new_indices[1::2] += 1

    count_dtype = int  # use in to eventually support non-binary RLE
    new_indices = new_indices.astype(count_dtype)

    total = int(np.prod(new_shape))
    if len(new_indices) == 0:
        trailing_counts = np.array([total], dtype=count_dtype)
        leading_counts = np.array([], dtype=count_dtype)
    else:
        leading_counts = np.array([new_indices[0]], dtype=count_dtype)
        trailing_counts = np.array([total - new_indices[-1]], dtype=count_dtype)

    body_counts = np.diff(new_indices)
    new_counts = np.hstack([leading_counts, body_counts, trailing_counts])

    new_rle = {
        'shape': tuple(new_shape.tolist()),
        'order': rle['order'],
        'counts': new_counts,
        'binary': rle['binary'],
    }
    return new_rle


def _rle_bytes_to_array(s, impl='auto'):
    """
    Uncompresses a coco-bytes RLE into an array representation.

    Args:
        s (bytes): compressed coco bytes rle
        impl (str): which implementation to use (defaults to cython is possible)

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/im_runlen.py _rle_bytes_to_array

    Benchmark:
        >>> import ubelt as ub
        >>> from kwimage.im_runlen import _rle_bytes_to_array
        >>> s = b';?1B10O30O4'
        >>> ti = ub.Timerit(1000, bestof=50, verbose=2)
        >>> # --- time python impl ---
        >>> for timer in ti.reset('python'):
        >>>     with timer:
        >>>         _rle_bytes_to_array(s, impl='python')
        >>> # --- time cython impl ---
        >>> # xdoctest: +REQUIRES(--mask)
        >>> for timer in ti.reset('cython'):
        >>>     with timer:
        >>>         _rle_bytes_to_array(s, impl='cython')
    """
    # verbatim inefficient impl.
    # It would be nice if this (un/)compression algo could get a better
    # description.
    from kwimage.structs.mask import _backends
    key, cython_mask = _backends.get_backend(['kwimage'])

    if impl == 'auto':
        if cython_mask is None:
            impl = 'python'
        else:
            impl = 'cython'
    if impl == 'python':
        import numpy as np
        cnts = np.empty(len(s), dtype=np.int64)
        p = 0
        m = 0
        for m in range(len(s)):
            if p >= len(s):
                break
            x = 0
            k = 0
            more = 1
            while more:
                # c = s[p] - 48
                c = s[p] - 48
                x |= (c & 0x1f) << 5 * k
                more = c & 0x20
                p += 1
                k += 1
                if more == 0 and (c & 0x10):
                    x |= (-1 << 5 * k)
            if m > 2:
                x += cnts[m - 2]
            cnts[m] = x
        cnts = cnts[:m]
        return cnts
    elif impl == 'cython':
        return cython_mask._rle_bytes_to_array(s)


def _rle_array_to_bytes(counts, impl='auto'):
    """
    Compresses an array RLE into a coco-bytes RLE.

    Args:
        counts (ndarray): uncompressed array rle
        impl (str): which implementation to use (defaults to cython is possible)

    Example:
        >>> # xdoctest: +REQUIRES(--mask)
        >>> from kwimage.im_runlen import _rle_array_to_bytes
        >>> from kwimage.im_runlen import _rle_bytes_to_array
        >>> arr_counts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> str_counts = _rle_array_to_bytes(arr_counts)
        >>> arr_counts2 = _rle_bytes_to_array(str_counts)
        >>> assert np.all(arr_counts2 == arr_counts)

    Benchmark:
        >>> # xdoctest: +REQUIRES(--mask)
        >>> import ubelt as ub
        >>> from kwimage.im_runlen import _rle_array_to_bytes
        >>> from kwimage.im_runlen import _rle_bytes_to_array
        >>> counts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> ti = ub.Timerit(1000, bestof=50, verbose=2)
        >>> # --- time python impl ---
        >>> #for timer in ti.reset('python'):
        >>> #    with timer:
        >>> #        _rle_array_to_bytes(s, impl='python')
        >>> # --- time cython impl ---
        >>> for timer in ti.reset('cython'):
        >>>     with timer:
        >>>         _rle_array_to_bytes(s, impl='cython')
    """
    # verbatim inefficient impl.
    # It would be nice if this (un/)compression algo could get a better
    # description.
    from kwimage.structs.mask import _backends
    key, cython_mask = _backends.get_backend(['kwimage'])
    if impl == 'auto':
        if cython_mask is None:
            impl = 'python'
        else:
            impl = 'cython'

    if impl == 'python':
        raise NotImplementedError('pure python rle is not available')
    elif impl == 'cython':
        counts = counts.astype(np.uint32)
        counts_str = cython_mask._rle_array_to_bytes(counts)
        return counts_str
    else:
        raise KeyError(impl)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.im_runlen
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
