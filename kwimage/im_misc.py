# -*- coding: utf-8 -*-
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
            runlen = np.hstack([[0], lengths])
        else:
            runlen = lengths
    else:
        runlen = np.hstack([values[:, None], lengths[:, None]]).ravel()
    encoding = {
        'shape': img.shape,
        'counts': runlen,
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
        ndarray: the image

    Example:
        >>> from kwimage.im_misc import *  # NOQA
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
    flat = recon.ravel()
    if binary:
        value = 0
        start = 0
        for num in counts:
            stop = start + num
            flat[start:stop] = value
            start = stop
            value = 1 - value
    else:
        start = 0
        for value, num in zip(counts[::2], counts[1::2]):
            stop = start + num
            flat[start:stop] = value
            start = stop
    return recon

if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.im_misc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
