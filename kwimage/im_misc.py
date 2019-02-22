# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def encode_run_length(img, binary=False):
    """
    Run length encoding.

    Args:
        img (ndarray): 2D image
        binary (bool, default=True): set to True for compatibility with COCO

    Returns:
        Tuple[Tuple[int, int], ndarray]: size and encoding

    TODO:
        - [ ]
        Fast cython implementation.
        Make RLE a data structure.
        See https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/_mask.pyx

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
        >>> (h, w), runlen = encode_run_length(img)
        >>> target = np.array([0,16,1,3,0,3,2,1,0,3,1,3,0,2,2,3,0,2,1,3,0,1,2,5,0,6,2,3,0,8,2,1,0,7])
        >>> assert np.all(target == runlen)

    Example:
        >>> binary = True
        >>> img = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0]])
        >>> (h, w), runlen = encode_run_length(img, binary=True)
        >>> assert runlen.tolist() == [0, 1, 1, 3, 2, 1, 1]
    """
    flat = img.ravel()
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
    shape = img.shape
    return shape, runlen


def decode_run_length(runlen, shape, binary=False):
    """
    Decode run length encoding back into an image.

    Args:
        runlen (ndarray): the encoding
        shape (Tuple[int, int]), the height / width of the mask
        binary (bool): if the RLU is binary or non-binary.
            Set to True for compatibility with COCO.

    Returns:
        ndarray: the image

    Example:
        >>> from kwimage.im_misc import *  # NOQA
        >>> binary = True
        >>> img = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0]])
        >>> shape, runlen = encode_run_length(img, binary=True)
        >>> recon = decode_run_length(runlen, shape, binary=True)
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
        >>> shape, runlen = encode_run_length(img)
        >>> recon = decode_run_length(runlen, shape, binary=False)
        >>> assert np.all(recon == img)
    """
    recon = np.zeros(shape, dtype=np.uint8)
    flat = recon.ravel()
    if binary:
        value = 0
        start = 0
        for num in runlen:
            stop = start + num
            flat[start:stop] = value
            start = stop
            value = 1 - value
    else:
        start = 0
        for value, num in zip(runlen[::2], runlen[1::2]):
            stop = start + num
            flat[start:stop] = value
            start = stop
    return recon
