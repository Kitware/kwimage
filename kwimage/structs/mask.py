"""
Structure for efficient encoding of per-annotation segmentation masks
Inspired by the cocoapi.

THIS IS CURRENTLY A WORK IN PROGRESS.

References:
    https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/mask.py
    https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocotools/_mask.pyx
    https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.c
    https://github.com/nightrome/cocostuffapi/blob/master/common/maskApi.h

"""

__ignore__ = True  # currently this file is a WIP


class Mask(object):
    """
    Python object interface to the C++ backend

    mask = cython_mask.Masks(10, 10, 1)
    """

    def __init__(self):
        pass

    @classmethod
    def demo(cls):
        from kwimage.structs._mask_backend import cython_mask
        import kwimage
        import numpy as np

        # From string
        img = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0]])
        shape, runlen = kwimage.encode_run_length(img, binary=True)
        rle_str = ' '.join(map(str, runlen))
        cython_mask._frString([rle_str], *shape)

        # From bounding boxes
        rle_objs = cython_mask.frPyObjects(
            np.array([[0, 0, 10, 10.]]), h=10, w=10)

        print('rle_objs = {!r}'.format(rle_objs))
