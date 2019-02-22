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
    Python object interface to the C++ backend for encoding binary segmentation
    masks

    mask = cython_mask.Masks(10, 10, 1)
    """

    def __init__(self):
        pass

    @classmethod
    def demo(cls):
        from kwimage.structs._mask_backend import cython_mask
        # import kwimage
        import numpy as np

        # From string
        mask1 = np.array([
            [0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
        ], dtype=np.uint8)
        mask2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        mask3 = np.array([
            [0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0],
        ], dtype=np.uint8)

        # The cython utility expects multiple masks in fortran order with the
        # shape [H, W, N], where N is an index over multiple instances.
        masks = np.asfortranarray(mask[..., None])
        encoded = cython_mask.encode(masks)
        decoded = cython_mask.decode(encoded)
        decoded_mask1 = decoded[..., 0]

        cython_mask.toBbox(encoded)

        assert np.all(mask1 == decoded_mask1)

        iscrowd = [0 for _ in encoded]
        cython_mask.iou(encoded, encoded, iscrowd)

        print('rle_objs = {!r}'.format(rle_objs))

        encoded1 = [{'counts': b'ga0:f8=dGC\\7\\1VMj20000M30000000000000000001O0000000000001O0000000O100001O00000000000000000000000000000000000000000000000001O000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000O11O00000NaKWK_4i4200000000001O1O0000000000000O1000000000000nKcKU3]4gLhKX3X4bLQL[3W5I5K3[MeIS2]6eMiI[2d60000000000000000000000000000000000O10000000000000000000000001O1O00000O100000000000000000000000000000000000000lM`M\\M`2d2dMXM\\2h2gMUMY2k2jMRMV2n2nMmLS2S3QNbLAjN^2d4bNcKS2]4nM^KW2a4Z100000000000000000000001O000000000000000000O1000000000000000000000000000000000000000000O0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000O100000000000000000000000001N10000000000000000000000000000000000000000000000O1000O10000000000000000000000000000000000000000000000000000000000000000N2000000H;Gd0]L^JX2g6TN^>', 'size': [300, 416]}]
        decoded1 = cython_mask.decode(encoded1)

        # Looks like you can't have multiple different sizes
        encoded3 = [
            {'size': [10, 10], 'counts': b';>7FJO4'},
            {'size': [5, 9], 'counts': b';>7FJO4'},
        ]
        decoded3 = cython_mask.decode(encoded3)

        # shape, runlen = kwimage.encode_run_length(img, binary=True)
        # rle_str = ' '.join(map(str, runlen))
        # cython_mask._frString([rle_str])

        # From bounding boxes
        # rle_objs = cython_mask.frPyObjects(
        #     np.array([[0, 0, 10, 10.]]), h=10, w=10)
        # Rs = cython_mask._frString(rle_objs)
