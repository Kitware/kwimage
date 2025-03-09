"""
Fast R-CNN
Copyright (c) 2015 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ross Girshick
"""
import numpy as np
import warnings


def py_nms(np_ltrb, np_scores, thresh, bias=1):
    """
    Pure Python NMS baseline.

    References:
        https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py

    CommandLine:
        xdoctest -m kwimage.algo._nms_backend.py_nms py_nms

    Example:
        >>> from kwimage.algo._nms_backend.py_nms import *  # NOQA
        >>> np_ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> np_scores = np.linspace(0, 1, len(np_ltrb))
        >>> thresh = 0.1
        >>> bias = 0.0
        >>> keep = sorted(map(int, py_nms(np_ltrb, np_scores, thresh, bias)))
        >>> print('keep = {!r}'.format(keep))
        keep = [2, 4, 5, 7]

    Example:
        >>> from kwimage.algo._nms_backend.py_nms import *  # NOQA
        >>> np_ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32)
        >>> np_scores = np.array([.1, .5, .9, .1])
        >>> keep = list(map(int, py_nms(np_ltrb, np_scores, thresh=0.0, bias=1.0)))
        >>> print('keep@0.0 = {!r}'.format(keep))
        >>> keep = list(map(int, py_nms(np_ltrb, np_scores, thresh=0.2, bias=1.0)))
        >>> print('keep@0.2 = {!r}'.format(keep))
        >>> keep = list(map(int, py_nms(np_ltrb, np_scores, thresh=0.5, bias=1.0)))
        >>> print('keep@0.5 = {!r}'.format(keep))
        >>> keep = list(map(int, py_nms(np_ltrb, np_scores, thresh=1.0, bias=1.0)))
        >>> print('keep@1.0 = {!r}'.format(keep))
        keep@0.0 = [2, 1]
        keep@0.2 = [2, 1]
        keep@0.5 = [2, 1, 3]
        keep@1.0 = [2, 1, 3, 0]

    Example:
        >>> # Test int16 case
        >>> from kwimage.algo._nms_backend.py_nms import *  # NOQA
        >>> np_ltrb = np.array([
        >>>    [  71.75,  -37.25,  373.8 ,  231.2 ],
        >>>    [ 282.  ,  395.5 ,  609.  ,  639.5 ],
        >>>    [  36.62,   -5.5 ,  386.  ,  321.8 ],
        >>>    [ 207.5 ,  546.5 ,  238.6 ,  563.  ]], dtype=np.float16)
        >>> np_scores = np.linspace(0, 1, len(np_ltrb))
        >>> thresh = 0.1
        >>> bias = 0.0
        >>> keep = sorted(map(int, py_nms(np_ltrb, np_scores, thresh, bias)))
        >>> print('keep = {!r}'.format(keep))
    """
    new_dtype = np.result_type(np_ltrb, np.float32)
    if new_dtype != np_ltrb.dtype:
        # Upcast if precision is too low
        np_ltrb = np_ltrb.astype(new_dtype)

    x1 = np_ltrb[:, 0]
    y1 = np_ltrb[:, 1]
    x2 = np_ltrb[:, 2]
    y2 = np_ltrb[:, 3]

    widths = (x2 - x1 + bias)
    heights = (y2 - y1 + bias)
    areas = widths * heights

    idxs_remain = np_scores.argsort()[::-1]

    keep = []

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value .* true_divide')
        warnings.filterwarnings("error")

        # n_conflicts = 0
        while idxs_remain.size > 0:
            i = idxs_remain[0]
            keep.append(i)
            # print('Keeping the i={}-th box'.format(i))

            idxs_remain = idxs_remain[1:]
            # print('Compute IoU between chosen and remaining boxes')
            xx1 = np.maximum(x1[i], x1[idxs_remain])
            yy1 = np.maximum(y1[i], y1[idxs_remain])
            xx2 = np.minimum(x2[i], x2[idxs_remain])
            yy2 = np.minimum(y2[i], y2[idxs_remain])

            w = np.maximum(0.0, xx2 - xx1 + bias)
            h = np.maximum(0.0, yy2 - yy1 + bias)
            inter = w * h
            denom = (areas[i] + areas[idxs_remain] - inter)
            iou = inter / denom
            iou = np.nan_to_num(iou)
            # print('Checking overlap between the i={}-th and remaining boxes'.format(i))

            # Keep anything that doesnt have a large overlap with this item
            flags = iou <= thresh
            inds = np.where(flags)[0]
            # print('Supress {} boxes that did overlap this one'.format(len(ovr) - len(inds)))
            # print('Consider {} boxes that dont overlap this one'.format(len(inds)))
            idxs_remain = idxs_remain[inds]
    return keep


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.algo._nms_backend.py_nms
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
