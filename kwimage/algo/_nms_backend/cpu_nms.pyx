# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
NUMPY_INCLUDE=$(python -c "import numpy as np; print(np.get_include())")
CPATH=$CPATH:$NUMPY_INCLUDE cythonize -a -i ~/code/kwimage/kwimage/algo/_nms_backend/cpu_nms.pyx

python -c "from kwimage.algo._nms_backend import cpu_nms"

References:
    https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.pyx
"""
from __future__ import absolute_import

import numpy as np
cimport numpy as np
cimport cython

# cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
#     return a if a >= b else b

# cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
#     return a if a <= b else b

cdef inline float max_(float a, float b) nogil:
    return a if a >= b else b

cdef inline float min_(float a, float b) nogil:
    return a if a <= b else b


ctypedef Py_ssize_t SIZE_T
SIZE_T_DTYPE = np.intp


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def cpu_nms(np.ndarray[np.float32_t, ndim=2] ltrb,
            np.ndarray[np.float32_t, ndim=1] scores,
            np.float thresh, np.float bias=0.0):
    """
    Fast nonmax supression implementation on the CPU using cython

    SeeAlso:
        ~/code/kwimage/dev/bench_nms.py

    Example:
        >>> from kwimage.algo._nms_backend.cpu_nms import cpu_nms
        >>> ltrb = np.array([
        >>>     [0, 0, 10, 10],
        >>>     [0, 0, 10, 10],
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .3, .4, .5], dtype=np.float32)
        >>> thresh = .5
        >>> bias = 0.0
        >>> cpu_nms(ltrb, scores, thresh, bias)

        >>> from kwimage.algo._nms_backend.cpu_nms import cpu_nms
        >>> ltrb = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [0, 0, 0, 0],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .3], dtype=np.float32)
        >>> thresh = .5
        >>> bias = 0.0
        >>> cpu_nms(ltrb, scores, thresh, bias)
    """
    cdef int n_boxes = ltrb.shape[0]

    cdef np.ndarray[np.float32_t, ndim=1] x1 = ltrb[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = ltrb[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = ltrb[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = ltrb[:, 3]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + bias) * (y2 - y1 + bias)
    cdef np.ndarray[SIZE_T, ndim=1] order = scores.argsort()[::-1].astype(SIZE_T_DTYPE)

    cdef np.ndarray[np.int32_t, ndim=1] suppressed = np.zeros(n_boxes, dtype=np.int32)

    # nominal indices
    cdef SIZE_T _i, _j
    # sorted indices
    cdef SIZE_T i, j
    # temp variables for box i's (the box currently under consideration)

    # cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    # cdef np.float32_t xx1, yy1, xx2, yy2
    # cdef np.float32_t w, h
    # cdef np.float32_t inter, ovr
    cdef float ix1, iy1, ix2, iy2, iarea
    cdef float xx1, yy1, xx2, yy2
    cdef float w, h
    cdef float inter, ovr

    cdef float[:] x1_view = x1
    cdef float[:] y1_view = y1
    cdef float[:] x2_view = x2
    cdef float[:] y2_view = y2
    cdef float[:] areas_view = areas
    cdef SIZE_T[:] order_view = order
    cdef int[:] suppressed_view = suppressed

    cdef float _thresh = thresh
    cdef float _bias = bias

    # keep = []
    cdef np.ndarray[SIZE_T, ndim=1] keep_idxs = np.empty(n_boxes, dtype=SIZE_T_DTYPE)
    cdef SIZE_T[:] keep_idxs_view = keep_idxs
    cdef SIZE_T _num_kept = 0

    with nogil:
        for _i in range(n_boxes):
            # Look at detection in order of descinding score
            i = order_view[_i]

            # If this detection was not supressed, we will keep it and then supress
            # anything it conflicts with
            if suppressed_view[i] == 0:

                # keep.append(i)

                # Alternative to append
                keep_idxs_view[_num_kept] = i
                _num_kept += 1

                ix1 = x1_view[i]
                iy1 = y1_view[i]
                ix2 = x2_view[i]
                iy2 = y2_view[i]
                iarea = areas_view[i]

                # Look at the other unsupressed detections
                for _j in range(_i + 1, n_boxes):
                    j = order_view[_j]
                    if suppressed_view[j] == 0:
                        xx1 = max_(ix1, x1_view[j])
                        yy1 = max_(iy1, y1_view[j])
                        xx2 = min_(ix2, x2_view[j])
                        yy2 = min_(iy2, y2_view[j])
                        w = max_(0.0, xx2 - xx1 + _bias)
                        h = max_(0.0, yy2 - yy1 + _bias)
                        # Supress any other detection that overlaps with the i-th
                        # detection, which we just kept.
                        inter = w * h
                        ovr = inter / (iarea + areas_view[j] - inter)
                        # NOTE: We are using following convention:
                        #     * suppress if overlap > thresh
                        #     * consider if overlap <= thresh
                        # This convention has the property that when thresh=0, we dont just
                        # remove everything.
                        if ovr > _thresh:
                            suppressed_view[j] = 1

    # Convert back to a list
    keep = keep_idxs[:_num_kept].tolist()
    return keep
