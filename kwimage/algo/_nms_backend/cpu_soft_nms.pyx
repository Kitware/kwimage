"""
cythonize -a -i ~/code/kwimage/kwimage/algo/_nms_backend/cpu_soft_nms.pyx

python -c "
import numpy as np
from kwimage.algo._nms_backend import cpu_soft_nms
tlbr = np.array([[0, 0, 100, 100], [100, 100, 10, 10]], dtype=np.float32)
scores = np.array([.1, .2], dtype=np.float32)
keep = cpu_soft_nms.soft_nms(tlbr, scores, thresh=.1)
print(keep)
"""
# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------
cimport cython
cimport numpy as np
from libc.math cimport exp
import numpy as np


SIZE_T_DTYPE = np.intp
ctypedef Py_ssize_t SIZE_T


cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b


# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# Modified version from Detectron
# ----------------------------------------------------------
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def soft_nms(np.ndarray[np.float32_t, ndim=2] tlbr, 
             np.ndarray[np.float32_t, ndim=1] scores, 
             float thresh=0.001,
             float overlap_thresh=0.3,
             float sigma=0.5,
             float bias=0.0,
             unsigned int method=0):
    cdef float iw, ih, box_area
    cdef float ua
    cdef float maxscore = 0

    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov, s

    cdef SIZE_T i = 0
    cdef SIZE_T pos = 0
    cdef SIZE_T N = tlbr.shape[0]

    cdef SIZE_T ti

    cdef float[:, :] bbox_view = tlbr
    cdef float[:] score_view = scores

    cdef np.ndarray[SIZE_T, ndim=1] inds = np.arange(N, dtype=SIZE_T_DTYPE)
    cdef SIZE_T[:] inds_view = inds

    with nogil:
        for i in range(N):
            maxscore = score_view[i]
            maxpos = i

            tx1 = bbox_view[i, 0]
            ty1 = bbox_view[i, 1]
            tx2 = bbox_view[i, 2]
            ty2 = bbox_view[i, 3]

            ts = score_view[i]
            ti = inds_view[i]

            pos = i + 1
            # get max box
            while pos < N:
                if maxscore < score_view[pos]:
                    maxscore = score_view[pos]
                    maxpos = pos
                pos = pos + 1

            # add max box as a detection
            bbox_view[i, 0] = bbox_view[maxpos, 0]
            bbox_view[i, 1] = bbox_view[maxpos, 1]
            bbox_view[i, 2] = bbox_view[maxpos, 2]
            bbox_view[i, 3] = bbox_view[maxpos, 3]

            score_view[i] = score_view[maxpos]
            inds_view[i] = inds_view[maxpos]

            # swap ith box with position of max box
            bbox_view[maxpos, 0] = tx1
            bbox_view[maxpos, 1] = ty1
            bbox_view[maxpos, 2] = tx2
            bbox_view[maxpos, 3] = ty2

            score_view[maxpos] = ts
            inds_view[maxpos] = ti

            tx1 = bbox_view[i, 0]
            ty1 = bbox_view[i, 1]
            tx2 = bbox_view[i, 2]
            ty2 = bbox_view[i, 3]

            ts = score_view[i]

            pos = i + 1
            # NMS iterations, note that N changes if detection bbox_view fall
            # below thresh
            while pos < N:
                x1 = bbox_view[pos, 0]
                y1 = bbox_view[pos, 1]
                x2 = bbox_view[pos, 2]
                y2 = bbox_view[pos, 3]

                s = score_view[pos]

                area = (x2 - x1 + bias) * (y2 - y1 + bias)
                iw = (min(tx2, x2) - max(tx1, x1) + bias)
                if iw > 0:
                    ih = (min(ty2, y2) - max(ty1, y1) + bias)
                    if ih > 0:
                        ua = float((tx2 - tx1 + bias) *
                                   (ty2 - ty1 + bias) + area - iw * ih)
                        ov = iw * ih / ua  # iou between max box and detection box

                        if method == 1:  # linear
                            if ov > overlap_thresh:
                                weight = 1 - ov
                            else:
                                weight = 1
                        elif method == 2:  # gaussian
                            weight = exp(-(ov * ov) / sigma)
                        else:  # original NMS
                            if ov > overlap_thresh:
                                weight = 0
                            else:
                                weight = 1

                        score_view[pos] = weight * score_view[pos]

                        # if box score falls below thresh, discard the box by
                        # swapping with last box update N
                        if score_view[pos] < thresh:
                            bbox_view[pos, 0] = bbox_view[N - 1, 0]
                            bbox_view[pos, 1] = bbox_view[N - 1, 1]
                            bbox_view[pos, 2] = bbox_view[N - 1, 2]
                            bbox_view[pos, 3] = bbox_view[N - 1, 3]
                            score_view[pos] = score_view[N - 1]
                            inds_view[pos] = inds_view[N - 1]
                            N = N - 1
                            pos = pos - 1

                pos = pos + 1
    return inds[:N]
