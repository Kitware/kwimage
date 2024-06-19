import numpy as np
import cv2

flags = cv2.INTER_LINEAR
borderMode = cv2.BORDER_CONSTANT
borderValue = np.nan
borderValue = [np.nan, np.nan, np.nan, np.nan]

img = np.random.rand(32, 32, 10)
M = np.array([[ 0.92387953, -0.38268343,  0.        ],
              [ 0.38268343,  0.92387953,  0.        ],
              [ 0.        ,  0.        ,  1.        ]])
dsize = (img.shape[1], img.shape[0])

warped = cv2.warpAffine(img, M[0:2], dsize=dsize, flags=flags,
                        borderMode=borderMode, borderValue=borderValue)


import kwimage
warped_canvas = kwimage.fill_nans_with_checkers(warped)
import kwplot
kwplot.autompl()
kwplot.imshow(img, fnum=1, pnum=(1, 2, 1))
kwplot.imshow(warped_canvas, fnum=1, pnum=(1, 2, 2))
