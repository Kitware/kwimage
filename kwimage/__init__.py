"""
The Kitware Image Module (kwimage) contains functions to accomplish lower-level
image operations via a high level API.
"""

__devnotes__ = """
mkinit ~/code/kwimage/kwimage/algo/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/structs/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomod  -w
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomod  --diff
"""

import ubelt as ub
try:
    import cv2  # NOQA
except ImportError as ex:
    msg = ub.paragraph(
        '''
        The kwimage module failed to import the cv2 module.  This may be due to
        an issue https://github.com/opencv/opencv-python/issues/467 which
        prevents us from marking cv2 as package dependency.
        To work around this we require that the user install this package with
        one of the following extras tags:
        `pip install kwimage[graphics]` xor
        `pip install kwimage[headless]`.

        Alternatively, the user can directly install the cv2 package as a post
        processing step via:
        `pip install opencv-python-headless` or
        `pip install opencv-python`.

        We appologize for this issue and hope this documentation is sufficient.

        orig_ex={!r}
        ''').format(ex)
    raise ImportError(msg)

__ignore__ = [
    'BASE_COLORS', 'XKCD_COLORS', 'TABLEAU_COLORS', 'CSS4_COLORS',
    'TORCH_GRID_SAMPLE_HAS_ALIGN',
]


__version__ = '0.7.7'

from .algo import (available_nms_impls, daq_spatial_nms, non_max_supression,)
from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                            overlay_alpha_layers,)
from .im_color import (Color,)
from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                      make_channels_comparable, normalize, num_channels,
                      padded_slice,)
from .im_cv2 import (convert_colorspace, gaussian_patch, imresize, imscale,
                     warp_affine,)
from .im_demodata import (checkerboard, grab_test_image,
                          grab_test_image_fpath,)
from .im_draw import (draw_boxes_on_image, draw_clf_on_image,
                      draw_line_segments_on_image, draw_text_on_image,
                      draw_vector_field, make_heatmask, make_orimask,
                      make_vector_field,)
from .im_filter import (fourier_mask, radial_fourier_mask,)
from .im_io import (imread, imwrite, load_image_shape,)
from .im_runlen import (decode_run_length, encode_run_length, rle_translate,)
from .im_stack import (stack_images, stack_images_grid,)
from .structs import (Boxes, Coords, Detections, Heatmap, Mask, MaskList,
                      MultiPolygon, Points, PointsList, Polygon, PolygonList,
                      Segmentation, SegmentationList, smooth_prob,)
from .transform import (Affine, Linear, Matrix, Projective, Transform,)
from .util_warp import (add_homog, remove_homog, subpixel_accum,
                        subpixel_align, subpixel_getvalue, subpixel_maximum,
                        subpixel_minimum, subpixel_set, subpixel_setvalue,
                        subpixel_slice, subpixel_translate, warp_image,
                        warp_points, warp_tensor,)

__all__ = ['Affine', 'Boxes', 'Color', 'Coords', 'Detections', 'Heatmap',
           'Linear', 'Mask', 'MaskList', 'Matrix', 'MultiPolygon', 'Points',
           'PointsList', 'Polygon', 'PolygonList', 'Projective',
           'Segmentation', 'SegmentationList', 'Transform', 'add_homog',
           'atleast_3channels', 'available_nms_impls', 'checkerboard',
           'convert_colorspace', 'daq_spatial_nms', 'decode_run_length',
           'draw_boxes_on_image', 'draw_clf_on_image',
           'draw_line_segments_on_image', 'draw_text_on_image',
           'draw_vector_field', 'encode_run_length', 'ensure_alpha_channel',
           'ensure_float01', 'ensure_uint255', 'fourier_mask',
           'gaussian_patch', 'grab_test_image', 'grab_test_image_fpath',
           'imread', 'imresize', 'imscale', 'imwrite', 'load_image_shape',
           'make_channels_comparable', 'make_heatmask', 'make_orimask',
           'make_vector_field', 'non_max_supression', 'normalize',
           'num_channels', 'overlay_alpha_images', 'overlay_alpha_layers',
           'padded_slice', 'radial_fourier_mask', 'remove_homog',
           'rle_translate', 'smooth_prob', 'stack_images', 'stack_images_grid',
           'subpixel_accum', 'subpixel_align', 'subpixel_getvalue',
           'subpixel_maximum', 'subpixel_minimum', 'subpixel_set',
           'subpixel_setvalue', 'subpixel_slice', 'subpixel_translate',
           'warp_affine', 'warp_image', 'warp_points', 'warp_tensor']
