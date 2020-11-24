"""
mkinit ~/code/kwimage/kwimage/algo/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/structs/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomod  -w
"""

__version__ = '0.6.10'

from .algo import (available_nms_impls, daq_spatial_nms, non_max_supression,)
from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                            overlay_alpha_layers,)
from .im_color import (BASE_COLORS, CSS4_COLORS, Color, TABLEAU_COLORS,
                       XKCD_COLORS,)
from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                      make_channels_comparable, normalize, num_channels,)
from .im_cv2 import (convert_colorspace, gaussian_patch, imresize, imscale,)
from .im_demodata import (grab_test_image, grab_test_image_fpath,)
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
from .util_warp import (TORCH_GRID_SAMPLE_HAS_ALIGN, add_homog, remove_homog,
                        subpixel_accum, subpixel_align, subpixel_getvalue,
                        subpixel_maximum, subpixel_minimum, subpixel_set,
                        subpixel_setvalue, subpixel_slice, subpixel_translate,
                        warp_image, warp_points, warp_tensor,)

__all__ = ['BASE_COLORS', 'Boxes', 'CSS4_COLORS', 'Color', 'Coords',
           'Detections', 'Heatmap', 'Mask', 'MaskList', 'MultiPolygon',
           'Points', 'PointsList', 'Polygon', 'PolygonList', 'Segmentation',
           'SegmentationList', 'TABLEAU_COLORS', 'TORCH_GRID_SAMPLE_HAS_ALIGN',
           'XKCD_COLORS', 'add_homog', 'atleast_3channels',
           'available_nms_impls', 'convert_colorspace', 'daq_spatial_nms',
           'decode_run_length', 'draw_boxes_on_image', 'draw_clf_on_image',
           'draw_line_segments_on_image', 'draw_text_on_image',
           'draw_vector_field', 'encode_run_length', 'ensure_alpha_channel',
           'ensure_float01', 'ensure_uint255', 'fourier_mask',
           'gaussian_patch', 'grab_test_image', 'grab_test_image_fpath',
           'imread', 'imresize', 'imscale', 'imwrite', 'load_image_shape',
           'make_channels_comparable', 'make_heatmask', 'make_orimask',
           'make_vector_field', 'non_max_supression', 'normalize',
           'num_channels', 'overlay_alpha_images', 'overlay_alpha_layers',
           'radial_fourier_mask', 'remove_homog', 'rle_translate',
           'smooth_prob', 'stack_images', 'stack_images_grid',
           'subpixel_accum', 'subpixel_align', 'subpixel_getvalue',
           'subpixel_maximum', 'subpixel_minimum', 'subpixel_set',
           'subpixel_setvalue', 'subpixel_slice', 'subpixel_translate',
           'warp_image', 'warp_points', 'warp_tensor']
