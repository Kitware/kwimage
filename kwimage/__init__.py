"""
mkinit ~/code/kwimage/kwimage/algo/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/structs/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/__init__.py --relative -w --nomod
"""

__version__ = '0.5.7'

from .algo import (available_nms_impls, daq_spatial_nms, non_max_supression,)
from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                            overlay_alpha_layers,)
from .im_color import (BASE_COLORS, CSS4_COLORS, Color, TABLEAU_COLORS,
                       XKCD_COLORS,)
from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                      make_channels_comparable, num_channels,)
from .im_cv2 import (convert_colorspace, gaussian_patch, imresize, imscale,)
from .im_demodata import (grab_test_image, grab_test_image_fpath,)
from .im_draw import (draw_boxes_on_image, draw_clf_on_image,
                      draw_text_on_image, make_heatmask, make_orimask,
                      make_vector_field,)
from .im_io import (imread, imwrite,)
from .im_runlen import (decode_run_length, encode_run_length, rle_translate,)
from .im_stack import (stack_images, stack_images_grid,)
from .structs import (Boxes, Coords, Detections, Heatmap, Mask, MaskList,
                      MultiPolygon, Points, PointsList, Polygon, PolygonList,
                      smooth_prob,)
from .util_warp import (TORCH_GRID_SAMPLE_HAS_ALIGN, subpixel_accum,
                        subpixel_align, subpixel_getvalue, subpixel_maximum,
                        subpixel_minimum, subpixel_set, subpixel_setvalue,
                        subpixel_slice, subpixel_translate, warp_points,
                        warp_tensor,)

__all__ = ['BASE_COLORS', 'Boxes', 'CSS4_COLORS', 'Color', 'Coords',
           'Detections', 'Heatmap', 'Mask', 'MaskList', 'MultiPolygon',
           'Points', 'PointsList', 'Polygon', 'PolygonList', 'TABLEAU_COLORS',
           'TORCH_GRID_SAMPLE_HAS_ALIGN', 'XKCD_COLORS', 'atleast_3channels',
           'available_nms_impls', 'convert_colorspace', 'daq_spatial_nms',
           'decode_run_length', 'draw_boxes_on_image', 'draw_clf_on_image',
           'draw_text_on_image', 'encode_run_length', 'ensure_alpha_channel',
           'ensure_float01', 'ensure_uint255', 'gaussian_patch',
           'grab_test_image', 'grab_test_image_fpath', 'imread', 'imresize',
           'imscale', 'imwrite', 'make_channels_comparable', 'make_heatmask',
           'make_orimask', 'make_vector_field', 'non_max_supression',
           'num_channels', 'overlay_alpha_images', 'overlay_alpha_layers',
           'rle_translate', 'smooth_prob', 'stack_images', 'stack_images_grid',
           'subpixel_accum', 'subpixel_align', 'subpixel_getvalue',
           'subpixel_maximum', 'subpixel_minimum', 'subpixel_set',
           'subpixel_setvalue', 'subpixel_slice', 'subpixel_translate',
           'warp_points', 'warp_tensor']
