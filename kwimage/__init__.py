"""
mkinit ~/code/kwimage/kwimage/algo/__init__.py --relative -w
mkinit ~/code/kwimage/kwimage/structs/__init__.py --relative -w
mkinit ~/code/kwimage/kwimage/__init__.py --relative -w
"""

__version__ = '0.3.0.dev0'

from . import algo
from . import im_alphablend
from . import im_core
from . import im_cv2
from . import im_demodata
from . import im_io
from . import im_misc
from . import im_stack
from . import structs
from . import util_warp

from .algo import (algo_nms, available_nms_impls, daq_spatial_nms,
                   non_max_supression,)
from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                            overlay_alpha_layers,)
from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                      make_channels_comparable, num_channels,)
from .im_cv2 import (convert_colorspace, draw_boxes_on_image,
                     draw_text_on_image, gaussian_patch, imscale,)
from .im_demodata import (grab_test_image, grab_test_image_fpath,)
from .im_io import (imread, imwrite,)
from .im_misc import (decode_run_length, encode_run_length,)
from .im_stack import (stack_images, stack_images_grid,)
from .structs import (Boxes, Detections, Heatmap, Mask, boxes, detections,
                      heatmap, mask, smooth_prob,)
from .util_warp import (subpixel_accum, subpixel_align, subpixel_maximum,
                        subpixel_minimum, subpixel_slice, subpixel_translate,
                        warp_tensor,)

__all__ = ['Boxes', 'Detections', 'Heatmap', 'Mask', 'algo', 'algo_nms',
           'atleast_3channels', 'available_nms_impls', 'boxes',
           'convert_colorspace', 'daq_spatial_nms', 'decode_run_length',
           'detections', 'draw_boxes_on_image', 'draw_text_on_image',
           'encode_run_length', 'ensure_alpha_channel', 'ensure_float01',
           'ensure_uint255', 'gaussian_patch', 'grab_test_image',
           'grab_test_image_fpath', 'heatmap', 'im_alphablend', 'im_core',
           'im_cv2', 'im_demodata', 'im_io', 'im_misc', 'im_stack', 'imread',
           'imscale', 'imwrite', 'make_channels_comparable', 'mask',
           'non_max_supression', 'num_channels', 'overlay_alpha_images',
           'overlay_alpha_layers', 'smooth_prob', 'stack_images',
           'stack_images_grid', 'structs', 'subpixel_accum', 'subpixel_align',
           'subpixel_maximum', 'subpixel_minimum', 'subpixel_slice',
           'subpixel_translate', 'util_warp', 'warp_tensor']
