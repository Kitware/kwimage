"""
The Kitware Image Module (kwimage) contains functions to accomplish lower-level
image operations via a high level API.

+------------------+-------------------------------------------------------+
| Read the docs    | https://kwimage.readthedocs.io                        |
+------------------+-------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/kwimage    |
+------------------+-------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/kwimage                    |
+------------------+-------------------------------------------------------+
| Pypi             | https://pypi.org/project/kwimage                      |
+------------------+-------------------------------------------------------+

Module features:

    * Image reader / writer functions with multiple backends

    * Wrapers around opencv that simplify and extend its functionality

    * Annotation datastructure with configurable backends.

    * Many function have awareness of torch tensors and can be used
      interchangably with ndarrays.

    * Misc image manipulation functions

"""

__devnotes__ = """
mkinit ~/code/kwimage/kwimage/algo/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/structs/__init__.py --relative -w --nomod
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomod  -w --lazy-loader
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomod  --diff --lazy-loader
"""

import os
if os.environ.get('REQUIRE_CV2', ''):
    try:
        if not os.environ.get('_ARGCOMPLETE', ''):
            import cv2  # NOQA
    except ImportError as ex:
        import ubelt as ub
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
            `pip install opencv-python-headless` xor
            `pip install opencv-python`.

            We appologize for this issue and hope this documentation is sufficient.

            orig_ex={!r}
            ''').format(ex)
        raise ImportError(msg)

__ignore__ = [
    'BASE_COLORS', 'XKCD_COLORS', 'TABLEAU_COLORS', 'CSS4_COLORS',
    'TORCH_GRID_SAMPLE_HAS_ALIGN',
]

__version__ = '0.11.2'

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={},
    submod_attrs={
        'algo': [
            'available_nms_impls',
            'daq_spatial_nms',
            'non_max_supression',
        ],
        'im_alphablend': [
            'ensure_alpha_channel',
            'overlay_alpha_images',
            'overlay_alpha_layers',
        ],
        'im_color': [
            'Color',
        ],
        'im_core': [
            'atleast_3channels',
            'ensure_float01',
            'ensure_uint255',
            'exactly_1channel',
            'find_robust_normalizers',
            'make_channels_comparable',
            'normalize',
            'normalize_intensity',
            'num_channels',
            'padded_slice',
        ],
        'im_cv2': [
            'connected_components',
            'convert_colorspace',
            'gaussian_blur',
            'gaussian_patch',
            'imcrop',
            'imscale',
            'morphology',
        ],
        'im_demodata': [
            'checkerboard',
            'grab_test_image',
            'grab_test_image_fpath',
        ],
        'im_draw': [
            'draw_boxes_on_image',
            'draw_clf_on_image',
            'draw_header_text',
            'draw_line_segments_on_image',
            'draw_text_on_image',
            'draw_vector_field',
            'fill_nans_with_checkers',
            'make_heatmask',
            'make_orimask',
            'make_vector_field',
            'nodata_checkerboard',
        ],
        'im_filter': [
            'fourier_mask',
            'radial_fourier_mask',
        ],
        'im_io': [
            'imread',
            'imwrite',
            'load_image_shape',
        ],
        'im_runlen': [
            'decode_run_length',
            'encode_run_length',
            'rle_translate',
        ],
        'im_stack': [
            'stack_images',
            'stack_images_grid',
        ],
        'im_transform': [
            'warp_affine',
            'warp_image',
            'warp_projective',
            'imresize',
        ],
        'structs': [
            'Box',
            'Boxes',
            'Coords',
            'Detections',
            'Heatmap',
            'Mask',
            'MaskList',
            'MultiPolygon',
            'Points',
            'PointsList',
            'Polygon',
            'PolygonList',
            'Segmentation',
            'SegmentationList',
            'smooth_prob',
        ],
        'transform': [
            'Affine',
            'Linear',
            'Matrix',
            'Projective',
            'Transform',
        ],
        'util_warp': [
            'add_homog',
            'remove_homog',
            'subpixel_accum',
            'subpixel_align',
            'subpixel_getvalue',
            'subpixel_maximum',
            'subpixel_minimum',
            'subpixel_set',
            'subpixel_setvalue',
            'subpixel_slice',
            'subpixel_translate',
            'warp_points',
            'warp_tensor',
        ],
    },
)

__all__ = ['Affine', 'Box', 'Boxes', 'Color', 'Coords', 'Detections',
           'Heatmap', 'Linear', 'Mask', 'MaskList', 'Matrix', 'MultiPolygon',
           'Points', 'PointsList', 'Polygon', 'PolygonList', 'Projective',
           'Segmentation', 'SegmentationList', 'Transform', 'add_homog',
           'atleast_3channels', 'available_nms_impls', 'checkerboard',
           'connected_components', 'convert_colorspace', 'daq_spatial_nms',
           'decode_run_length', 'draw_boxes_on_image', 'draw_clf_on_image',
           'draw_header_text', 'draw_line_segments_on_image',
           'draw_text_on_image', 'draw_vector_field', 'encode_run_length',
           'ensure_alpha_channel', 'ensure_float01', 'ensure_uint255',
           'exactly_1channel', 'fill_nans_with_checkers',
           'find_robust_normalizers', 'fourier_mask', 'gaussian_blur',
           'gaussian_patch', 'grab_test_image', 'grab_test_image_fpath',
           'imcrop', 'imread', 'imresize', 'imscale', 'imwrite',
           'load_image_shape', 'make_channels_comparable', 'make_heatmask',
           'make_orimask', 'make_vector_field', 'morphology',
           'nodata_checkerboard', 'non_max_supression', 'normalize',
           'normalize_intensity', 'num_channels', 'overlay_alpha_images',
           'overlay_alpha_layers', 'padded_slice', 'radial_fourier_mask',
           'remove_homog', 'rle_translate', 'smooth_prob', 'stack_images',
           'stack_images_grid', 'subpixel_accum', 'subpixel_align',
           'subpixel_getvalue', 'subpixel_maximum', 'subpixel_minimum',
           'subpixel_set', 'subpixel_setvalue', 'subpixel_slice',
           'subpixel_translate', 'warp_affine', 'warp_image', 'warp_points',
           'warp_projective', 'warp_tensor']
