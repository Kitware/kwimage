"""Static contracts for the public geometry API.

This module intentionally contains no runtime tests.  ``ty check kwimage
 tests/`` evaluates the TYPE_CHECKING block and ensures these public APIs do
not regress back to ``Any`` while the runtime test suite simply imports this
module.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numbers import Number
    from os import PathLike
    from collections.abc import Generator, Iterator, Mapping, Sequence
    from matplotlib.patches import PathPatch
    from shapely.geometry import MultiPoint
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon
    import torch
    from typing import Any, Literal, cast
    from typing_extensions import assert_type

    import affine
    import kwimage
    from kwimage._typing import (
        ArrayData, ImgAugBoundingBoxesOnImage, ImgAugKeypointsOnImage,
        RNGInput,
    )

    from kwimage.structs.detections import (
        CocoAnnotsDatasetLike, CocoDatasetLike, CocoDetection,
        DetectionDemoImageInfo, DetectionDemoSamplerLike,
        DetectionsCoerceKwargs, DetectionArray,
        DetectionClasses,
        DetectionDType,
        DetectionIndices,
        DetectionKeypoints,
        DetectionSegmentations,
    )
    from kwimage.structs.mask import (
        CocoMaskRLE, MaskArea, MaskData, MaskFormat, MaskFormatName,
    )
    from kwimage.structs.points import (
        CocoKeypointColumns, CocoKeypointDict, CocoKeypoints, PointClasses,
    )
    from kwimage.structs.segmentation import (
        SegmentationBackend, SegmentationCoco,
    )
    from kwimage.structs.single_box import BoxDType, BoxScalar
    from kwimage.structs.heatmap import (
        HeatmapColorMap, HeatmapImageDims, HeatmapInterpolation, HeatmapShape,
        HeatmapSpatialData, HeatmapTransform, HeatmapWarpMatrix,
    )
    from kwimage.structs.polygon import (
        CocoPolygon, CocoPolygonDict, CocoPolygonStyle, MultiPolygonGeoJSON,
        PolygonData, PolygonGeoJSON,
    )
    from kwimage.im_io import (
        ImageReadBackend, ImageShapeBackend, ImageWriteBackend,
    )
    from kwimage.im_core import (
        PaddedSliceInfo, PaddedSlicePadKw, RobustNormalizerInfo,
        RobustNormalizerParams, RobustNormalizerScalar,
    )
    from kwimage.im_cv2 import (
        ConnectedComponentsInfo, ConnectedComponentsStatsInfo,
    )
    from kwimage.im_transform import ResizeInfo, WarpInfo
    from kwimage.algo.algo_nms import NMSIndex, NMSIndices
    from kwimage.im_draw import (
        CV2LineKwargs, HeaderTextKwargs, TextDrawInfo, TextDrawKwargs,
    )
    from kwimage.im_runlen import RunLengthEncoding
    from kwimage.im_stack import StackTransform
    from kwimage.transform import (
        AffineConcise, AffineDecomposition, AffineRandomKwargs,
        AffineRandomParams, MatrixIndexResult, ProjectiveDecomposition,
        TransformScalar,
    )


    color = kwimage.Color.coerce('red', alpha=0.5, space='rgb')
    assert_type(color, kwimage.Color)
    distinct_colors = kwimage.Color.distinct(3)
    assert_type(
        distinct_colors, list[list[float] | tuple[float, ...]]
    )
    assert_type(
        kwimage.Color.distinct(2, existing=distinct_colors),
        list[list[float] | tuple[float, ...]],
    )
    assert_type(
        kwimage.Color.random(pool='rgb-uniform', rng=0), kwimage.Color
    )

    rng_input: RNGInput = 0
    assert_type(kwimage.Color.random(rng=rng_input), kwimage.Color)
    assert_type(kwimage.Boxes.random(rng=rng_input), kwimage.Boxes)
    assert_type(kwimage.Coords.random(rng=rng_input), kwimage.Coords)
    assert_type(kwimage.Points.random(rng=rng_input), kwimage.Points)
    assert_type(kwimage.Polygon.random(rng=rng_input), kwimage.Polygon)
    assert_type(kwimage.MultiPolygon.random(rng=rng_input), kwimage.MultiPolygon)
    assert_type(kwimage.PolygonList.random(rng=rng_input), kwimage.PolygonList)
    assert_type(kwimage.Mask.random(rng=rng_input), kwimage.Mask)
    assert_type(kwimage.Segmentation.random(rng=rng_input), kwimage.Segmentation)
    assert_type(kwimage.Heatmap.random(rng=rng_input), kwimage.Heatmap)
    assert_type(kwimage.Detections.random(rng=rng_input), kwimage.Detections)

    matrix = kwimage.Matrix.eye(3)
    assert_type(kwimage.Matrix.random(3, rng=0), kwimage.Matrix)
    assert_type(matrix.det(), TransformScalar)
    assert_type(matrix[0, 0], MatrixIndexResult)
    assert_type(matrix[0], MatrixIndexResult)

    affine_tf = kwimage.Affine(None)
    assert_type(affine_tf.det(), TransformScalar)
    assert_type(affine_tf.decompose(), AffineDecomposition)
    assert_type(affine_tf.concise(), AffineConcise)
    assert_type(kwimage.Affine.random_params(rng=0), AffineRandomParams)
    affine_random_kw: AffineRandomKwargs = {
        'scale': (0.5, 1.5),
        'offset': 0.0,
        'theta': (-0.2, 0.2),
    }
    assert_type(
        kwimage.Affine.random_params(rng=0, **affine_random_kw),
        AffineRandomParams,
    )
    assert_type(
        kwimage.Affine.random(rng=0, **affine_random_kw), kwimage.Affine
    )
    assert_type(affine_tf.to_affine(), affine.Affine)

    assert_type(
        kwimage.Projective.random(rng=0, **affine_random_kw),
        kwimage.Projective,
    )

    projective_tf = kwimage.Projective(None)
    assert_type(projective_tf.decompose(), ProjectiveDecomposition)

    boxes = kwimage.Boxes(np.zeros((2, 4), dtype=np.float32), 'ltrb')
    assert_type(boxes.dtype, np.dtype[Any] | torch.dtype)
    assert_type(boxes.device, torch.device | None)
    assert_type(boxes.to_imgaug((16, 20)), ImgAugBoundingBoxesOnImage)
    assert_type(kwimage.Boxes.from_imgaug(boxes.to_imgaug((16, 20))), kwimage.Boxes)
    assert_type(boxes.astype('float32'), kwimage.Boxes)
    assert_type(boxes.take([0]), kwimage.Boxes)
    assert_type(boxes.draw(), None)
    assert_type(boxes.tensor('cpu'), kwimage.Boxes)
    assert_type(boxes.to_tlbr(copy=False), kwimage.Boxes)
    assert_type(kwimage.Box.random(rng=rng_input), kwimage.Box)
    assert_type(
        kwimage.Box.random(num=1, scale=(10.0, 20.0), format='ltrb'),
        kwimage.Box,
    )
    assert_type(
        kwimage.Box.from_data([0.0, 1.0, 2.0, 3.0], 'xywh'),
        kwimage.Box,
    )

    opaque_box_data: object = [0.0, 1.0, 2.0, 3.0]
    assert_type(
        kwimage.Boxes.coerce(opaque_box_data, format='xywh'), kwimage.Boxes
    )
    assert_type(
        kwimage.Box.coerce(opaque_box_data, format='xywh'), kwimage.Box
    )

    image = np.zeros((16, 20, 3), dtype=np.uint8)
    binary = np.zeros((16, 20), dtype=np.uint8)

    read_backend: ImageReadBackend = 'pil'
    write_backend: ImageWriteBackend = 'cv2'
    assert_type(kwimage.imread('demo.png', backend=read_backend), np.ndarray)
    assert_type(
        kwimage.imwrite('demo.png', image, backend=write_backend), str
    )
    shape_backend: ImageShapeBackend = 'pil'
    assert_type(
        kwimage.load_image_shape(
            'demo.png', backend=shape_backend, include_channels=False
        ),
        tuple[int, int],
    )

    float_image = image.astype(np.float32) / 255.0
    fourier_mask_data = np.ones((16, 20), dtype=np.float32)
    assert_type(
        kwimage.fourier_mask(float_image, fourier_mask_data), np.ndarray
    )
    assert_type(kwimage.fourier_mask(float_image, 0), np.ndarray)
    assert_type(
        kwimage.radial_fourier_mask(float_image, radius=3), np.ndarray
    )

    assert_type(kwimage.checkerboard(dsize=(8, 8)), np.ndarray)
    assert_type(
        kwimage.grab_test_image('checkerboard', dsize=(8, None)),
        np.ndarray,
    )
    assert_type(
        kwimage.grab_test_image_fpath('astro'), str | PathLike[str]
    )

    assert_type(kwimage.ensure_alpha_channel(image), np.ndarray)
    assert_type(
        kwimage.overlay_alpha_images(image, image), np.ndarray
    )
    assert_type(
        kwimage.overlay_alpha_layers([image, image]), np.ndarray
    )

    assert_type(kwimage.stack_images([image, image]), np.ndarray)
    assert_type(
        kwimage.stack_images([image, image], return_info=True),
        tuple[np.ndarray, list[StackTransform]],
    )
    assert_type(kwimage.stack_images_grid([image, image]), np.ndarray)
    assert_type(
        kwimage.stack_images_grid([image, image], return_info=True),
        tuple[np.ndarray, list[StackTransform]],
    )

    encoded = kwimage.encode_run_length(binary, binary=True)
    assert_type(encoded, RunLengthEncoding)
    assert_type(kwimage.decode_run_length(**encoded), np.ndarray)
    assert_type(
        kwimage.rle_translate(encoded, (1, 2)), RunLengthEncoding
    )

    warp_pts_np = np.zeros((3, 2), dtype=np.float32)
    warp_mat_np = np.eye(3, dtype=np.float32)
    assert_type(kwimage.warp_points(warp_mat_np, warp_pts_np), np.ndarray)
    assert_type(
        kwimage.warp_points(warp_mat_np, warp_pts_np, homog_mode='keep'),
        np.ndarray,
    )
    assert_type(kwimage.add_homog(warp_pts_np), np.ndarray)
    assert_type(
        kwimage.remove_homog(np.zeros((3, 3), dtype=np.float32)),
        np.ndarray,
    )
    assert_type(
        kwimage.remove_homog(
            np.zeros((3, 3), dtype=np.float32), mode='drop'
        ),
        np.ndarray,
    )

    subpixel_np = np.zeros((5, 5), dtype=np.float32)
    subpixel_src_np = np.ones((2, 2), dtype=np.float32)
    subpixel_index = (slice(1, 3), slice(1, 3))
    assert_type(
        kwimage.subpixel_align(subpixel_np, 1.0, subpixel_index),
        tuple[np.ndarray, tuple[slice, ...]],
    )
    assert_type(
        kwimage.subpixel_align(subpixel_np, subpixel_src_np, subpixel_index),
        tuple[np.ndarray, tuple[slice, ...]],
    )
    assert_type(
        kwimage.subpixel_set(subpixel_np, subpixel_src_np, subpixel_index),
        np.ndarray,
    )
    assert_type(
        kwimage.subpixel_accum(subpixel_np, subpixel_src_np, subpixel_index),
        np.ndarray,
    )
    assert_type(
        kwimage.subpixel_maximum(
            subpixel_np, subpixel_src_np, subpixel_index
        ),
        np.ndarray,
    )
    assert_type(
        kwimage.subpixel_minimum(
            subpixel_np, subpixel_src_np, subpixel_index
        ),
        np.ndarray,
    )
    assert_type(
        kwimage.subpixel_slice(subpixel_np, subpixel_index), np.ndarray
    )
    assert_type(
        kwimage.subpixel_translate(subpixel_np, (0.5, -0.25)), np.ndarray
    )
    assert_type(
        kwimage.subpixel_translate(
            subpixel_np, np.array([0.5, -0.25]), interp_axes=(0, 1)
        ),
        np.ndarray,
    )
    sample_pts_np = np.array([[1.0, 1.0]], dtype=np.float32)
    assert_type(
        kwimage.subpixel_getvalue(subpixel_np, sample_pts_np), np.ndarray
    )
    assert_type(
        kwimage.subpixel_getvalue(
            subpixel_np, sample_pts_np, interp='nearest', bordermode='edge'
        ),
        np.ndarray,
    )
    assert_type(
        kwimage.subpixel_setvalue(subpixel_np, sample_pts_np, 0.0),
        np.ndarray,
    )

    warp_pts_torch = torch.zeros((3, 2), dtype=torch.float32)
    warp_mat_torch = torch.eye(3, dtype=torch.float32)
    assert_type(
        kwimage.warp_points(warp_mat_torch, warp_pts_torch), torch.Tensor
    )
    assert_type(kwimage.add_homog(warp_pts_torch), torch.Tensor)
    assert_type(
        kwimage.remove_homog(torch.zeros((3, 3))), torch.Tensor
    )
    subpixel_torch = torch.zeros((5, 5), dtype=torch.float32)
    subpixel_src_torch = torch.ones((2, 2), dtype=torch.float32)
    assert_type(
        kwimage.subpixel_translate(subpixel_torch, (0.5, -0.25)),
        torch.Tensor,
    )
    assert_type(
        kwimage.subpixel_slice(subpixel_torch, subpixel_index), torch.Tensor
    )
    assert_type(
        kwimage.warp_tensor(
            subpixel_torch[None, None], warp_mat_torch, (5, 5)
        ),
        torch.Tensor,
    )

    assert_type(kwimage.imread('demo.png'), np.ndarray)
    assert_type(kwimage.imwrite('demo.png', image), str)
    assert_type(
        kwimage.load_image_shape('demo.png'), tuple[int, int, int]
    )
    assert_type(
        kwimage.load_image_shape('demo.png', include_channels=False),
        tuple[int, int],
    )

    nms_ltrb = np.zeros((3, 4), dtype=np.float32)
    nms_scores = np.zeros(3, dtype=np.float32)
    assert_type(kwimage.available_nms_impls(), list[str])
    assert_type(
        kwimage.non_max_supression(nms_ltrb, nms_scores, 0.5),
        NMSIndices,
    )
    assert_type(
        kwimage.daq_spatial_nms(
            nms_ltrb, nms_scores, diameter=10, thresh=0.5
        ),
        list[NMSIndex],
    )

    text_draw_kw: TextDrawKwargs = {
        'color': 'red',
        'fontScale': 1.0,
        'halign': 'center',
        'valign': 'top',
        'border': {'color': 'black', 'thickness': 1},
    }
    assert_type(
        kwimage.draw_text_on_image(image, 'text', **text_draw_kw),
        np.ndarray,
    )
    assert_type(kwimage.draw_text_on_image(image, 'text'), np.ndarray)
    assert_type(
        kwimage.draw_text_on_image(image, 'text', return_info=True),
        tuple[np.ndarray, TextDrawInfo],
    )
    assert_type(
        kwimage.draw_clf_on_image(image, ['class']), np.ndarray
    )
    assert_type(
        kwimage.draw_boxes_on_image(
            image, np.empty((0, 4)), box_format='xywh'
        ),
        np.ndarray,
    )
    draw_pts = np.empty((0, 2), dtype=np.float32)
    assert_type(
        kwimage.draw_line_segments_on_image(image, draw_pts, draw_pts),
        np.ndarray,
    )
    line_draw_kw: CV2LineKwargs = {'lineType': 8, 'shift': 0}
    assert_type(
        kwimage.draw_line_segments_on_image(
            image, draw_pts, draw_pts, **line_draw_kw
        ),
        np.ndarray,
    )
    draw_field = np.zeros((16, 20), dtype=np.float32)
    assert_type(kwimage.make_heatmask(draw_field), np.ndarray)
    assert_type(kwimage.make_orimask(draw_field), np.ndarray)
    assert_type(
        kwimage.make_vector_field(draw_field, draw_field, alpha=False),
        np.ndarray,
    )
    assert_type(
        kwimage.draw_vector_field(
            image, draw_field, draw_field, alpha=False
        ),
        np.ndarray,
    )
    assert_type(
        kwimage.draw_header_text(image, 'header'), np.ndarray
    )
    header_kw: HeaderTextKwargs = {
        'fontScale': 1.0,
        'thickness': 1,
        'bg_value': 'black',
    }
    assert_type(
        kwimage.draw_header_text(
            image, 'header', fit='shrink', stack='auto', **header_kw
        ),
        np.ndarray,
    )
    assert_type(kwimage.fill_nans_with_checkers(draw_field), np.ndarray)
    assert_type(kwimage.nodata_checkerboard(draw_field), np.ndarray)

    assert_type(kwimage.warp_image(image, np.eye(3)), np.ndarray)
    assert_type(
        kwimage.warp_image(image, np.eye(3), return_info=True),
        tuple[np.ndarray, WarpInfo],
    )
    assert_type(kwimage.warp_affine(image, np.eye(3)), np.ndarray)
    assert_type(
        kwimage.warp_affine(image, np.eye(3), return_info=True),
        tuple[np.ndarray, WarpInfo],
    )
    assert_type(kwimage.warp_projective(image, np.eye(3)), np.ndarray)
    assert_type(
        kwimage.warp_projective(image, np.eye(3), return_info=True),
        tuple[np.ndarray, WarpInfo],
    )
    assert_type(kwimage.imresize(image, scale=0.5), np.ndarray)
    assert_type(
        kwimage.imresize(image, scale=0.5, return_info=True),
        tuple[np.ndarray, ResizeInfo],
    )

    assert_type(kwimage.num_channels(image), int)
    assert_type(kwimage.ensure_float01(image), np.ndarray)
    assert_type(kwimage.ensure_uint255(image), np.ndarray)
    assert_type(
        kwimage.make_channels_comparable(image, image),
        tuple[np.ndarray, np.ndarray],
    )
    assert_type(kwimage.atleast_3channels(binary), np.ndarray)
    assert_type(kwimage.exactly_1channel(binary), np.ndarray)
    padkw: PaddedSlicePadKw = {'mode': 'constant'}
    assert_type(
        kwimage.padded_slice(
            binary, (slice(0, 4),), pad=1, padkw=padkw
        ),
        np.ndarray,
    )
    assert_type(kwimage.padded_slice(binary, (slice(0, 4),)), np.ndarray)
    assert_type(
        kwimage.padded_slice(
            binary, (slice(0, 4),), return_info=True
        ),
        tuple[np.ndarray, PaddedSliceInfo],
    )
    robust_params: RobustNormalizerParams = {
        'low': 0.01,
        'mid': 0.5,
        'high': 0.9,
        'scaling': 'linear',
    }
    robust_info = kwimage.find_robust_normalizers(
        binary, params=robust_params
    )
    assert_type(robust_info, RobustNormalizerInfo)
    assert_type(robust_info['type'], Literal['normalize'] | None)
    assert_type(robust_info['min_val'], RobustNormalizerScalar)
    assert_type(kwimage.normalize_intensity(binary), np.ndarray)
    assert_type(
        kwimage.normalize_intensity(binary, return_info=True),
        tuple[np.ndarray, RobustNormalizerInfo],
    )
    assert_type(
        kwimage.normalize_intensity(
            binary, return_info=True, axis=0, params=robust_params
        ),
        tuple[np.ndarray, list[RobustNormalizerInfo]],
    )
    assert_type(kwimage.crop_border_by_color(image), np.ndarray)

    assert_type(kwimage.imcrop(image, (8, 8)), np.ndarray)
    assert_type(
        kwimage.convert_colorspace(image, 'RGB', 'BGR'), np.ndarray
    )
    assert_type(kwimage.adjust(image), np.ndarray)
    assert_type(kwimage.gaussian_patch((7, 7)), np.ndarray)
    assert_type(kwimage.gaussian_blur(image), np.ndarray)
    assert_type(kwimage.morphology(binary, 'dilate'), np.ndarray)
    assert_type(
        kwimage.connected_components(binary, with_stats=False),
        tuple[np.ndarray, ConnectedComponentsInfo],
    )
    assert_type(
        kwimage.connected_components(binary),
        tuple[np.ndarray, ConnectedComponentsStatsInfo],
    )

    box = kwimage.Box.coerce([0.0, 1.0, 2.0, 3.0], 'xywh')
    assert_type(box.data, ArrayData)
    assert_type(box.contains(np.empty((3, 2))), ArrayData)
    assert_type(box.corners(), np.ndarray)
    assert_type(box.aspect_ratio, BoxScalar)
    assert_type(box.center, tuple[BoxScalar, BoxScalar])
    assert_type(box.center_x, BoxScalar)
    assert_type(box.center_y, BoxScalar)
    assert_type(box.width, BoxScalar)
    assert_type(box.height, BoxScalar)
    assert_type(box.tl_x, BoxScalar)
    assert_type(box.tl_y, BoxScalar)
    assert_type(box.br_x, BoxScalar)
    assert_type(box.br_y, BoxScalar)
    assert_type(box.dtype, BoxDType)
    assert_type(box.area, BoxScalar)
    assert_type(box.to_shapely(), ShapelyPolygon)
    assert_type(box.to_polygon(), kwimage.Polygon)
    assert_type(box.to_coco(), list[float])
    assert_type(box.draw_on(np.zeros((8, 8, 3), dtype=np.uint8)), np.ndarray)
    assert_type(box.draw(), None)
    assert_type(box.translate((1.0, 2.0)), kwimage.Box)
    assert_type(box.scale(2.0), kwimage.Box)
    assert_type(box.warp(np.eye(3)), kwimage.Box)
    assert_type(box.clip(0, 0, 10, 10), kwimage.Box)
    assert_type(box.pad(1, 2, 3, 4), kwimage.Box)
    assert_type(box.resize(width=4, height=5), kwimage.Box)
    assert_type(box.round(), kwimage.Box)
    assert_type(box.quantize(), kwimage.Box)
    assert_type(box.copy(), kwimage.Box)
    assert_type(box.to_ltrb(copy=False), kwimage.Box)
    assert_type(box.to_xywh(copy=False), kwimage.Box)
    assert_type(box.to_cxywh(copy=False), kwimage.Box)
    assert_type(box.toformat('ltrb', copy=False), kwimage.Box)
    assert_type(box.astype(np.float32), kwimage.Box)

    det_coerce_kw: DetectionsCoerceKwargs = {
        'boxes': boxes,
        'cnames': ['a', 'b'],
    }
    opaque_det_data: object = {}
    assert_type(
        kwimage.Detections.coerce(opaque_det_data, **det_coerce_kw),
        kwimage.Detections,
    )

    opaque_segmentation: object = None
    assert_type(
        kwimage.Segmentation.coerce(opaque_segmentation),
        kwimage.Segmentation | None,
    )
    opaque_mask: object = binary
    assert_type(kwimage.Mask.coerce(opaque_mask), kwimage.Mask)
    segmentation_items: list[object] = [None]
    assert_type(
        kwimage.SegmentationList.coerce(segmentation_items),
        kwimage.SegmentationList | None | float,
    )

    heatmap = kwimage.Heatmap(
        class_probs=np.empty((2, 8, 8), dtype=np.float32),
        img_dims=(16, 16),
        classes=['a', 'b'],
    )
    assert_type(heatmap.class_probs, ArrayData)
    assert_type(heatmap[0], ArrayData)
    assert_type(heatmap.shape, HeatmapShape | None)
    assert_type(heatmap.bounds, HeatmapShape)
    assert_type(heatmap.dims, HeatmapShape)
    assert_type(heatmap.offset, HeatmapSpatialData | None)
    assert_type(heatmap.diameter, HeatmapSpatialData | None)
    assert_type(heatmap.img_dims, HeatmapImageDims | None)
    assert_type(heatmap.tf_data_to_img, HeatmapTransform | None)
    assert_type(heatmap.classes, DetectionClasses | None)
    assert_type(heatmap.numpy(), kwimage.Heatmap)
    assert_type(heatmap.tensor(), kwimage.Heatmap)
    assert_type(heatmap.tensor('cpu'), kwimage.Heatmap)
    assert_type(heatmap.warp(np.eye(3)), kwimage.Heatmap)
    assert_type(heatmap.scale(2.0), kwimage.Heatmap)
    assert_type(heatmap.translate((1.0, 2.0)), kwimage.Heatmap)
    assert_type(heatmap.colorize(0), np.ndarray)
    assert_type(heatmap.draw_stacked(top=1, chosen_cxs=[0]), np.ndarray)
    assert_type(heatmap.draw_on(kpts=[0]), np.ndarray)
    assert_type(heatmap.draw_on(kpts=True), np.ndarray)
    assert_type(heatmap.upscale(0), np.ndarray)
    assert_type(heatmap.draw(imgspace=False), None)
    assert_type(
        kwimage.Heatmap.combine([heatmap], dtype=np.float32), kwimage.Heatmap
    )
    assert_type(heatmap.detect(0), kwimage.Detections)

    coords = kwimage.Coords(np.empty((3, 2)))
    assert_type(coords.data, ArrayData)
    assert_type(coords.shape, tuple[int, ...] | torch.Size)
    assert_type(coords.copy(), kwimage.Coords)
    assert_type(coords.compress([True, False, True]), kwimage.Coords)
    assert_type(coords.take([0, 2]), kwimage.Coords)
    assert_type(coords.scale(2.0), kwimage.Coords)
    assert_type(coords.translate((1.0, 2.0)), kwimage.Coords)
    assert_type(coords.warp(np.eye(3)), kwimage.Coords)
    assert_type(coords.numpy(), kwimage.Coords)
    assert_type(coords.tensor(), kwimage.Coords)
    assert_type(coords.tensor('cpu'), kwimage.Coords)
    assert_type(coords.to_wkt(), str)
    assert_type(coords.to_shapely(), MultiPoint)
    assert_type(coords.to_imgaug((10, 10)), ImgAugKeypointsOnImage)
    kpoi = cast(ImgAugKeypointsOnImage, object())
    assert_type(kwimage.Coords.from_imgaug(kpoi), kwimage.Coords)

    points = kwimage.Points(xy=np.empty((3, 2)))
    assert_type(points.xy, ArrayData)
    assert_type(points.shape, tuple[int, ...] | torch.Size)
    assert_type(points.scale(2.0), kwimage.Points)
    assert_type(points.translate((1.0, 2.0)), kwimage.Points)
    assert_type(points.warp(np.eye(3)), kwimage.Points)
    assert_type(points.compress([True, False, True]), kwimage.Points)
    assert_type(points.take([0, 2]), kwimage.Points)
    assert_type(points.numpy(), kwimage.Points)
    assert_type(points.tensor(), kwimage.Points)
    assert_type(points.tensor('cpu'), kwimage.Points)
    assert_type(points.to_wkt(), str)
    assert_type(points.to_shapely(), MultiPoint)
    assert_type(points.to_imgaug((10, 10)), ImgAugKeypointsOnImage)
    assert_type(kwimage.Points.from_imgaug(kpoi), kwimage.Points)
    assert_type(points.to_coco(), CocoKeypoints)
    assert_type(points.to_coco(style='new-v2'), CocoKeypoints)
    coco_kpoint: CocoKeypointDict = {'xy': [1.0, 2.0], 'visible': 2}
    coco_kpoint_columns: CocoKeypointColumns = {
        'x': [1.0], 'y': [2.0], 'visible': [2]
    }
    point_classes: PointClasses = ['nose', 'tail']
    assert_type(
        kwimage.Points.from_coco([coco_kpoint], classes=point_classes),
        kwimage.Points,
    )
    assert_type(
        kwimage.Points.from_coco(coco_kpoint_columns), kwimage.Points
    )
    assert_type(kwimage.Points.from_coco([0.0, 0.0, 2.0]), kwimage.Points)
    assert_type(kwimage.Points.from_coco(None), None)
    assert_type(kwimage.Points.coerce(np.empty((3, 2))), kwimage.Points)
    assert_type(
        kwimage.Points.random(classes=point_classes, rng=0), kwimage.Points
    )

    point_list = kwimage.PointsList([points])
    assert_type(point_list[0], kwimage.Points)
    assert_type(point_list.scale(2.0), kwimage.PointsList)
    assert_type(point_list.translate((1.0, 2.0)), kwimage.PointsList)
    assert_type(point_list.warp(np.eye(3)), kwimage.PointsList)
    assert_type(
        point_list.draw_on(np.zeros((16, 16, 3), dtype=np.uint8)),
        np.ndarray,
    )
    assert_type(point_list.to_coco(), Iterator[CocoKeypoints])


    poly = kwimage.Polygon(exterior=np.empty((4, 2)))
    assert_type(poly.data, PolygonData)
    assert_type(poly.exterior, kwimage.Coords)
    assert_type(poly.interiors, list[kwimage.Coords])
    assert_type(poly.copy(), kwimage.Polygon)
    assert_type(poly.scale(2.0), kwimage.Polygon)
    assert_type(poly.translate((1.0, 2.0)), kwimage.Polygon)
    assert_type(poly.rotate(0.5), kwimage.Polygon)
    assert_type(poly.warp(np.eye(3)), kwimage.Polygon)
    assert_type(poly.round(), kwimage.Polygon)
    assert_type(poly.astype(np.float32), kwimage.Polygon)
    assert_type(poly.numpy(), kwimage.Polygon)
    assert_type(poly.tensor(), kwimage.Polygon)
    assert_type(poly.tensor('cpu'), kwimage.Polygon)
    assert_type(poly.to_shapely(), ShapelyPolygon)
    assert_type(poly.to_geojson(), PolygonGeoJSON)
    assert_type(poly.to_coco(), list[int | float])
    assert_type(poly.to_coco(style='new'), CocoPolygonDict)
    assert_type(poly.to_multi_polygon(), kwimage.MultiPolygon)
    assert_type(poly.box(), kwimage.Box)
    assert_type(poly.to_mask((10, 10)), kwimage.Mask)
    assert_type(poly.to_relative_mask(), kwimage.Mask)
    assert_type(
        poly.to_relative_mask(return_offset=True),
        tuple[kwimage.Mask, tuple[Number, Number]],
    )
    assert_type(
        poly.buffer(1.0), kwimage.Polygon | kwimage.MultiPolygon
    )
    assert_type(
        poly.convex_hull, kwimage.Polygon | kwimage.MultiPolygon
    )

    polygon_coco: CocoPolygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert_type(kwimage.Polygon.from_coco(polygon_coco), kwimage.Polygon)
    polygon_geojson: PolygonGeoJSON = poly.to_geojson()
    assert_type(kwimage.Polygon.from_geojson(polygon_geojson), kwimage.Polygon)
    polygon_style: CocoPolygonStyle = 'new'
    assert_type(poly.to_coco(style=polygon_style), CocoPolygon)

    mpoly = kwimage.MultiPolygon([poly])
    assert_type(mpoly[0], kwimage.Polygon)
    assert_type(mpoly.scale(2.0), kwimage.MultiPolygon)
    assert_type(mpoly.translate((1.0, 2.0)), kwimage.MultiPolygon)
    assert_type(mpoly.warp(np.eye(3)), kwimage.MultiPolygon)
    assert_type(mpoly.numpy(), kwimage.MultiPolygon)
    assert_type(mpoly.tensor(), kwimage.MultiPolygon)
    assert_type(mpoly.tensor('cpu'), kwimage.MultiPolygon)
    assert_type(mpoly.swap_axes(), kwimage.MultiPolygon)
    assert_type(mpoly.to_shapely(), ShapelyMultiPolygon)
    assert_type(mpoly.to_geojson(), MultiPolygonGeoJSON)
    assert_type(mpoly.to_coco(), list[CocoPolygon])
    assert_type(mpoly.to_coco(style='new'), list[CocoPolygon])
    assert_type(mpoly.draw(), list[PathPatch | None])

    multi_geojson: MultiPolygonGeoJSON = mpoly.to_geojson()
    assert_type(
        kwimage.MultiPolygon.from_geojson(multi_geojson), kwimage.MultiPolygon
    )
    assert_type(
        kwimage.MultiPolygon.from_geojson(polygon_geojson), kwimage.MultiPolygon
    )
    assert_type(
        kwimage.MultiPolygon.from_coco([polygon_coco]), kwimage.MultiPolygon
    )

    polygon_items: list[kwimage.Polygon | kwimage.MultiPolygon | None] = [
        poly, mpoly, None
    ]
    polygon_list = kwimage.PolygonList(polygon_items)
    assert_type(
        polygon_list[0], kwimage.Polygon | kwimage.MultiPolygon | None
    )
    assert_type(polygon_list.scale(2.0), kwimage.PolygonList)
    assert_type(polygon_list.scale(np.array([2.0, 3.0])), kwimage.PolygonList)
    assert_type(polygon_list.translate((1.0, 2.0)), kwimage.PolygonList)
    assert_type(polygon_list.warp(np.eye(3)), kwimage.PolygonList)
    assert_type(
        polygon_list.draw_on(np.zeros((16, 16, 3), dtype=np.uint8)),
        np.ndarray,
    )
    assert_type(polygon_list.numpy(), kwimage.PolygonList)
    assert_type(polygon_list.tensor(), kwimage.PolygonList)
    assert_type(polygon_list.tensor('cpu'), kwimage.PolygonList)
    assert_type(polygon_list.to_boxes(), kwimage.Boxes)
    assert_type(
        polygon_list.to_coco(),
        Iterator[CocoPolygon | list[CocoPolygon] | None],
    )

    mask = kwimage.Mask(np.zeros((8, 8), dtype=np.uint8), 'c_mask')
    assert_type(mask.data, MaskData)
    assert_type(mask.dtype, np.dtype[Any] | torch.dtype)
    assert_type(mask.shape, Sequence[int] | None)
    assert_type(mask.area, MaskArea)
    assert_type(mask.copy(), kwimage.Mask)
    assert_type(mask.to_c_mask(), kwimage.Mask)
    assert_type(mask.to_fortran_mask(), kwimage.Mask)
    assert_type(mask.to_array_rle(), kwimage.Mask)
    assert_type(mask.to_bytes_rle(), kwimage.Mask)
    assert_type(mask.numpy(), kwimage.Mask)
    assert_type(mask.tensor(), kwimage.Mask)
    assert_type(mask.tensor('cpu'), kwimage.Mask)
    assert_type(mask.scale(2.0), kwimage.Mask)
    assert_type(mask.translate((1.0, 2.0)), kwimage.Mask)
    assert_type(mask.warp(np.eye(3)), kwimage.Mask)
    assert_type(mask.get_patch(), ArrayData)
    assert_type(mask.get_xywh(), np.ndarray)
    assert_type(mask.box(), kwimage.Box)
    assert_type(mask.to_boxes(), kwimage.Boxes)
    assert_type(mask.to_multi_polygon(), kwimage.MultiPolygon)
    assert_type(mask.get_convex_hull(), np.ndarray)
    assert_type(mask.iou(mask), float | np.floating[Any])
    assert_type(mask.to_coco(), CocoMaskRLE)

    assert_type(MaskFormat.BYTES_RLE, Literal['bytes_rle'])
    assert_type(MaskFormat.ARRAY_RLE, Literal['array_rle'])
    assert_type(MaskFormat.C_MASK, Literal['c_mask'])
    assert_type(MaskFormat.F_MASK, Literal['f_mask'])

    mask_format: MaskFormatName = 'array_rle'
    assert_type(mask.toformat(mask_format), kwimage.Mask)
    assert_type(kwimage.Mask.from_mask(binary, method='naive'), kwimage.Mask)
    assert_type(mask.warp(np.eye(3), output_dims='same'), kwimage.Mask)

    mask_list = kwimage.MaskList([mask, None])
    assert_type(mask_list[0], kwimage.Mask | None)
    assert_type(mask_list.to_mask_list(), kwimage.MaskList)
    assert_type(mask_list.to_polygon_list(), kwimage.PolygonList)
    assert_type(mask_list.to_segmentation_list(), kwimage.SegmentationList)
    assert_type(mask_list.scale(2.0), kwimage.MaskList)
    assert_type(mask_list.translate((1.0, 2.0)), kwimage.MaskList)
    assert_type(mask_list.warp(np.eye(3)), kwimage.MaskList)
    assert_type(mask_list.warp(np.eye(3), output_dims='same'), kwimage.MaskList)
    assert_type(
        mask_list.draw_on(np.zeros((16, 16, 3), dtype=np.uint8)),
        np.ndarray,
    )
    assert_type(mask_list.numpy(), kwimage.MaskList)
    assert_type(mask_list.tensor(), kwimage.MaskList)
    assert_type(mask_list.tensor('cpu'), kwimage.MaskList)
    assert_type(mask_list.to_coco(), Iterator[CocoMaskRLE | None])

    segmentation = kwimage.Segmentation(mask, 'mask')
    assert_type(segmentation.data, SegmentationBackend)
    assert_type(segmentation.to_mask(), kwimage.Mask)
    assert_type(segmentation.to_multi_polygon(), kwimage.MultiPolygon)
    assert_type(segmentation.box(), kwimage.Box)
    assert_type(segmentation.area, Number | torch.Tensor)
    assert_type(segmentation.meta, Mapping[str, object])
    assert_type(segmentation.warp(np.eye(3)), SegmentationBackend)
    assert_type(segmentation.scale(2.0), SegmentationBackend)
    assert_type(segmentation.translate((1.0, 2.0)), SegmentationBackend)
    assert_type(segmentation.numpy(), SegmentationBackend)
    assert_type(segmentation.tensor(), SegmentationBackend)
    assert_type(segmentation.tensor('cpu'), SegmentationBackend)
    assert_type(segmentation.to_coco(), SegmentationCoco)
    assert_type(kwimage.Segmentation.coerce(mask), kwimage.Segmentation)

    segmentation_list = kwimage.SegmentationList([segmentation, None])
    assert_type(
        segmentation_list[0], kwimage.Segmentation | None
    )
    assert_type(
        segmentation_list.to_segmentation_list(), kwimage.SegmentationList
    )
    assert_type(segmentation_list.to_mask_list(), kwimage.MaskList)
    assert_type(segmentation_list.to_polygon_list(), kwimage.PolygonList)
    assert_type(segmentation_list.scale(2.0), kwimage.SegmentationList)
    assert_type(
        segmentation_list.translate((1.0, 2.0)), kwimage.SegmentationList
    )
    assert_type(segmentation_list.warp(np.eye(3)), kwimage.SegmentationList)
    assert_type(
        segmentation_list.draw_on(np.zeros((16, 16, 3), dtype=np.uint8)),
        np.ndarray,
    )
    assert_type(segmentation_list.numpy(), kwimage.SegmentationList)
    assert_type(segmentation_list.tensor(), kwimage.SegmentationList)
    assert_type(segmentation_list.tensor('cpu'), kwimage.SegmentationList)
    assert_type(
        segmentation_list.to_coco(), Iterator[SegmentationCoco | None]
    )
    assert_type(
        kwimage.SegmentationList.coerce(None, none_policy='return-None'),
        kwimage.SegmentationList | None | float,
    )

    det_boxes = kwimage.Boxes(np.empty((3, 4)), 'xywh')
    det_scores = np.empty(3, dtype=np.float32)
    det_class_idxs = np.empty(3, dtype=np.int64)
    dets = kwimage.Detections(
        boxes=det_boxes,
        scores=det_scores,
        class_idxs=det_class_idxs,
        classes=['a', 'b'],
    )
    assert_type(dets.boxes, kwimage.Boxes)
    assert_type(dets.class_idxs, DetectionArray | None)
    assert_type(dets.scores, DetectionArray | None)
    assert_type(dets.probs, DetectionArray | None)
    assert_type(dets.weights, DetectionArray | None)
    assert_type(dets.classes, DetectionClasses | None)
    assert_type(dets.keypoints, DetectionKeypoints | None)
    assert_type(dets.segmentations, DetectionSegmentations | None)
    assert_type(dets.copy(), kwimage.Detections)
    assert_type(dets.warp(np.eye(3)), kwimage.Detections)
    assert_type(dets.scale(2.0), kwimage.Detections)
    assert_type(dets.translate((1.0, 2.0)), kwimage.Detections)
    assert_type(dets.argsort(), DetectionArray)
    assert_type(dets.non_max_supression(), DetectionIndices)
    assert_type(dets.non_max_supress(), kwimage.Detections)
    assert_type(dets.sort(), kwimage.Detections)
    assert_type(dets.compress([True, False, True]), kwimage.Detections)
    assert_type(dets.take([0, 2]), kwimage.Detections)
    assert_type(dets[[0, 2]], kwimage.Detections)
    assert_type(dets.numpy(), kwimage.Detections)
    assert_type(dets.tensor(), kwimage.Detections)
    assert_type(dets.tensor('cpu'), kwimage.Detections)
    assert_type(dets.device, torch.device | None)
    assert_type(dets.dtype, DetectionDType)
    coco_anns: Sequence[Mapping[str, object]] = [
        {'category_id': 1, 'bbox': [0.0, 0.0, 4.0, 5.0]}
    ]
    coco_cats: Sequence[Mapping[str, object]] = [
        {'id': 1, 'name': 'class1'}
    ]
    assert_type(
        kwimage.Detections.from_coco_annots(
            coco_anns,
            coco_cats,
            classes=['class1'],
            shape=(16, 16),
        ),
        kwimage.Detections,
    )
    coco_ann_dset = cast(CocoAnnotsDatasetLike, object())
    assert_type(
        kwimage.Detections.from_coco_annots(coco_anns, dset=coco_ann_dset),
        kwimage.Detections,
    )
    assert_type(
        kwimage.Detections.demo(),
        tuple[
            kwimage.Detections,
            DetectionDemoImageInfo,
            DetectionDemoSamplerLike,
        ],
    )
    assert_type(dets.to_coco(), Generator[CocoDetection, None, None])
    assert_type(
        dets.to_coco(style='new'), Generator[CocoDetection, None, None]
    )
    coco_dset = cast(CocoDatasetLike, object())
    assert_type(
        dets.to_coco(dset=coco_dset),
        Generator[CocoDetection, None, None],
    )
    assert_type(
        dets.draw_on(image, color=['red', 'green', 'blue']), np.ndarray
    )
    assert_type(
        kwimage.Detections.random(classes=['a', 'b'], rng=0),
        kwimage.Detections,
    )
    assert_type(dets.rasterize((8, 8), (16, 16)), kwimage.Heatmap)
    assert_type(
        dets.rasterize(
            (8, 8), (16, 16), tf_data_to_img=np.eye(3), img_dims=(16, 16)
        ),
        kwimage.Heatmap,
    )
    assert_type(
        dets.non_max_supression(device_id=0), DetectionIndices
    )

    coords_fill = kwimage.Coords(np.array([[1.0, 2.0]], dtype=np.float32))
    assert_type(coords_fill.fill(image.copy(), 1.0), np.ndarray)
    assert_type(
        coords_fill.fill(image.copy(), [1.0, 0.5, 0.25], interp='nearest'),
        np.ndarray,
    )

    heatmap_interp: HeatmapInterpolation = 'bilinear'
    heatmap_cmap: HeatmapColorMap = 'plasma'
    heatmap_warp_matrix: HeatmapWarpMatrix = np.eye(3)
    heatmap_for_types = kwimage.Heatmap.random(rng=0, dims=(8, 8))
    assert_type(
        heatmap_for_types.colorize(0, cmap=heatmap_cmap), np.ndarray
    )
    assert_type(
        heatmap_for_types.upscale(0, interpolation=heatmap_interp), np.ndarray
    )
    assert_type(
        heatmap_for_types.warp(heatmap_warp_matrix, version='new'),
        kwimage.Heatmap,
    )
    assert_type(
        heatmap_for_types.draw(
            channel=np.int64(0), interpolation='nearest', kpts=True
        ),
        None,
    )

if TYPE_CHECKING:
    from kwimage.cli.crop_border import CropBorderCLI
    from kwimage.cli.stack_images import StackImagesCLI

    stack_cli = StackImagesCLI(input_fpaths=['a.png', 'b.png'])
    assert_type(stack_cli.input_fpaths, list[str])
    assert_type(stack_cli.axis, Literal['grid'] | int)
    assert_type(stack_cli.pad, int | None)
    assert_type(stack_cli.out, str | None)

    crop_cli = CropBorderCLI(src='input.png')
    assert_type(crop_cli.src, str)
    assert_type(crop_cli.dst, str | None)
