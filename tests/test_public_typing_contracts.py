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
    from collections.abc import Generator, Iterator, Mapping, Sequence
    from matplotlib.patches import PathPatch
    from shapely.geometry import MultiPoint
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon
    import torch
    from typing import Any, cast
    from typing_extensions import assert_type

    import kwimage
    from kwimage._typing import ArrayData, ImgAugKeypointsOnImage

    from kwimage.structs.detections import (
        CocoDetection,
        DetectionArray,
        DetectionClasses,
        DetectionDType,
        DetectionIndices,
        DetectionKeypoints,
        DetectionSegmentations,
    )
    from kwimage.structs.mask import (
        CocoMaskRLE, MaskArea, MaskData,
    )
    from kwimage.structs.points import CocoKeypoints
    from kwimage.structs.segmentation import (
        SegmentationBackend, SegmentationCoco,
    )
    from kwimage.structs.single_box import BoxDType, BoxScalar
    from kwimage.structs.heatmap import (
        HeatmapImageDims, HeatmapShape, HeatmapSpatialData, HeatmapTransform,
    )
    from kwimage.structs.polygon import (
        CocoPolygon, CocoPolygonDict, MultiPolygonGeoJSON, PolygonData,
        PolygonGeoJSON,
    )
    from kwimage.im_core import PaddedSliceInfo, RobustNormalizerInfo
    from kwimage.im_cv2 import (
        ConnectedComponentsInfo, ConnectedComponentsStatsInfo,
    )
    from kwimage.im_transform import ResizeInfo, WarpInfo
    from kwimage.algo.algo_nms import NMSIndex, NMSIndices
    from kwimage.im_draw import TextDrawInfo
    from kwimage.im_runlen import RunLengthEncoding
    from kwimage.im_stack import StackTransform

    image = np.zeros((16, 20, 3), dtype=np.uint8)
    binary = np.zeros((16, 20), dtype=np.uint8)

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
    assert_type(kwimage.padded_slice(binary, (slice(0, 4),)), np.ndarray)
    assert_type(
        kwimage.padded_slice(
            binary, (slice(0, 4),), return_info=True
        ),
        tuple[np.ndarray, PaddedSliceInfo],
    )
    assert_type(
        kwimage.find_robust_normalizers(binary), RobustNormalizerInfo
    )
    assert_type(kwimage.normalize_intensity(binary), np.ndarray)
    assert_type(
        kwimage.normalize_intensity(binary, return_info=True),
        tuple[np.ndarray, RobustNormalizerInfo],
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
    assert_type(heatmap.warp(np.eye(3)), kwimage.Heatmap)
    assert_type(heatmap.scale(2.0), kwimage.Heatmap)
    assert_type(heatmap.translate((1.0, 2.0)), kwimage.Heatmap)
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
    assert_type(points.to_wkt(), str)
    assert_type(points.to_shapely(), MultiPoint)
    assert_type(points.to_imgaug((10, 10)), ImgAugKeypointsOnImage)
    assert_type(kwimage.Points.from_imgaug(kpoi), kwimage.Points)
    assert_type(points.to_coco(), CocoKeypoints)
    assert_type(kwimage.Points.from_coco([0.0, 0.0, 2.0]), kwimage.Points)
    assert_type(kwimage.Points.from_coco(None), None)
    assert_type(kwimage.Points.coerce(np.empty((3, 2))), kwimage.Points)

    point_list = kwimage.PointsList([points])
    assert_type(point_list[0], kwimage.Points)
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

    mpoly = kwimage.MultiPolygon([poly])
    assert_type(mpoly[0], kwimage.Polygon)
    assert_type(mpoly.scale(2.0), kwimage.MultiPolygon)
    assert_type(mpoly.translate((1.0, 2.0)), kwimage.MultiPolygon)
    assert_type(mpoly.warp(np.eye(3)), kwimage.MultiPolygon)
    assert_type(mpoly.numpy(), kwimage.MultiPolygon)
    assert_type(mpoly.tensor(), kwimage.MultiPolygon)
    assert_type(mpoly.swap_axes(), kwimage.MultiPolygon)
    assert_type(mpoly.to_shapely(), ShapelyMultiPolygon)
    assert_type(mpoly.to_geojson(), MultiPolygonGeoJSON)
    assert_type(mpoly.to_coco(), list[CocoPolygon])
    assert_type(mpoly.to_coco(style='new'), list[CocoPolygon])
    assert_type(mpoly.draw(), list[PathPatch | None])

    polygon_items: list[kwimage.Polygon | kwimage.MultiPolygon | None] = [
        poly, mpoly, None
    ]
    polygon_list = kwimage.PolygonList(polygon_items)
    assert_type(
        polygon_list[0], kwimage.Polygon | kwimage.MultiPolygon | None
    )
    assert_type(polygon_list.scale(2.0), kwimage.PolygonList)
    assert_type(polygon_list.translate((1.0, 2.0)), kwimage.PolygonList)
    assert_type(polygon_list.warp(np.eye(3)), kwimage.PolygonList)
    assert_type(polygon_list.numpy(), kwimage.PolygonList)
    assert_type(polygon_list.tensor(), kwimage.PolygonList)
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

    mask_list = kwimage.MaskList([mask, None])
    assert_type(mask_list[0], kwimage.Mask | None)
    assert_type(mask_list.to_mask_list(), kwimage.MaskList)
    assert_type(mask_list.to_polygon_list(), kwimage.PolygonList)
    assert_type(mask_list.to_segmentation_list(), kwimage.SegmentationList)
    assert_type(mask_list.numpy(), kwimage.MaskList)
    assert_type(mask_list.tensor(), kwimage.MaskList)
    assert_type(mask_list.to_coco(), Iterator[CocoMaskRLE | None])

    segmentation = kwimage.Segmentation(mask, 'mask')
    assert_type(segmentation.data, SegmentationBackend)
    assert_type(segmentation.to_mask(), kwimage.Mask)
    assert_type(segmentation.to_multi_polygon(), kwimage.MultiPolygon)
    assert_type(segmentation.box(), kwimage.Box)
    assert_type(segmentation.area, Number | torch.Tensor)
    assert_type(segmentation.meta, Mapping[str, Any])
    assert_type(segmentation.warp(np.eye(3)), SegmentationBackend)
    assert_type(segmentation.scale(2.0), SegmentationBackend)
    assert_type(segmentation.translate((1.0, 2.0)), SegmentationBackend)
    assert_type(segmentation.numpy(), SegmentationBackend)
    assert_type(segmentation.tensor(), SegmentationBackend)
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
    assert_type(segmentation_list.numpy(), kwimage.SegmentationList)
    assert_type(segmentation_list.tensor(), kwimage.SegmentationList)
    assert_type(
        segmentation_list.to_coco(), Iterator[SegmentationCoco | None]
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
    assert_type(dets.device, torch.device | None)
    assert_type(dets.dtype, DetectionDType)
    assert_type(dets.to_coco(), Generator[CocoDetection, None, None])
    assert_type(dets.rasterize((8, 8), (16, 16)), kwimage.Heatmap)
