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
    from kwimage.structs.polygon import (
        CocoPolygon, CocoPolygonDict, MultiPolygonGeoJSON, PolygonData,
        PolygonGeoJSON,
    )

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
