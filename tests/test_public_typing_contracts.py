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
    from collections.abc import Iterator
    from matplotlib.patches import PathPatch
    from shapely.geometry import MultiPoint
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon
    import torch
    from typing import cast
    from typing_extensions import assert_type

    import kwimage
    from kwimage._typing import ArrayData, ImgAugKeypointsOnImage
    from kwimage.structs.points import CocoKeypoints
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
