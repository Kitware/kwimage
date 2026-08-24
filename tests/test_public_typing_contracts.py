"""Static contracts for the public geometry API.

This module intentionally contains no runtime tests.  ``ty check kwimage
 tests/`` evaluates the TYPE_CHECKING block and ensures these public APIs do
not regress back to ``Any`` while the runtime test suite simply imports this
module.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from shapely.geometry import MultiPoint
    import torch
    from typing import cast
    from typing_extensions import assert_type

    import kwimage
    from kwimage._typing import ArrayData, ImgAugKeypointsOnImage
    from kwimage.structs.points import CocoKeypoints

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
