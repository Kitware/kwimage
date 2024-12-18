# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.11.2 - Unreleased

### Added
* Added `area` property to `Segmentation`
* Added `box` property to `Segmentation`
* Added `box` method to `Mask`
* Added `none_policy` to `SegmentationList.coerce`
* Added `remove_holes` to `Polygon` and `MultiPolygon`
* Added `draw_polyline_on_image` to `im_draw.py`
* Handle `Polygon.draw_on` where the input image is None.


### Deprecated
* Deprecate `Mask.to_boxes`
* Deprecate `Mask.bounding_box`

### Changed

* `kwimage.Polygon.to_mask` will now pick dimensions to fit the polygon if they are unspecified.
* `kwimage.Detections.to_coco` now handles the case where an entry in data or meta is None
* Generic spatial list objects used by `kwimage.Detections` now implement the
  full MutableSequence API, and may explicitly inherit from it in the future.

### Fixed
* Fixed issue in `Detections.from_coco_annots` where column arrays would not be
  aligned if an annotation was missing specific data.
* Fixed an issue that disallowed empty masks / heatmaps in some cases
* Issue where `Boxes.draw_on` would not allocate a correctly sized image when it was not given.
* Fix `Polygon.draw` where `facecolor='none'`.
* `Polygon.circle` now produces polygons with the correct specified number of sides. 
* Rare case where Boxes.warp with None would warp by a null matrix instead of identity.


## Version 0.11.1 - Released 2024-10-17

### Added
* `Points.to_shapely` and `Coords.to_shapely`
* `Points.from_shapely` and `Coords.from_shapely`
* `Points.wkt` and `Coords.wkt`
* classes arg to `Points.coerce`


### Changed
* `kwcoco.CategoryTree` now returns other metadata with it when converting to COCO dictionaries 
* Made `Points.to_coco` allow conversion to coco without category ids, by dropping them

### Fixed
* Issue in Points.coerce where unspecified keypoints categories were interpreted as specified with length zero.


## Version 0.11.0 - Released 2024-10-15

### Added

* `warp_affine` can now be used with `backend='itk'`.

* `kwimage.Points` now contains a "new-v2" style of coco keypoints that it can handle.

### Fixed
* Fix issue of coercing detections when segmentations are null.

* Patched issue where checkerboard would produce the wrong sized image, fix is not perfect.


### Changed

* Move `warp_affine`, `warp_projective`, and `warp_image` into their own module
  to support the using multiple backends for the underlying operation.

* Module init now uses lazy imports

* Default `grab_test_image` interpolation changed from lanczos to linear.

* cv2 is now officially optional, but a large amount of functionality will not be available without it.


## Version 0.10.2 - Released 2024-08-14

### Added
* Add classmethod `PolygonList.random`
* Add method `PolygonList.to_boxes`

### Fixed
* `kwimage.Polygon.draw` facecolor argument now properly handles coercible colors.
* `kwimage.PolygonList.draw` now handles setlim correctly
* `kwimage.Boxes.coerce` now handles the format argument correctly.


## Version 0.10.1 - Released 2024-07-23

### Added:
* Added `canonical` flag to Boxes to speed up creation time when input is known to be good.
* Added `bayer_value` argument to `kwimage.checkerboard`.

### Changed
* kwimage.Boxes and kwimage.Box now use `__slots__` 

### Fixed:
* Passing a list of alphas to `Detections.draw_on` now works correctly.
* Usage of distutils
* Issue in checkerboard function where the next multiple of X was not computed correctly.


## Version 0.10.0 - Released 2024-06-19

### Added

* Support for numpy 2.0 on Python 3.9+
* Add "opacity" argument to `kwimage.Color.adjust`.
* Add `__json__` to `kwimage.Boxes`
* Allow user to specify backend order in `load_image_shape`.
* Add the superstar test image
* `kwimage.Color.nearest_named` for finding the approximate name of a color.

### Removed
* Removed support for Python 3.6 and 3.7


### Changed
* Demodata is now more robust to networking outages. If no mirror can be
  reached, it generates random data to replace the real image.

### Fixed
* Fix issue with Boxes.draw when Boxes is empty
* Allow heatmap.draw to do something if classes are not defined.



## Version 0.9.25 - Released 2024-03-19

### Fixed

* `kwimage.Boxes.clip` now works on torch.Tensor data.


## Version 0.9.24 - Released 2024-02-23

### Added

* Basic CLI with the `stack_images` command.
* Workaround for `colorize` when heatmap is an index raster with negative ignore values.


## Version 0.9.23 - Released 2023-12-16

### Added:

* Add `drop_non_polygons` argument to `_ShapelyMixin.fix`.

* Changed: argument to `grab_test_image_fpath` can now be an existing path and
  it will return that. (allows users to substitute custom images for demodata).

* Added `clip`, `shape`, `endpoint`, and `wrap` argument to `Box.from_slice`


### Changed

* Can now call `draw_header_text` without an image argument.


## Version 0.9.22 - Released 2023-10-13

### Added

* `kwimage.Affine.fliprot` - a constructor that creates a transform equivalent to a flip and rotation with the positive image quadrant
* `kwimage.Boxes._ensure_nonnegative_extent` - experimental box method to fix negative width/heights


### Fixed

* `draw_on` methods for `Boxes`, `Polygon`, and `Points` no longer crashes if you pass it an image with 0 width or height.
* Fixed error in `subpixel_setvalue` where the coordinate extents did not
  always match the requested axes (e.g. sometimes the y axis would be clipped by width)


## Version 0.9.21 - Released 2023-10-01

### Added
* Added a too-big check in `imwrite` for better error messages.

### Changed
* Modified paths of augmented demodata to remove bad path characters.
* `Boxes.from_slice` now allows the left coordinate to be bigger than the right in the case you are clipping.


## Version 0.9.20 - Released 2023-08-09

### Fixed
* Fixed issue when giving `Affine.coerce` an empty dictionary
* Issue in box `intersection` / `union_hull` with integer data
* Fixed `Box.to_coco()`

### Added
* intersection to Box class.


## Version 0.9.19 - Released 2023-06-04

### Changed
* `grab_test_image` now falls back to mirrors if the first URL returns an error
* Allow `format` to be positional in `kwimage.Box.coerce` and `kwimage.Boxes.coerce`

### Fixed
* Fixed compatibility with new scikit-image
* Mixin methods are now shown in the docs


## Version 0.9.18 - Released 2023-05-02

### Changed
Disable torchvision NMS on windows by default


## Version 0.9.17 - Released 2023-04-28

### Changed
* Avoid importing torch globally

## Version 0.9.16 - Released 2023-04-27

### Added

* Add "regular" and "star" classmethods to Polygon

### Fixed
* float128 issues on windows

### Changed
* switch from ubelt.repr2 to ubelt.urepr
* On windows the default gdal compression is changed to LZW.
  The default everywhere else is still DEFLATE.


## Version 0.9.15 - Released 2023-04-04

### Added
* `center_x` and `center_y` to Box and Boxes.
* `Boxes.resize` now has `about` argument.

### Fixed
* Bug in `warp_affine` when the input image has more than 4 dims, has an empty dimension, and `border_value` is an array.
* Issue in `nodata_checkerboard` with masked arrays.


## Version 0.9.14 - Released 2023-03-16

### Added
* New function: `exactly_1channel`.

### Fixed
* kwimage.imresize now works when the input image has dsize=(1, 1)


## Version 0.9.13 - Released 2023-03-04

### Changed
* Small speedups
* Add cv2 backend to `fourier_mask`, which is a 3x speedup, and set this to the default.


## Version 0.9.12 - Released 2023-02-07

### Changed
* imresize now returns the original image if all scale / dsize arguments are unspecified or None.
* Polygon.fill can now work in non-inplace cases, but the `assert_inplace` flag must be set to 0.
* Allow nan as a nodata_method as alias for float

### Fixed
* `labels` in `draw_on` now interprets integers as booleans
* Fixed shapely warning about the "type" attribute. Now using `geom_type` instead.


## Version 0.9.11 - Released 2023-01-02


## Version 0.9.10 - Released 2023-01-02


### Fixed

* imresize now accepts input dtypes: bool, int32
* imresize now outputs the same channel dimensions as given in the input
* Polygon.union is fixed. (previously it did intersection)
* replaced np.bool with bool
* Fixed issue when drawing text on an unspecified image with an rgba background color

### Changed

* Better shapely and gdal transform support in Affine
* Improved agreement between polygon and box draw_on methods.
* Added Color.adjust 
* Moved color data to its own module
* `fill_nans_with_checkers` uses a muted checkerboard which is less visually jarring


## Version 0.9.9 - Released 2022-12-16

### Fixed

* Fixed issue in kwimage.Box where some functions (e.g. `tl_x`) were returning
  incorrectly shaped.
* Bug in nms with cython cpu.


## Version 0.9.8 - Released 2022-12-03

### Added
* Added new `Box` class to represent a single box.
* Basic shapely features in Polygon and MultiPolygon
* `Polygon.oriented_bounding_box`
* Added `__geo_interface__` to Polygon and MultiPolygon
* Add imwrite "pil" backend. 


### Changed:
* imresize no longer fails if none of the scale params are given, it just
  returns the image as-is.


### Fixed
* Bug in `kwimage.Boxes.draw(setlim=1)`
* But in `fill_nans_with_checkers` with what channels were filled with checkers.

### Changed
* support for torch in `Boxes.iooa` (requires kwarray update)
* imresize with letterbox now works when dsize has a None component.
* Expose `on_value` and `off_value` in `fill_nans_with_checkers`
* Better color support


## Version 0.9.7 - Released 2022-08-23

### Changed
* Changed image of Carl Sagan to one I know has a CC license. 
* Added IPFS CIDs to all demo images.
* Minor doc improvements.


### Added
* interpolate to Polygon and Color

## Version 0.9.6 - Released 2022-08-10

### Added
* The `bg_color` arg to `draw_header_text`
* Implemented `kwimage.warp_image`
* The `kwimage.Color.forimage` function is now public
* Basic support for polygon interpolation.

### Fixed
* Issue in Detections.draw where "cids" was not respected.
* Minor fixes in warp projective.
* `nodata_checkerboard` now works better with uint8 data.

### Changed
* Enhanced the checkerboard function, which can now take "on" and "off" string
  colors and return different dtypes.


## Version 0.9.5 - Released 2022-08-06

### Added
* `kwimage.warp_projective`
* `kwimage.connected_components`
* `kwimage.Boxes.resize` for setting the width/height of a box.
* `kwimage.Polygon.circle` can now be constructed as an ellipse by specifying radii as a tuple

### Fixed
* Passing single and multiple colors now works correctly for `Points.draw_on`.
* Fixed morphology when kernel was 0


## Version 0.9.4 - Released 2022-08-01


### Added
* Added "resize" option to `stack_images_grid`
* The `Coords` object now implements `__array__`.
* The `about` parameter for `Polygon` transformations has been
  expanded to include codes: "top,left", "top,right", etc...
* A `from_text` method for masks.


### Fixed
* Fixed issue in copy methods of polygon and coords


## Version 0.9.3 - Released 2022-07-26


### Fixed
* Issue where `Mask.to_multi_polygon` would break if the underlying data had
  dtype bool.
* Minor issues in heatmap (other than the giant issue that is that class)

### Added
* Support for decomposing projective transforms
* Stack images now can do automatic casting between uint255 and float01


### Changed
* In `warp_affine` change behavior so `border_value` scalars are transformed to tuples instead of 
  the current opencv  behavior where other channels are zeroed.


## Version 0.9.2 - Released 2022-07-01

### Added
* Add `wrap` kwarg to `Boxes.from_slice`

## Version 0.9.1 - Released 2022-06-30


### Added:
* `dsize` arg to `grab_test_image_fpath`

### Fixed:
* `kwimage.Affine.coerce` now correctly respects "about"

### Changed:
* The `kwimage.imread` gdal backend `nodata` argument is deprecated, and should
  now be specified as `nodata_method` to indicate how to handle nodata.
  Specific nodata values that are not embedded in image metadata should be
  handled outside of this function.

## Version 0.9.0 - Released 2022-06-21

### Added:
* Working on QOI support
* Added `kitware_green` and `kitware_blue` "Kitware colors" in `kwimage.Color`.
* Type annotations stubs


### Changed:

* Moved binary backends to `kwimage_ext` module as optional dependencies. Kwimage is now a pure python package. 


## Version 0.8.6 - Released 2022-04-27

### Added:
* Intention: ndarray functions in kwimage and kwarray should respect masked arrays.

### Changed:
* Deprecate using overview=-1 to get the coarsest overview. Use the string "coarsest" instead.
* The "overview" argument to gdal now uses 0-based indexing such that overview=0 is no overview, 
  and overview=1 is the first overview. This corresponds to the scale being `2^{overview}`.


## Version 0.8.5 - Released 2022-04-19

### Added:
* `kwimage.Affine.to_shapely` method for converting affine transforms to shapely-style 

### Fixed
* Fix issue in `Polygon.draw` where facecolor and edgecolor were not respected.
* Added `to_shapely` for `kwimage.Affine`
* Issue in `kwimage.load_image_shape` when using pathlib.Path


## Version 0.8.4 - Released 2022-04-02

### Fixed
* Bug in `Detections.draw_on(..., color='classes')` when a class is unknown
* Added workaround to cv2 bug to `Polygon.draw_on` when polygon coordinates are
  too large. Currently clips the polygon and emits a warning, this prevents
  crashes, but results may not be correct.


### Changed
* `Boxes.from_slice` no longer handles wraparound slices as it is not well
  defined and previous behavior was buggy.


## Version 0.8.3 - Released 2022-03-28

### Added
* facecolor and edgecolor in `Polygon.draw_on`

### Fixed
* Bug in MultiPolygon draw, `Polygon.draw_on` no longer crashes when polygon is empty.
* imread overviews now work for grayscale images

### Changed
* Speed up drawing in PolygonList and other generic draw funcs
* improve checkerboard

## Version 0.8.2 - Released 2022-03-14

### Added
* Added option to mask nodata regions when using imread and gdal backend.
* Added area property to Polygon and MultiPolygon.


## Version 0.8.1 - Released 2022-03-04

### Added

* Added option to `ignore_color_table` when using gdal imread backend.

### Changes

* Fixes for `intensity_normalize` 
* Allow user to specify nodata, transform, and crs when writing a geotiff


### Changed
* Consolidated environment variables in a new `kwiamge._internal` module.

### Fixed
* Previously Polygon.fill would only fill the first channel, this is now fixed
* `kwimage.imwrite` now raises an exception if the file fails to write.
* Fixed alpha in `Polygon.draw_on` for polygon borders.
* Fixed issue in `load_image_shape` when PIL fails.

## Version 0.7.17 - Released 2021-12-16

### Added

* `kwimage.Color.distinct` now uses `distinctipy` as a backend, and gains
  functionality to exclude existing colors.


### Fixed

* Serious bug in `kwimage.Affine`, where `decompose` and `affine` were not
  inverse. Behavior of kwimage.Affine with shear is now deprecated as it was
  inconsistent. New parameter that impacts shear is `xshear`.


## Version 0.7.16 - Released 2021-12-02

### Changed
* Mask translation is now more efficient for c/f formats.

### Fixed
* Bug introduced in `Mask.to_multipolygon` which caused bad offsets

## Version 0.7.15 - Released 2021-12-01

### Added:

* `pixels_are` flag to `to_mask` / `to_multi_polygon` methods, which can be
  points or areas. The latter uses a rasterio contour finding mechanism.
* `imread` gdalbackend can now read from overviews if they exist.

### Fixed:
* `warp_affine` now properly raises an error when cv2.warpAffine raises an unhandled error
* Polygon to/from methods now handle the empty polygon case

### Changed
* Improved polygon / segmentation coercion methods


## Version 0.7.14 - Released 2021-11-05

### Added
* kwimage.morphology
* kwimage.draw_header_text
* `Mask.to_multi_polygon` and `Polygon.to_mask` now take a flag `pixels_are`
  which will treat pixels as areas or points.


### Fixed
* But in imresize when a dim was automatically computed to be zero.

### Changed
* Using `math` is much faster than `numpy` for scalars, switching to that in kwimage.Affine
* Added default behavior to `draw_text_on_image` when origin is unspecified.
* Polygon and MultiPolygon coerce work better with geojson
* Work on new robust normalizers (see `normalize_intensity`)


## Version 0.7.13 - Released 2021-10-29

### Added
* Add function `gaussian_blur`
* Added `pad` as an argument to `stack_images` to override a negative `overlap`
* Added gaussian sigma kernel heuristic
* Added interleave as an option to COG write
* Add `fill` to `PolygonList`

### Fixed
* Polygons now better respect the color parameter in detections draw on
* kwimage.imread can now take pathlib.Path objects
* Fixed issue with warp of empty keypoints

### Changed
* Changed default GDAL compression in `kwimage.imwrite` from RAW to DEFLATE 
* Better overview support in `kwimage.imwrite`
* Improved speed of several `kwimage.Affine` routines.


## Version 0.7.12 - Released 2021-08-19

### Added
* Wrapped `itk.imread` with `kwimage.imread`.
* Added `kwimage.imcrop`
* Add `large_warp_dim` to `kwimage.warp_affine`

### Fixed
* `kwimage.warp_affine` now returns a sensible result when the source or
  destination image has no size.
* `kwimage.grab_test_image` now checks sha256 instead of sha1.
* Fixed tests that downloaded data, but did not check hashes


## Version 0.7.11 - Released 2021-08-11


### Changed
* Improved `cv2.imread` auto-space default based on the available backend.
  Should no longer need to specify it unless working with cv2.

### Fixed
* C-Extension errors for boxes now fallback on pure-python 


## Version 0.7.10 - Released 2021-08-02

### Added
* Enhanced capabilities of `draw_text_on_image`
* `Mask.draw_on` will now draw on an empty image if none is provided

### Fixed
* C-Extension errors for masks now fallback on pure-python 
* Not having `imgaug` no longer warns
* Fixed issues with generic warps

## Version 0.7.9 - Released 2021-07-23

### Added

* added `border_mode` and `border_value` to `warp_affine`.

* The `img` arg in `draw_text_on_image` can now be specified as a dictionary
  for control over canvas generation.


### Fixed

* issue computing canvas size in `draw_text_on_image` when an image is not given.

* failing to have imgaug no longer crashes arbitrary functions in annotation warp methods.


## Version 0.7.8 - Released 2021-06-17


### Changed

* `kwimage.structs.*.warp` can now accept a `kwimage.Affine` object.


## Version 0.7.7 - Released 2021-06-07

### Added

* `Boxes.to_slices`
* `Affine.concise`


### Fixed

* counterclockwise polygon check


### Changed

* improvements to `kwimage.Affine`.


## Version 0.7.6 - Released 2021-05-25

### Fixed

* Fixed failure in Affine.coerce
* Fixed random test failure in Boxes

## Version 0.7.5 - Released 2021-05-24


### Added

* New CI for building wheels with cibuildwheel
* Checkerboard demo image
* `warp_affine` with optional antialiasing for downsample operations.
* `imresize` now has optional antialiasing.
* `Affine.decompose` which extracts the scale, translation, rotation, and shear
  from an affine matrix.


### Changed
* `imscale` is deprecated and now results in an error
* `warp_image`, which did not do what you think it does, is deprecated and now results in an error


## Version 0.7.4 - Released 2021-05-13


## Version 0.7.3 - Released 2021-05-10


### Added
* New `transform.py` module

### Fixed
* Fixed numpy warning by using `float` instead of `np.float`.

### Changed
* Errors instead of warns for opencv import issues
* Warp methods now treat None as the identity transform.
* `kwimage.num_channels` no longer errors when the number of channels is not 1, 3, or 4. 


## Version 0.7.2 - Released 2021-04-22


### Fixed

* Using `from osgeo import gdal` instead of `import gdal` to fix for gdal 3.2.1

* Fixed numpy warning by using `int` instead of `np.int`.


### Changed

* opencv-python and opencv-python-headless are now optional dependencies.


## Version 0.7.1 - Released 2021-03-26

### Fixed

* Issue with RLE to fortran mask conversion: https://gitlab.kitware.com/computer-vision/kwimage/-/issues/2


## Version 0.7.0 - Released 2021-03-04

### Added

* Add `.jp2` to known GDAL extensions.

* Add `soft_fill` to `Coords`, which aims to paint Gaussian blobs as coordinate
  locations.

* Add `kwimage.padded_slice` ported and refactored from ndsampler.

* Add `reorder_axes` to `Coords`, which can change xy to yx representations etc...

* Added `Boxes.bounding_box` method 

* Added quantize method to bounding boxes which pushes the left-top coordinates
  to their floor and the right-bottom coordinates to their ceiling.

* `Detections.draw_on` can now accept color='classes'

* expose `thickness` in `Detections.draw_on`.

* Added `about` to Polygon and Coords scale and rotate.

* Add `to_geojson` to `PolygonList`

### Fixed
* `kwimage.Detections` now correctly handles `None` data values. Previously
  `None` was converted to an `array(None, dtype=object)`.

* demodata images now have the correct extension

* Fixed issue with channelless data in `Heatmap.draw_on`

* Bug in `Detections._make_labels` when scores are List[None]

### Changed

* BREAKING: TLBR has been internally switched to LTRB 

* Better Polygon coercion from geojson


## Version 0.6.10 - Released 2020-11-24

### 
* Added rotate to Coords and Polygon
* Added arg to control where text is drawn on `Boxes.draw_on`.

### Fixed
* GPG Keys needed to be renewed


## Version 0.6.9 - Released 2020-11-24

### Added
* Added support for `KWIMAGE_DISABLE_TORCHVISION_NMS` environ


## Version 0.6.8 - Released 2020-11-23

### Fixed
* Error in `Mask.get_xywh` when mask was empty.

## Version 0.6.7 - Released 2020-10-27

### Changed
* Torch and pandas are now optional

## Version 0.6.6 - Released 2020-10-05

### Added
* `kwimage.draw_text_on_image` now support `halign`
* `kwimage.Boxes.draw_on` now supports different colors for each box.

### Changed
* Removed explicit Python 3.5 support. Note, 3.5 should still still work using
  the universal `py2.py3-none-any` wheel.

### Fixed
* Issue with `Detections.from_coco` with keypoint categories
* Fixed `kwimage.Boxes.draw_on` when images are non-contiguous


## Version 0.6.5 - Released 2020-08-26 

### Added
* Add `to_boxes` to MultiPolygon

### Changed
* More methods in the Mask object should work without the c-extensions being built. 
* The `Mask.to_coco` method now returns a format based on the native encoding.
* Support for the new is preferred "ltrb" format over "tlbr" which will eventually be deprecated.
* No longer publishing wheels / CI testing for Python 2.7

### Fixed
* Fix bug with check for turbojpeg in imread.
* `subpixel_slice` now works with non-integer slices

## Version 0.6.4 - Released 2020-07-08 

### Added
* `Detections.from_coco_annots` now accepts `dset` keyword argument that
  attempts to fill the category_id coco field.
* `Boxes.iooas` - intersection-over-other-area
* `kwimage.imread` - now has a turbojpeg backend.

### Fixed
* Fix bug in `Detections.from_coco_annots` when categories are null.
* Fix bug `Detections.compress` when flags are in tensor space.

### Changed
* `kwimage.imwrite` now always returns the file path that was written to
  regardless of the backend.


## Version 0.6.2 - Released 2020-05-01 

### Added
* `draw_line_segments_on_image`
* Boxes.scale now accepts `about` keyword arg (can use to scale about center).
* Boxes.warp now accepts matrices and does inexact corner warping
* kwimage structures `warp` function now accepts a generic callable for mapping array-based points.
* add ``normalize`` function for intensity rebalance. 

### Changed
* Renamed `_rectify_interpolation` to `_coerce_interpolation`. Old function is deprecated and removed in the future.
* `_coerce_interpolation` now accepts strings for fallback interpolation flags.
* `Detections.from_coco_annots` now returns classes as ndsampler.CategoryTree when possible


## Version 0.6.1 -

### Added
* Added `im_filter` module with `fourier_mask` function. 
* Add "amazon" demo image


## Version 0.6.0 - Released 2020-02-19 

### Added
* thickness to `Boxes.draw_on`
* add `.round` to Boxes, Points, and Coords
* add `Boxes.contains`
* add `kwiamge.load_image_shape` 
* add `MultiPolygon.fill` 
* `kwimage.imwrite` now accepts the `backend` keyword, which can be `cv2`, `skimage`, or `gdal`. The `gdal` backend writes images as cloud-optimized-geotiffs.
* add `kwimage.structs.Segmentation` which encapsulates Masks, Polygons, and MultiPolygons
* Add `im_filter` which currently contains fourier domain filters.

### Fixed:
* issue with `draw_clf_on_image` when truth is unknown


## Version 0.5.7 - Released 2020-January-23


## Version 0.5.6 - Released 2020-January-17


### Fixed:
* fixed issue where non-max-suppression couldn't find a default implementation. 

### Changed:
* Faster color casting


## Version 0.5.5 - ??

### Fixed:
* Misc coco fixes.

## Version 0.5.4 - Released 2019-Dec-18

### Added
* Tentative `Color._forimage` method

### Changed
* Simpler demo data file names.
* The kwimage.struct `draw_on` methods now operate in-place (more often) when possible.

### Fixed
* Fixed color in kwimage.struct `draw_on` methods.
* Detections concatenate now works when segmentations are populated.


## Version 0.5.3 - Released 2019-Dec-17

### Added
* `imresize` now accepts `letterbox` flag.
* add `numpy` and `tensor` methods to `Mask`

### Changed
* `ensure_alpha_channel` now accepts `dtype` and `copy` kwargs.
* `Mask.draw_on` will now draw on images larger or smaller than the mask itself


### Fixed
* Fixed Boxes.draw
* `Boxes.draw_on` now works correctly on uint8 and float32 images with 1, 3, or 4 channels
* `Masks.draw_on` now works correctly uint8 or float32 images 
* Fixed error in `draw_clf_on_images`
* Fixed scale error in `Detections.random` when `segmentations=True`


## Version 0.5.2 - Released 2019-Nov-24

### Added 
* "torchvision" nms mode.
* Ported pure-image drawing functions from kwplot. These are `draw_boxes_on_image`, `draw_clf_on_image`,
                      `draw_text_on_image`, `make_heatmask`, `make_orimask`,
                      `make_vector_field`.
* Ported Color from kwplot


### Fixed
* Using the new (correct) torch defaults for `align_corners` in `warp_tensor` when possible.
* Fixed bug in "numpy" nms mode.

### Changed
* reworked nms auto mode
* nms impl=`cpu` / `gpu` / `py` are now deprecated for `cython_cpu` / `cython_gpu` / `numpy` instead.


## Version 0.5.1

### Changed
* First public release


## Version 0.5.0

### Added 
* Add option (hack) to build without C-extensions


## Version 0.4.0

### Added
* Add `imresize` as a more powerful alternative to `imscale`.
* The `imread` function now accepts a `backend` kwarg, which allows the user to control how an image is loaded.
* Add `clip` method to `Polygon`
* Add `class_idx` data-key to `Heatmap` for semantic segmentation support.
* Add `class_energy` data-key to `Heatmap` for non-probabilistic output support.
* Add `Detections.from_coco_annots`, to create detections from COCO-style annotations.
* Add `to_coco` methods to all structures.
* Add `meta` to `Polygons`

### Changed
* `Coords.warp` now tentatively supports OSR transforms.
* Continue improvements of annotation structures
* Increased efficiency of Cython CPU non-max-suppression
* `imread` now read ptif and tiff files using GDAL by default
* `imread` now reads `.r0` images using GDAL.
* Tweaked implementation of `Heatmap.random`.
* `ensure_uint255` and `ensure_float01` now raise proper `ValueErrors` instead of using assert statements.
* `Points` draw methods now accept 'classes' as a color arg

### Fixed
* `imread` can now handle nsf and color table images.
* Python2 issues with `Boxes.__repr__`
* Can now correctly draw 1D `Boxes` objects
* Python2 issues mask shape using List[long] instead of List[int]
* Zero division in Cython `non-maximum-supression` with zero sized boxes.
* `Heatmap.shape` now works even if `class_probs` is not set.
* `Coords.warp` now works with tensors.
* `Coords.warp` uses nearest neighbor interpolation for integer warping.

### Issues
* Heatmap.warp may have some odd behavior and emit warnings. 


## Version 0.3.0

### Added
* Add `Detections.rasterize` method to which is a lossy "pseudo-inverse" of `Heatmap.detect`.
* All classes in `kwimage.structs` should now have warp and draw method
* Add `subpixel_setvalue` and `subpixel_getvalue` 
* Add `Coords` data structure
* Add `Mask` data structure
* Add `Points` data structure
* Add `Polygon` data structure
* Add `_generic.ObjectList` data structure for `MaskList` and `PointsList`.
* Add `util_warp.warp_points`

### Changed
* Speedup `overlay_alpha_layers` by bypassing redundant `np.dstack` operations. 
* Changed `encode_run_length` to return a dictionary
* Add `output_shape` and `input_shape` as generally accepted kwargs for warp (even if they are unused)
* Spatial data structure `warp` methods can now accept `imgaug` augmenters
* Add checks to detection types

## Version 0.0.1

### Added
* Initial port of image-related utility code from KWIL. This includes:
    - non-maximum suppression
    - `Boxes`
    - `Detections`
    - `Heatmap`
    - `im_stack`
    - `im_core`
    - `im_io`
    - `im_demodata`
    - `im_misc`
    - `util_warp`
    - `im_runlen`
    - `im_cv2`
    - `im_alphablend`
