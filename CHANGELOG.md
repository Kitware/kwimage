# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.9.0 - Unreleased

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
