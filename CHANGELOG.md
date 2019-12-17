# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


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
