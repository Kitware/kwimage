# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.4.0

### Added
* The `imread` function now accepts a `backend` kwarg, which allows the user to control how an image is loaded.
* Add `clip` method to `Polygon`
* Add `class_idx` data-key to `Heatmap` for semantic segmentation support.
* Add `class_energy` data-key to `Heatmap` for non-probabilistic output support.
* Add `Detections.from_coco_annots`, to create detections from COCO-style annotations.
* Add `to_coco` methods to all structures.

### Changed
* `Coords.warp` now tentatively supports OSR transforms.
* Continue improvements of annotation structures
* Increased efficiency of Cython CPU non-max-suppression
* `imread` now read ptif and tiff files using GDAL by default
* Tweaked implementation of `Heatmap.random`.
* `ensure_uint255` and `ensure_float01` now raise proper `ValueErrors` instead of using assert statements.
* `Points` draw methods now accept 'classes' as a color arg

### Fixed
* Python2 issues with `Boxes.__repr__`
* Can now correctly draw 1D `Boxes` objects
* Python2 issues mask shape using List[long] instead of List[int]
* Zero division in Cython `non-maximum-supression` with zero sized boxes.
* `Heatmap.shape` now works even if `class_probs` is not set.

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
