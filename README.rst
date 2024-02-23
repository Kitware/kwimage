The Kitware Image Module
========================


.. # TODO Get CI services running on gitlab
.. #

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |PypiDownloads| |ReadTheDocs|

+------------------+-------------------------------------------------------+
| Read the docs    | https://kwimage.readthedocs.io                        |
+------------------+-------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/kwimage    |
+------------------+-------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/kwimage                    |
+------------------+-------------------------------------------------------+
| Pypi             | https://pypi.org/project/kwimage                      |
+------------------+-------------------------------------------------------+

The main webpage for this project is: https://gitlab.kitware.com/computer-vision/kwimage

The ``kwimage`` module handles low-level image operations at a high level.

The core ``kwimage`` is a functional library with image-related helper
functions that are either unimplemented in or more have a more general
interface then their opencv counterparts.

The ``kwimage`` module builds on ``kwarray`` and provides tools commonly needed
when addressing computer vision problems. This includes functions for reading
images, resizing, image warp transformations, run-length-encoding, and
non-maximum-suppression.


The ``kwimage`` module is also the current home of my annotation data
structures, which provide efficient ways to interoperate between different
common annotation formats (e.g. different bounding box / polygon / point
formats).  These data structures have both a ``.draw`` and ``.draw_on`` method
for overlaying visualizations on matplotlib axes or numpy image matrices
respectively.


Read the docs at: http://kwimage.readthedocs.io/en/main/


The top-level API is:


.. code:: python

    from .algo import (available_nms_impls, daq_spatial_nms, non_max_supression,)
    from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                                overlay_alpha_layers,)
    from .im_color import (Color,)
    from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                          exactly_1channel, find_robust_normalizers,
                          make_channels_comparable, normalize, normalize_intensity,
                          num_channels, padded_slice,)
    from .im_cv2 import (connected_components, convert_colorspace, gaussian_blur,
                         gaussian_patch, imcrop, imresize, imscale, morphology,
                         warp_affine, warp_image, warp_projective,)
    from .im_demodata import (checkerboard, grab_test_image,
                              grab_test_image_fpath,)
    from .im_draw import (draw_boxes_on_image, draw_clf_on_image, draw_header_text,
                          draw_line_segments_on_image, draw_text_on_image,
                          draw_vector_field, fill_nans_with_checkers,
                          make_heatmask, make_orimask, make_vector_field,
                          nodata_checkerboard,)
    from .im_filter import (fourier_mask, radial_fourier_mask,)
    from .im_io import (imread, imwrite, load_image_shape,)
    from .im_runlen import (decode_run_length, encode_run_length, rle_translate,)
    from .im_stack import (stack_images, stack_images_grid,)
    from .structs import (Box, Boxes, Coords, Detections, Heatmap, Mask, MaskList,
                          MultiPolygon, Points, PointsList, Polygon, PolygonList,
                          Segmentation, SegmentationList, smooth_prob,)
    from .transform import (Affine, Linear, Matrix, Projective, Transform,)
    from .util_warp import (add_homog, remove_homog, subpixel_accum,
                            subpixel_align, subpixel_getvalue, subpixel_maximum,
                            subpixel_minimum, subpixel_set, subpixel_setvalue,
                            subpixel_slice, subpixel_translate, warp_points,
                            warp_tensor,)


NOTE: THE KWIMAGE STRUCTS MAY? EVENTUALLY MOVE TO THE KWANNOT REPO
(But this transition might take awhile)


The most notable feature of the ``kwimage`` module are the ``kwimage.structs``
objects. This includes the primitive ``Boxes``, ``Mask``, and ``Coords`` objects, The
semi-primitive ``Points``, ``Polygon`` structures, and the composite ``Heatmap`` and
``Detections`` structures (note: ``Heatmap`` is just a composite of array-like
structures).

The primitive and semi-primitive objects store and manipulate annotation
geometry, and the composite structures combine primitives into a single
object that jointly manipulates the primitives using ``warp`` operations.

The ``Detections`` structure is a meta-structure that associates the other more
primitive components, and allows a developer to compose them into something
that represents objects of interest.  The details of this composition are left
up to the end-application.

The ``Detections`` object can also be "rasterized" and converted into a ``Heatmap``
object, which represents the same information, but is in a form that is more
suitable for use when training convolutional neural networks. Likewise, the
output of neural networks can be directly encoded in a ``kwimage.Heatmap``
object. The ``Heatmap.detect`` method can then be used to convert the dense
heatmap representation into a spare ``Detections`` representation that is more
suitable for use in an object-detection system. We note that the ``detect``
function is not a special detection algorithm. The detection algorithm (which
is outside the scope of kwimage) produces the heatmap, and the ``detect`` method
effectively "inverts" the ``rasterize`` procedure of ``Detections`` by finding
peaks in the heatmap, and running non-maximum suppression.


This module contains data structures for three image annotation primitives:

    * Boxes  # technically this could be made out of Coords, probably not for efficiency and decoupling
    * Mask   # likewise this could be renamed to Raster
    * Coords #

These primative structures are used to define these metadata-containing composites:

    * Detections
    * Polygon
    * Heatmap
    * MultiPolygon
    * PolygonList
    * MaskList

All of these structures have a ``self.data`` attribute that holds a pointer to
the underlying data representation.

Some of these structures have a ``self.format`` attribute describing the
underlying data representation.

Most of the composite structures also have a ``self.meta`` attribute, which
holds user-level metadata (e.g. info about the classes).


Installation
------------

There are a few small quirks with installing kwimage. There is an issue with
the opencv python bindings such that we could rely on either the
`opencv-python` or `opencv-python-headless` package. If you have either of
these module already installed you can simply `pip install kwimage` without
encountering any issues related to this. But if you do not already have a
module that provides `import cv2` installed, then you should install kwimage
with one of the following "extra install" tags:

.. code-block:: bash

    # We recommend using the headless version
    pip install kwimage[headless]

    # OR

    # If other parts of your system depend on the opencv qt libs
    # (NOT RECOMMENDED: this can conflict with pyqt5)
    pip install kwimage[graphics]


Some features also require the ``kwimage_ext`` package to be installed, which
contains binary extensions that used to be distributed with this package in
older versions. These extension can be obtained by explicitly
``pip install kwimage_ext`` or via ``pip install kwimage[optional]`` (which also
brings in other optional libraries). You can disable loading of c-extensions at
runtime by setting the environment variable: `KWIMAGE_DISABLE_C_EXTENSIONS=1`.


A Note on GDAL
--------------

The kwimage library can use `GDAL <https://github.com/OSGeo/gdal/>`_ library
for certain tasks (e.g. IO of geotiffs).  GDAL can be a pain to install without
relying on conda.  Kitware also has a pypi index that hosts GDAL wheels for
linux systems:

.. code-block:: bash

    pip install --find-links https://girder.github.io/large_image_wheels GDAL



.. |Pypi| image:: https://img.shields.io/pypi/v/kwimage.svg
   :target: https://pypi.python.org/pypi/kwimage

.. |PypiDownloads| image:: https://img.shields.io/pypi/dm/kwimage.svg
   :target: https://pypistats.org/packages/kwimage

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwimage/badge/?version=release
    :target: http://kwimage.readthedocs.io/en/release/

.. # See: https://ci.appveyor.com/project/jon.crall/kwimage/settings/badges
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/main?svg=true
   :target: https://ci.appveyor.com/project/jon.crall/kwimage/branch/main

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/kwimage/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/kwimage/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwimage/badges/main/coverage.svg
    :target: https://gitlab.kitware.com/computer-vision/kwimage/commits/main
