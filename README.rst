The Kitware Image Module
========================


.. # TODO Get CI services running on gitlab 
.. # |ReadTheDocs|

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |Downloads| 


The `kwimage` module handles low-level image operations at a high level.

The `kwimage` module builds on `kwarray` and provides tools commonly needed
when addressing computer vision problems. This includes functions for reading
images, non-maximum-suppression, image warp transformations, and
run-length-encoding.

The top-level API is:


.. code:: python

    from .algo import (available_nms_impls, daq_spatial_nms, non_max_supression,)
    from .im_alphablend import (ensure_alpha_channel, overlay_alpha_images,
                                overlay_alpha_layers,)
    from .im_core import (atleast_3channels, ensure_float01, ensure_uint255,
                          make_channels_comparable, num_channels,)
    from .im_cv2 import (convert_colorspace, draw_boxes_on_image,
                         draw_text_on_image, gaussian_patch, imscale, imresize,)
    from .im_demodata import (grab_test_image, grab_test_image_fpath,)
    from .im_io import (imread, imwrite,)
    from .im_runlen import (decode_run_length, encode_run_length, rle_translate,)
    from .im_stack import (stack_images, stack_images_grid,)
    from .structs import (Boxes, Coords, Detections, Heatmap, Mask, MaskList,
                          MultiPolygon, Points, PointsList, Polygon, PolygonList,
                          smooth_prob,)
    from .util_warp import (subpixel_accum, subpixel_align, subpixel_getvalue,
                            subpixel_maximum, subpixel_minimum, subpixel_set,
                            subpixel_setvalue, subpixel_slice, subpixel_translate,
                            warp_points, warp_tensor,)


NOTE: THE KWIMAGE STRUCTS WILL EVENTUALLY MOVE TO THE KWANNOT REPO


The most notable feature of the `kwimage` module are the `kwimage.structs`
objects. This includes the primitive `Boxes`, `Mask`, and `Coords` objects, The
semi-primitive `Points`, `Polygon` structures, and the composite `Heatmap` and
`Detections` structures (note: `Heatmap` is just a composite of array-like
structures). 

The primitive and semi-primitive objects store and manipulate annotation
geometry, and the composite structures combine primitives into a single
object that jointly manipulates the primitives using `warp` operations.

The `Detections` structure is a meta-structure that associates the other more
primitive components, and allows a developer to compose them into something
that represents objects of interest.  The details of this composition are left
up to the end-application.

The `Detections` object can also be "rasterized" and converted into a `Heatmap`
object, which represents the same information, but is in a form that is more
suitable for use when training convolutional neural networks. Likewise, the
output of neural networks can be directly encoded in a `kwimage.Heatmap`
object. The `Heatmap.detect` method can then be used to convert the dense
heatmap representation into a spare `Detections` representation that is more
suitable for use in an object-detection system. We note that the `detect`
function is not a special detection algorithm. The detection algorithm (which
is outside the scope of kwimage) produces the heatmap, and the `detect` method
effectively "inverts" the `rasterize` procedure of `Detections` by finding
peaks in the heatmap, and running non-maximum suppression.


This module contains data structures for image annotation primitives:

    * Boxes
    * Mask
    * Coords

And composites of these primitives:

    * Detections
    * Polygon
    * MultiPolygon
    * PolygonList
    * MaskList

    
.. |Pypi| image:: https://img.shields.io/pypi/v/kwimage.svg
   :target: https://pypi.python.org/pypi/kwimage

.. |Downloads| image:: https://img.shields.io/pypi/dm/kwimage.svg
   :target: https://pypistats.org/packages/kwimage

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwimage/badge/?version=latest
    :target: http://kwimage.readthedocs.io/en/latest/

.. # See: https://ci.appveyor.com/project/jon.crall/kwimage/settings/badges
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
   :target: https://ci.appveyor.com/project/jon.crall/kwimage/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/kwimage/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/kwimage/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwimage/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/kwimage/commits/master
