# The Kitware Image Module

The `kwimage` module handles low-level image operations at a high level.

The `kwimage` module builds on `kwarray` and provides tools commonly needed
when addressing computer vision problems. This includes functions for reading
images, non-maximum-suppression, image warp transformations, and
run-length-encoding.

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


