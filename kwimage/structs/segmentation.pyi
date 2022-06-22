import ubelt as ub
from . import _generic
from _typeshed import Incomplete


class _WrapperObject(ub.NiceRepr):

    def __nice__(self):
        ...

    def draw(self, *args, **kw):
        ...

    def draw_on(self, *args, **kw):
        ...

    def warp(self, *args, **kw):
        ...

    def translate(self, *args, **kw):
        ...

    def scale(self, *args, **kw):
        ...

    def to_coco(self, *args, **kw):
        ...

    def numpy(self, *args, **kw):
        ...

    def tensor(self, *args, **kw):
        ...


class Segmentation(_WrapperObject):
    data: Incomplete
    format: Incomplete

    def __init__(self, data, format: Incomplete | None = ...) -> None:
        ...

    @classmethod
    def random(cls, rng: Incomplete | None = ...):
        ...

    def to_multi_polygon(self):
        ...

    def to_mask(self, dims: Incomplete | None = ..., pixels_are: str = ...):
        ...

    @property
    def meta(self):
        ...

    @classmethod
    def coerce(cls, data, dims: Incomplete | None = ...):
        ...


class SegmentationList(_generic.ObjectList):

    def to_polygon_list(self):
        ...

    def to_mask_list(self,
                     dims: Incomplete | None = ...,
                     pixels_are: str = ...):
        ...

    def to_segmentation_list(self):
        ...

    @classmethod
    def coerce(cls, data):
        ...
