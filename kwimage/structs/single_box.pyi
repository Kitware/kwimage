import ubelt as ub
from _typeshed import Incomplete


class Box(ub.NiceRepr):
    boxes: Incomplete

    def __init__(self, boxes, _check: bool = False) -> None:
        ...

    @property
    def format(self):
        ...

    @property
    def data(self):
        ...

    def __nice__(self):
        ...

    @classmethod
    def random(self, **kwargs):
        ...

    @classmethod
    def from_slice(self, slice_):
        ...

    @classmethod
    def from_shapely(self, geom):
        ...

    @classmethod
    def from_dsize(self, dsize):
        ...

    @classmethod
    def from_data(self, data, format):
        ...

    @classmethod
    def coerce(cls, data, format: Incomplete | None = ..., **kwargs):
        ...

    @property
    def dsize(self):
        ...

    def translate(self, *args, **kwargs):
        ...

    def warp(self, *args, **kwargs):
        ...

    def scale(self, *args, **kwargs):
        ...

    def clip(self, *args, **kwargs):
        ...

    def quantize(self, *args, **kwargs):
        ...

    def copy(self, *args, **kwargs):
        ...

    def round(self, *args, **kwargs):
        ...

    def pad(self, *args, **kwargs):
        ...

    def resize(self, *args, **kwargs):
        ...

    def intersection(self, other):
        ...

    def union_hull(self, other):
        ...

    def to_ltrb(self, *args, **kwargs):
        ...

    def to_xywh(self, *args, **kwargs):
        ...

    def to_cxywh(self, *args, **kwargs):
        ...

    def toformat(self, *args, **kwargs):
        ...

    def astype(self, *args, **kwargs):
        ...

    def corners(self, *args, **kwargs):
        ...

    def to_boxes(self):
        ...

    @property
    def aspect_ratio(self):
        ...

    @property
    def center(self):
        ...

    @property
    def center_x(self):
        ...

    @property
    def center_y(self):
        ...

    @property
    def width(self):
        ...

    @property
    def height(self):
        ...

    @property
    def tl_x(self):
        ...

    @property
    def tl_y(self):
        ...

    @property
    def br_x(self):
        ...

    @property
    def br_y(self):
        ...

    @property
    def dtype(self):
        ...

    @property
    def area(self):
        ...

    def to_slice(self, endpoint: bool = ...):
        ...

    def to_shapely(self):
        ...

    def to_polygon(self):
        ...

    def to_coco(self):
        ...

    def draw_on(self,
                image: Incomplete | None = ...,
                color: str = ...,
                alpha: Incomplete | None = ...,
                label: Incomplete | None = ...,
                copy: bool = ...,
                thickness: int = ...,
                label_loc: str = ...):
        ...

    def draw(self,
             color: str = ...,
             alpha: Incomplete | None = ...,
             label: Incomplete | None = ...,
             centers: bool = ...,
             fill: bool = ...,
             lw: int = ...,
             ax: Incomplete | None = ...,
             setlim: bool = ...,
             **kwargs):
        ...
