from typing import Any
from typing import Tuple
from typing import Callable
from numpy.typing import ArrayLike
import kwimage
from typing import List
from typing import Iterable
from numbers import Number
from numpy import ndarray
import shapely
from typing import Dict
import numpy as np
import matplotlib
import ubelt as ub
from _typeshed import Incomplete
from kwimage.structs import _generic

from kwimage._typing import SKImageGeometricTransform

__docstubs__: str


class _ShapelyMixin:

    def oriented_bounding_box(self):
        ...

    def buffer(self, *args, **kwargs):
        ...

    def simplify(self, tolerance, preserve_topology: bool = ...):
        ...

    @property
    def __geo_interface__(self):
        ...

    def union(self, other):
        ...

    def intersection(self, other):
        ...

    def difference(self, other):
        ...

    def symmetric_difference(self, other):
        ...

    def iooa(self, other):
        ...

    def iou(self, other):
        ...

    @property
    def area(self) -> float:
        ...

    @property
    def convex_hull(self):
        ...

    def is_invalid(self, explain: bool = False) -> bool | str:
        ...

    def fix(self):
        ...


class _PolyArrayBackend:

    def is_numpy(self):
        ...

    def is_tensor(self):
        ...

    def tensor(self, device=...):
        ...

    def numpy(self):
        ...


class _PolyWarpMixin:

    def to_imgaug(self, shape):
        ...

    def warp(self,
             transform: SKImageGeometricTransform | ArrayLike | Any | Callable,
             input_dims: Tuple | None = None,
             output_dims: Tuple | None = None,
             inplace: bool = False):
        ...

    def scale(self,
              factor: float | Tuple[float, float],
              about: Tuple | None = None,
              output_dims: Tuple | None = None,
              inplace: bool = False):
        ...

    def translate(self,
                  offset,
                  output_dims: Tuple | None = None,
                  inplace: bool = False):
        ...

    def rotate(self,
               theta: float,
               about: Tuple | None | str = None,
               output_dims: Tuple | None = None,
               inplace: bool = ...):
        ...

    def round(self, decimals: int = 0, inplace: bool = False) -> Polygon:
        ...

    def astype(self, dtype, inplace: bool = False) -> Polygon:
        ...

    def swap_axes(self, inplace: bool = False) -> Polygon:
        ...


class Polygon(_generic.Spatial, _PolyArrayBackend, _PolyWarpMixin,
              _ShapelyMixin, ub.NiceRepr):
    __datakeys__: Incomplete
    __metakeys__: Incomplete
    data: Incomplete
    meta: Incomplete

    def __init__(self,
                 data: Incomplete | None = ...,
                 meta: Incomplete | None = ...,
                 datakeys: Incomplete | None = ...,
                 metakeys: Incomplete | None = ...,
                 **kwargs) -> None:
        ...

    @property
    def exterior(self) -> kwimage.Coords:
        ...

    @property
    def interiors(self) -> List[kwimage.Coords]:
        ...

    def __nice__(self) -> str:
        ...

    @classmethod
    def circle(cls,
               xy: Iterable[Number] = ...,
               r: float | Number | Tuple[Number, Number] = 1.0,
               resolution: int = 64) -> Polygon:
        ...

    @classmethod
    def regular(cls, num, xy=..., r: int = ...):
        ...

    @classmethod
    def star(cls, xy=..., r: int = ...):
        ...

    @classmethod
    def random(cls,
               n: int = 6,
               n_holes: int = 0,
               convex: bool = True,
               tight: bool = False,
               rng: Incomplete | None = ...) -> Polygon:
        ...

    def to_mask(self,
                dims: Tuple | None = None,
                pixels_are: str = 'points') -> kwimage.Mask:
        ...

    def to_relative_mask(self, return_offset: bool = ...) -> kwimage.Mask:
        ...

    @classmethod
    def coerce(Polygon, data: object) -> kwimage.Polygon:
        ...

    @classmethod
    def from_shapely(
            Polygon,
            geom: shapely.geometry.polygon.Polygon) -> kwimage.Polygon:
        ...

    @classmethod
    def from_wkt(Polygon, data: str) -> kwimage.Polygon:
        ...

    @classmethod
    def from_geojson(Polygon, data_geojson: dict) -> Polygon:
        ...

    def to_shapely(self,
                   fix: bool = False) -> shapely.geometry.polygon.Polygon:
        ...

    def to_geojson(self) -> Dict[str, object]:
        ...

    def to_wkt(self) -> str:
        ...

    @classmethod
    def from_coco(cls,
                  data: List[Number] | Dict,
                  dims: None | Tuple[int, ...] = None) -> Polygon:
        ...

    def to_coco(self, style: str = 'orig') -> List | Dict:
        ...

    def to_multi_polygon(self) -> MultiPolygon:
        ...

    def to_boxes(self) -> kwimage.Boxes:
        ...

    @property
    def centroid(self) -> Tuple[Number, Number]:
        ...

    def to_box(self) -> kwimage.Box:
        ...

    def bounding_box(self) -> kwimage.Boxes:
        ...

    def box(self) -> kwimage.Box:
        ...

    def bounding_box_polygon(self) -> kwimage.Polygon:
        ...

    def copy(self) -> Polygon:
        ...

    def clip(self, x_min, y_min, x_max, y_max, inplace: bool = ...) -> Polygon:
        ...

    def fill(self,
             image: ndarray,
             value: int | Tuple[int] = 1,
             pixels_are: str = 'points',
             assert_inplace: bool = False) -> ndarray:
        ...

    def draw_on(self,
                image: ndarray,
                color: str | tuple = 'blue',
                fill: bool = True,
                border: bool = False,
                alpha: float = 1.0,
                edgecolor: str | tuple | None = None,
                facecolor: str | tuple | None = None,
                copy: bool = False) -> np.ndarray:
        ...

    def draw(
            self,
            color: str | Tuple = 'blue',
            ax: Incomplete | None = ...,
            alpha: float = 1.0,
            radius: int = ...,
            setlim: bool | str = False,
            border: bool | None = None,
            linewidth: bool | None = None,
            edgecolor: None | Any = None,
            facecolor: None | Any = None,
            fill: bool = True,
            vertex: float = False,
            vertexcolor: Any | None = None
    ) -> matplotlib.patches.PathPatch | None:
        ...

    def interpolate(self, other, alpha):
        ...

    def morph(self, other: kwimage.Polygon,
              alpha: float | List[float]) -> Polygon | List[Polygon]:
        ...


class MultiPolygon(_generic.ObjectList, _ShapelyMixin):

    @classmethod
    def random(self,
               n: int = ...,
               n_holes: int = ...,
               rng: Incomplete | None = ...,
               tight: bool = ...) -> MultiPolygon:
        ...

    def fill(self,
             image: ndarray,
             value: int | Tuple[int, ...] = 1,
             pixels_are: str = ...,
             assert_inplace: bool = ...) -> ndarray:
        ...

    def to_multi_polygon(self) -> MultiPolygon:
        ...

    def to_boxes(self) -> kwimage.Boxes:
        ...

    def to_box(self) -> kwimage.Box:
        ...

    def bounding_box(self) -> kwimage.Boxes:
        ...

    def box(self) -> kwimage.Box:
        ...

    def to_mask(self,
                dims: Incomplete | None = ...,
                pixels_are: str = ...) -> kwimage.Mask:
        ...

    def to_relative_mask(self, return_offset: bool = ...) -> kwimage.Mask:
        ...

    @classmethod
    def coerce(cls,
               data,
               dims: Incomplete | None = ...) -> None | MultiPolygon:
        ...

    def to_shapely(self, fix: bool = False) -> shapely.geometry.MultiPolygon:
        ...

    @classmethod
    def from_shapely(
        MultiPolygon,
        geom: shapely.geometry.MultiPolygon | shapely.geometry.Polygon
    ) -> MultiPolygon:
        ...

    @classmethod
    def from_geojson(MultiPolygon, data_geojson: Dict) -> MultiPolygon:
        ...

    def to_geojson(self) -> Dict:
        ...

    @classmethod
    def from_coco(cls,
                  data: List[List[Number] | Dict],
                  dims: None | Tuple[int, ...] = None) -> MultiPolygon:
        ...

    def to_coco(self, style: str = 'orig'):
        ...

    def swap_axes(self, inplace: bool = False) -> MultiPolygon:
        ...

    def draw_on(self, image, **kwargs):
        ...


class PolygonList(_generic.ObjectList):

    def to_mask_list(self,
                     dims: Incomplete | None = ...,
                     pixels_are: str = ...) -> kwimage.MaskList:
        ...

    def to_polygon_list(self) -> PolygonList:
        ...

    def to_segmentation_list(self) -> kwimage.SegmentationList:
        ...

    def swap_axes(self, inplace: bool = ...) -> PolygonList:
        ...

    def to_geojson(self, as_collection: bool = False) -> List[Dict] | Dict:
        ...

    def fill(self,
             image: ndarray,
             value: int | Tuple[int, ...] = 1,
             pixels_are: str = ...,
             assert_inplace: bool = ...) -> ndarray:
        ...

    def draw_on(self, *args, **kw):
        ...

    def unary_union(self):
        ...
