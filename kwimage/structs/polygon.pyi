from typing import Union
from typing import Callable
from skimage.transform._geometric import GeometricTransform
from numpy.typing import ArrayLike
from typing import Any
from typing import Tuple
import kwimage
import shapely
from typing import Dict
from typing import List
from numpy import ndarray
import numpy as np
import matplotlib
import ubelt as ub
from _typeshed import Incomplete
from kwimage.structs import _generic
from typing import Any


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
             transform: Union[GeometricTransform, ArrayLike, Any, Callable],
             input_dims: Tuple = None,
             output_dims: Tuple = None,
             inplace: bool = False):
        ...

    def scale(self,
              factor: Union[float, Tuple[float, float]],
              about: Union[Tuple, None] = None,
              output_dims: Tuple = None,
              inplace: bool = False):
        ...

    def translate(self,
                  offset,
                  output_dims: Tuple = None,
                  inplace: bool = False):
        ...

    def rotate(self,
               theta: float,
               about: Union[Tuple, None, str] = None,
               output_dims: Tuple = None,
               inplace: bool = ...):
        ...

    def swap_axes(self, inplace: bool = False) -> Polygon:
        ...


class Polygon(_generic.Spatial, _PolyArrayBackend, _PolyWarpMixin,
              ub.NiceRepr):
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
    def exterior(self):
        ...

    @property
    def interiors(self):
        ...

    def __nice__(self):
        ...

    @classmethod
    def circle(cls, xy, r, resolution: int = ...):
        ...

    @classmethod
    def random(cls,
               n: int = 6,
               n_holes: int = 0,
               convex: bool = True,
               tight: bool = False,
               rng: Incomplete | None = ...):
        ...

    def to_mask(self,
                dims: Tuple = None,
                pixels_are: str = ...) -> kwimage.Mask:
        ...

    def to_relative_mask(self) -> kwimage.Mask:
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
    def from_geojson(Polygon, data_geojson: dict):
        ...

    def to_shapely(self):
        ...

    @property
    def area(self):
        ...

    def to_geojson(self) -> Dict[str, object]:
        ...

    def to_wkt(self):
        ...

    @classmethod
    def from_coco(cls, data, dims: Incomplete | None = ...):
        ...

    def to_coco(self, style: str = ...) -> List | Dict:
        ...

    def to_multi_polygon(self):
        ...

    def to_boxes(self):
        ...

    @property
    def centroid(self):
        ...

    def bounding_box(self) -> kwimage.Boxes:
        ...

    def bounding_box_polygon(self) -> kwimage.Polygon:
        ...

    def copy(self):
        ...

    def clip(self, x_min, y_min, x_max, y_max, inplace: bool = ...):
        ...

    def fill(self,
             image: ndarray,
             value: Union[int, Tuple[int]] = 1,
             pixels_are: str = 'points') -> ndarray:
        ...

    def draw_on(self,
                image: ndarray,
                color: Union[str, tuple] = 'blue',
                fill: bool = True,
                border: bool = False,
                alpha: float = 1.0,
                edgecolor: Union[str, tuple] = None,
                facecolor: Union[str, tuple] = None,
                copy: bool = False) -> np.ndarray:
        ...

    def draw(self,
             color: Union[str, Tuple] = 'blue',
             ax: Incomplete | None = ...,
             alpha: float = 1.0,
             radius: int = ...,
             setlim: bool = False,
             border: bool = None,
             linewidth: bool = None,
             edgecolor: Union[None, Any] = None,
             facecolor: Union[None, Any] = None,
             fill: bool = True,
             vertex: float = False,
             vertexcolor: Any = None) -> matplotlib.patches.PathPatch | None:
        ...


class MultiPolygon(_generic.ObjectList):

    @property
    def area(self):
        ...

    @classmethod
    def random(self,
               n: int = ...,
               n_holes: int = ...,
               rng: Incomplete | None = ...,
               tight: bool = ...) -> MultiPolygon:
        ...

    def fill(self,
             image: ndarray,
             value: Union[int, Tuple[int, ...]] = 1,
             pixels_are: str = ...) -> ndarray:
        ...

    def to_multi_polygon(self):
        ...

    def to_boxes(self):
        ...

    def bounding_box(self) -> kwimage.Boxes:
        ...

    def to_mask(self, dims: Incomplete | None = ..., pixels_are: str = ...):
        ...

    def to_relative_mask(self) -> kwimage.Mask:
        ...

    @classmethod
    def coerce(cls, data, dims: Incomplete | None = ...):
        ...

    def to_shapely(self):
        ...

    @classmethod
    def from_shapely(MultiPolygon, geom):
        ...

    @classmethod
    def from_geojson(MultiPolygon, data_geojson):
        ...

    def to_geojson(self):
        ...

    @classmethod
    def from_coco(cls, data, dims: Incomplete | None = ...):
        ...

    def to_coco(self, style: str = ...):
        ...

    def swap_axes(self, inplace: bool = ...):
        ...

    def draw_on(self, image, **kwargs):
        ...


class PolygonList(_generic.ObjectList):

    def to_mask_list(self,
                     dims: Incomplete | None = ...,
                     pixels_are: str = ...):
        ...

    def to_polygon_list(self):
        ...

    def to_segmentation_list(self):
        ...

    def swap_axes(self, inplace: bool = ...):
        ...

    def to_geojson(self, as_collection: bool = False) -> List[Dict] | Dict:
        ...

    def fill(self,
             image: ndarray,
             value: Union[int, Tuple[int, ...]] = 1,
             pixels_are: str = ...) -> ndarray:
        ...

    def draw_on(self, *args, **kw):
        ...
