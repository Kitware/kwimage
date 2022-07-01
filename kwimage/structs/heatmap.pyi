from typing import Union
from numpy import ndarray
from numpy.typing import ArrayLike
from typing import Sequence
from typing import Tuple
import kwimage
from . import _generic
from _typeshed import Incomplete


class _HeatmapDrawMixin:

    def colorize(self,
                 channel: Union[int, str] = None,
                 invert: bool = ...,
                 with_alpha: float = ...,
                 interpolation: str = ...,
                 imgspace: bool = False,
                 cmap: Incomplete | None = ...):
        ...

    def draw_stacked(self,
                     image: Incomplete | None = ...,
                     dsize=...,
                     ignore_class_idxs=...,
                     top: Incomplete | None = ...,
                     chosen_cxs: Incomplete | None = ...):
        ...

    def draw(self,
             channel: Union[int, str] = None,
             image: Incomplete | None = ...,
             imgspace: Incomplete | None = ...,
             **kwargs) -> None:
        ...

    def draw_on(self,
                image: ndarray = None,
                channel: Union[int, str] = None,
                invert: bool = ...,
                with_alpha: float = ...,
                interpolation: str = ...,
                vecs: bool = ...,
                kpts: Incomplete | None = ...,
                imgspace: bool = None):
        ...


class _HeatmapWarpMixin:

    def upscale(self,
                channel: Incomplete | None = ...,
                interpolation: str = ...):
        ...

    def warp(self,
             mat: ArrayLike = None,
             input_dims: tuple = None,
             output_dims: tuple = None,
             interpolation: str = 'linear',
             modify_spatial_coords: bool = ...,
             int_interpolation: str = 'nearest',
             mat_is_xy: bool = True,
             version: Incomplete | None = ...) -> Heatmap:
        ...

    def scale(self,
              factor,
              output_dims: Incomplete | None = ...,
              interpolation: str = ...):
        ...

    def translate(self,
                  offset,
                  output_dims: Incomplete | None = ...,
                  interpolation: str = ...):
        ...


class _HeatmapAlgoMixin:

    @classmethod
    def combine(cls,
                heatmaps: Sequence[Heatmap],
                root_index: int = None,
                dtype=...) -> Heatmap:
        ...

    def detect(self,
               channel: Union[int, ArrayLike],
               invert: bool = False,
               min_score: float = 0.01,
               num_min: int = 10,
               max_dims: Tuple[int, int] = None,
               min_dims: Tuple[int, int] = None,
               dim_thresh_space: str = 'image') -> kwimage.Detections:
        ...


class Heatmap(_generic.Spatial, _HeatmapDrawMixin, _HeatmapWarpMixin,
              _HeatmapAlgoMixin):
    __datakeys__: Incomplete
    __metakeys__: Incomplete
    __spatialkeys__: Incomplete
    data: Incomplete
    meta: Incomplete

    def __init__(self,
                 data: Incomplete | None = ...,
                 meta: Incomplete | None = ...,
                 **kwargs) -> None:
        ...

    def __nice__(self):
        ...

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def bounds(self):
        ...

    @property
    def dims(self):
        ...

    def is_numpy(self):
        ...

    def is_tensor(self):
        ...

    @classmethod
    def random(cls,
               dims: Tuple = ...,
               classes: int = ...,
               diameter: bool = ...,
               offset: bool = ...,
               keypoints: bool = ...,
               img_dims: Tuple = None,
               dets: Incomplete | None = ...,
               nblips: int = ...,
               noise: float = ...,
               rng: Incomplete | None = ...,
               ensure_background: bool = ...) -> Heatmap:
        ...

    @property
    def class_probs(self):
        ...

    @property
    def offset(self):
        ...

    @property
    def diameter(self):
        ...

    @property
    def img_dims(self):
        ...

    @property
    def tf_data_to_img(self):
        ...

    @property
    def classes(self):
        ...

    def numpy(self):
        ...

    def tensor(self, device=...):
        ...


def smooth_prob(prob, k: int = ..., inplace: bool = ..., eps: float = ...):
    ...
