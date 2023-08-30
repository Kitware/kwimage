from nptyping import UInt8
from numpy import ndarray
from typing import Any
from typing import List
from typing import Dict
from nptyping import Shape
from nptyping import Integer
import kwimage
from numpy.typing import ArrayLike
import kwcoco
from typing import Callable
from typing import Tuple
from typing import Sequence
import torch
from nptyping import Bool
from numpy.random import RandomState
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator


class _DetDrawMixin:

    def draw(self,
             color: str = ...,
             alpha: Incomplete | None = ...,
             labels: bool = ...,
             centers: bool = ...,
             lw: int = ...,
             fill: bool = ...,
             ax: Incomplete | None = ...,
             radius: int = ...,
             kpts: bool = ...,
             sseg: bool = ...,
             setlim: bool = ...,
             boxes: bool = ...) -> None:
        ...

    def draw_on(self,
                image: ndarray[Any, UInt8] | None = None,
                color: str | Any | List[Any] = 'blue',
                alpha: float | None = None,
                labels: bool | str | List[str] = True,
                radius: float = 5,
                kpts: bool = True,
                sseg: bool = True,
                boxes: bool = True,
                ssegkw: dict | None = None,
                label_loc: str = 'top_left',
                thickness: int = 2) -> ndarray[Any, UInt8]:
        ...


class _DetAlgoMixin:

    def non_max_supression(
            self,
            thresh: float = 0.0,
            perclass: bool = False,
            impl: str = 'auto',
            daq: bool | Dict = False,
            device_id: Incomplete | None = ...
    ) -> ndarray[Shape['*'], Integer]:
        ...

    def non_max_supress(self,
                        thresh: float = ...,
                        perclass: bool = ...,
                        impl: str = ...,
                        daq: bool = ...):
        ...

    def rasterize(self,
                  bg_size,
                  input_dims,
                  soften: int = ...,
                  tf_data_to_img: Incomplete | None = ...,
                  img_dims: Incomplete | None = ...,
                  exclude=...) -> kwimage.Heatmap:
        ...


class Detections(ub.NiceRepr, _DetAlgoMixin, _DetDrawMixin):
    __datakeys__: Incomplete
    __metakeys__: Incomplete
    data: Dict
    meta: Dict

    def __init__(self,
                 data: Dict[str, ArrayLike] | None = None,
                 meta: Dict[str, object] | None = None,
                 datakeys: List[str] | None = None,
                 metakeys: List[str] | None = None,
                 checks: bool = True,
                 **kwargs) -> None:
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    def copy(self):
        ...

    @classmethod
    def coerce(cls, data: Incomplete | None = ..., **kwargs):
        ...

    @classmethod
    def from_coco_annots(cls,
                         anns: List[Dict],
                         cats: List[Dict] | None = None,
                         classes: kwcoco.CategoryTree | None = None,
                         kp_classes: kwcoco.CategoryTree | None = None,
                         shape: tuple | None = None,
                         dset: kwcoco.CocoDataset | None = None) -> Detections:
        ...

    def to_coco(
            self,
            cname_to_cat: Incomplete | None = ...,
            style: str = 'orig',
            image_id: int | None = None,
            dset: kwcoco.CocoDataset | None = None
    ) -> Generator[dict, None, None]:
        ...

    @property
    def boxes(self):
        ...

    @property
    def class_idxs(self):
        ...

    @property
    def scores(self):
        ...

    @property
    def probs(self):
        ...

    @property
    def weights(self):
        ...

    @property
    def classes(self):
        ...

    def num_boxes(self):
        ...

    def warp(self,
             transform: kwimage.Affine | ndarray | Callable | Any,
             input_dims: Tuple[int, int] | None = None,
             output_dims: Tuple[int, int] | None = None,
             inplace: bool = False) -> Detections:
        ...

    def scale(self,
              factor,
              output_dims: Incomplete | None = ...,
              inplace: bool = ...):
        ...

    def translate(self,
                  offset,
                  output_dims: Incomplete | None = ...,
                  inplace: bool = ...):
        ...

    @classmethod
    def concatenate(cls, dets) -> Detections:
        ...

    def argsort(self, reverse: bool = ...) -> torch.Tensor:
        ...

    def sort(self, reverse: bool = ...) -> kwimage.structs.Detections:
        ...

    def compress(self,
                 flags: ndarray[Any, Bool] | torch.Tensor,
                 axis: int = ...) -> kwimage.structs.Detections:
        ...

    def take(self,
             indices: ndarray[Any, Integer],
             axis: int = ...) -> kwimage.structs.Detections:
        ...

    def __getitem__(self, index):
        ...

    @property
    def device(self):
        ...

    def is_tensor(self):
        ...

    def is_numpy(self):
        ...

    def numpy(self):
        ...

    @property
    def dtype(self):
        ...

    def tensor(self, device=...):
        ...

    @classmethod
    def demo(Detections):
        ...

    @classmethod
    def random(cls,
               num: int = 10,
               scale: float | tuple = 1.0,
               classes: int | Sequence = 3,
               keypoints: bool = False,
               segmentations: bool = False,
               tensor: bool = False,
               rng: int | RandomState | None = None) -> Detections:
        ...


class _UnitDoctTests:
    ...
