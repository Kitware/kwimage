"""
Generic segmentation object that can use either a Mask or (Multi)Polygon
backend.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

# from kwimage.structs import _generic
import numpy as np
import ubelt as ub

from . import _generic

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from numbers import Number
    from typing import Literal, overload

    from matplotlib.patches import PathPatch
    from numpy import ndarray
    from torch import Tensor

    import kwimage
    from kwimage._typing import TorchDeviceLike
    from kwimage.structs.mask import CocoMaskRLE
    from kwimage.structs.polygon import CocoPolygon

    SegmentationBackend = kwimage.Mask | kwimage.Polygon | kwimage.MultiPolygon
    SegmentationFormat = Literal['mask', 'polygon', 'multipolygon']
    SegmentationCoco = CocoMaskRLE | CocoPolygon | list[CocoPolygon]


class _WrapperObject(ub.NiceRepr):
    if TYPE_CHECKING:
        data: SegmentationBackend

        def draw_on(
            self, image: ndarray | None = None, **kw: Any
        ) -> ndarray: ...

        def warp(
            self,
            transform: ndarray | kwimage.Affine | None,
            input_dims: tuple[int, int] | None = None,
            output_dims: tuple[int, int] | None = None,
            inplace: bool = False,
        ) -> SegmentationBackend: ...

        def translate(
            self,
            offset: Number | tuple[Number, Number],
            output_dims: tuple[int, int] | None = None,
            inplace: bool = False,
        ) -> SegmentationBackend: ...

        def scale(
            self,
            factor: float | tuple[float, float],
            output_dims: tuple[int, int] | None = None,
            inplace: bool = False,
        ) -> SegmentationBackend: ...

        def to_coco(self, style: str = 'orig') -> SegmentationCoco: ...
        def numpy(self) -> SegmentationBackend: ...

        @overload
        def tensor(self) -> SegmentationBackend: ...

        @overload
        def tensor(
            self, device: TorchDeviceLike
        ) -> SegmentationBackend: ...

    def __nice__(self) -> str:
        data: Any = self.data
        return data.__nice__()

    def draw(
        self, *args: Any, **kw: Any
    ) -> PathPatch | list[PathPatch | None] | None:
        data: Any = self.data
        return data.draw(*args, **kw)

    if not TYPE_CHECKING:
        def draw_on(self, *args: Any, **kw: Any) -> ndarray:
            """See help(self.data.draw_on)"""
            return self.data.draw_on(*args, **kw)

        def warp(self, *args: Any, **kw: Any) -> SegmentationBackend:
            return self.data.warp(*args, **kw)

        def translate(self, *args: Any, **kw: Any) -> SegmentationBackend:
            return self.data.translate(*args, **kw)

        def scale(self, *args: Any, **kw: Any) -> SegmentationBackend:
            return self.data.scale(*args, **kw)

        def to_coco(self, *args: Any, **kw: Any) -> SegmentationCoco:
            return self.data.to_coco(*args, **kw)

        def numpy(self, *args: Any, **kw: Any) -> SegmentationBackend:
            return self.data.numpy(*args, **kw)

        def tensor(self, *args: Any, **kw: Any) -> SegmentationBackend:
            return self.data.tensor(*args, **kw)


class Segmentation(_WrapperObject):
    """
    Either holds a MultiPolygon, Polygon, or Mask

    Args:
        data (object): the underlying object
        format (str): either 'mask', 'polygon', or 'multipolygon'
    """

    data: SegmentationBackend
    format: SegmentationFormat | None

    def __init__(
        self, data: SegmentationBackend, format: SegmentationFormat | None = None
    ) -> None:
        self.data = data
        self.format = format

    @classmethod
    def random(cls, rng: Any | None = None) -> Segmentation:
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:cv2)
            >>> self = Segmentation.random()
            >>> print('self = {!r}'.format(self))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> self.draw()
            >>> kwplot.show_if_requested()
        """

        import kwarray

        import kwimage

        rng = kwarray.ensure_rng(rng)
        if rng.rand() > 0.5:
            data: Any = kwimage.Polygon.random()
        else:
            data: Any = kwimage.Mask.random()
        result: Any = cls.coerce(data)
        return result

    def to_multi_polygon(self) -> kwimage.MultiPolygon:
        return self.data.to_multi_polygon()

    def to_mask(
        self, dims: tuple[int, int] | None = None, pixels_are: str = 'points'
    ) -> kwimage.Mask:
        return self.data.to_mask(dims=dims, pixels_are=pixels_are)

    def box(self) -> kwimage.Box:
        return self.data.box()

    @property
    def area(self) -> Number | Tensor:
        return self.data.area

    @property
    def meta(self) -> Mapping[str, Any]:
        data: Any = self.data
        return data.meta

    if TYPE_CHECKING:
        @classmethod
        @overload
        def coerce(
            cls,
            data: Segmentation | SegmentationBackend,
            dims: tuple[int, int] | None = None,
        ) -> Segmentation: ...

        @classmethod
        @overload
        def coerce(
            cls, data: Any, dims: tuple[int, int] | None = None
        ) -> Segmentation | None: ...

    @classmethod
    def coerce(
        cls, data: Any, dims: tuple[int, int] | None = None
    ) -> Segmentation | None:
        import kwimage

        if _generic._isinstance2(data, kwimage.Segmentation):
            self = data
        elif _generic._isinstance2(data, kwimage.Mask):
            self = Segmentation(data, 'mask')
        elif _generic._isinstance2(data, kwimage.Polygon):
            self = Segmentation(data, 'polygon')
        elif _generic._isinstance2(data, kwimage.MultiPolygon):
            self = Segmentation(data, 'multipolygon')
        else:
            data: Any = _coerce_coco_segmentation(data, dims=dims)
            if data is None:
                return None
            self = cls.coerce(data, dims=dims)
        return self


class SegmentationList(_generic.ObjectList[Segmentation | None]):
    if TYPE_CHECKING:
        def to_coco(
            self, style: str = 'orig'
        ) -> Iterator[SegmentationCoco | None]: ...

    """
    Store and manipulate multiple segmentations (masks or polygons), usually
    within the same image
    """

    def to_polygon_list(self) -> kwimage.PolygonList:
        """
        Converts all mask objects to multi-polygon objects
        """
        import kwimage

        new = kwimage.PolygonList(
            [None if item is None else item.to_multi_polygon() for item in self]
        )
        return new

    def to_mask_list(
        self, dims: tuple[int, int] | None = None, pixels_are: str = 'points'
    ) -> kwimage.MaskList:
        """
        Converts all mask objects to multi-polygon objects
        """
        import kwimage

        new = kwimage.MaskList(
            [
                None
                if item is None
                else item.to_mask(dims=dims, pixels_are=pixels_are)
                for item in self
            ]
        )
        return new

    def to_segmentation_list(self) -> SegmentationList:
        return self

    @classmethod
    def coerce(
        cls, data: list[Any] | _generic.ObjectList[Any] | None, none_policy: str = 'raise'
    ) -> SegmentationList | None | float:
        """
        Interpret data as a list of Segmentations

        Args:
            none_policy (str):
                Determines how to handle None inputs.
                Can be: 'return-None', or 'raise'.
        """
        if isinstance(data, (list, _generic.ObjectList)):
            data = [
                None if item is None else Segmentation.coerce(item)
                for item in data
            ]
        else:
            if data is None:
                return _handle_null_policy(none_policy)
            else:
                raise TypeError(data)
        self = cls(data)
        return self


def _handle_null_policy(
    policy: str, ex_type: type[Exception] = TypeError, ex_msg: str = 'cannot accept null input'
) -> None | float:
    """
    For handling a nan or None policy.

    Args:
        policy (str):
            How null inputs are handled. Can be:
                'return-None': returns None
                'return-nan': returns nan
                'raise': raises an error

        ex_type (type): Exception type to raise if policy is raise

        ex_msg (msg): Exception arguments

    TODO: rectify with similar logic in kwutil/util_time
    """
    if policy == 'return-None':
        return None
    elif policy == 'return-nan':
        return float('nan')
    elif policy == 'raise':
        raise ex_type(ex_msg)
    else:
        raise KeyError(
            ub.paragraph(
                f"""
            Unknown null policy={policy!r}.
            Valid choices are "return-None", "return-nan", and "raise".
            """
            )
        )


def _coerce_coco_segmentation(
    data: Any, dims: tuple[int, int] | None = None
) -> SegmentationBackend | Segmentation | None:
    """
    Attempts to auto-inspect the format of segmentation data

    Args:
        data : the data to coerce

             2D-C-ndarray -> C_MASK
             2D-F-ndarray -> F_MASK

             Dict(counts=bytes) -> BYTES_RLE
             Dict(counts=ndarray) -> ARRAY_RLE

             Dict(exterior=ndarray) -> ARRAY_RLE

             # List[List[int]] -> Polygon
             List[int] -> Polygon
             List[Dict] -> MultPolygon

        dims (Tuple): required for certain formats like polygons
            height / width of the source image

    TODO:
        - [ ] Handle WKT

    Returns:
        Mask | Polygon | MultiPolygon | Segmentation - depending on which is appropriate

    Example:
        >>> segmentation = {'size': [5, 9], 'counts': ';?1B10O30O4'}
        >>> dims = (9, 5)
        >>> raw_mask = (np.random.rand(32, 32) > .5).astype(np.uint8)
        >>> _coerce_coco_segmentation(segmentation)
        >>> _coerce_coco_segmentation(raw_mask)

        >>> coco_polygon = [
        >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
        >>>     np.array([[2, 1],[2, 2],[4, 2],[4, 1]]),
        >>> ]
        >>> self = _coerce_coco_segmentation(coco_polygon, dims)
        >>> print('self = {!r}'.format(self))
        >>> coco_polygon = [
        >>>     np.array([[3, 0],[2, 1],[2, 4],[4, 4],[4, 3],[7, 0]]),
        >>> ]
        >>> self = _coerce_coco_segmentation(coco_polygon, dims)
        >>> print('self = {!r}'.format(self))
    """
    import kwimage
    from kwimage.structs.mask import MaskFormat

    self: Any
    if isinstance(data, np.ndarray):
        # INPUT TYPE: RAW MASK
        if dims is not None:
            assert dims == data.shape[0:2]
        if data.flags['F_CONTIGUOUS']:
            self = kwimage.Mask(data, MaskFormat.F_MASK)
        else:
            self = kwimage.Mask(data, MaskFormat.C_MASK)
    elif isinstance(data, dict):
        if 'counts' in data:
            # INPUT TYPE: COCO RLE DICTIONARY
            if dims is not None:
                data_shape = data.get(
                    'dims', data.get('shape', data.get('size', None))
                )
                if data_shape is None:
                    data['shape'] = dims
                else:
                    assert tuple(map(int, dims)) == tuple(
                        map(int, data_shape)
                    ), '{} {}'.format(dims, data_shape)
            if isinstance(data['counts'], (str, bytes)):
                self = kwimage.Mask(data, MaskFormat.BYTES_RLE)
            else:
                self = kwimage.Mask(data, MaskFormat.ARRAY_RLE)
        elif 'exterior' in data:
            # TODO: kwimage.Polygon.from_coco
            self = kwimage.Polygon(**data)
            # raise NotImplementedError('explicit polygon coerce')
        elif 'type' in data:
            if data['type'] == 'Polygon':
                self = kwimage.Polygon.from_geojson(data)
            elif data['type'] == 'MultiPolygon':
                self = kwimage.MultiPolygon.from_geojson(data)
            else:
                raise NotImplementedError(data['type'])
        else:
            raise TypeError(
                'Unable to interpret dictionary format {}'.format(data)
            )
    elif isinstance(data, list):
        # THIS IS NOT AN IDEAL FORMAT. IDEALLY WE WILL MODIFY COCO TO USE
        # DICTIONARIES FOR POLYGONS, WHICH ARE UNAMBIGUOUS
        if len(data) == 0:
            self = None
        else:
            first = ub.peek(data)
            if isinstance(first, dict):
                # TODO: kwimage.MultiPolygon.from_coco
                self = kwimage.MultiPolygon(
                    [kwimage.Polygon(**item) for item in data]
                )
            elif isinstance(first, numbers.Number):
                # TODO: kwimage.Polygon.from_coco
                exterior = np.array(data).reshape(-1, 2)
                self = kwimage.Polygon(exterior=exterior)
            elif isinstance(first, list):
                # TODO: kwimage.MultiPolygon.from_coco
                poly_list = [
                    kwimage.Polygon(exterior=np.array(item).reshape(-1, 2))
                    for item in data
                ]
                if len(poly_list) == 1:
                    self = poly_list[0]
                else:
                    self = kwimage.MultiPolygon(poly_list)
            elif isinstance(first, np.ndarray):
                poly_list = [
                    kwimage.Polygon(exterior=item.reshape(-1, 2))
                    for item in data
                ]
                if len(poly_list) == 1:
                    self = poly_list[0]
                else:
                    self = kwimage.MultiPolygon(poly_list)
            else:
                raise TypeError(
                    'Unable to interpret list format {}'.format(data)
                )
    elif isinstance(
        data,
        (
            kwimage.Polygon,
            kwimage.MultiPolygon,
            kwimage.Mask,
            kwimage.Segmentation,
        ),
    ):
        self = data
    else:
        from shapely.geometry.multipolygon import MultiPolygon
        from shapely.geometry.polygon import Polygon

        if isinstance(data, MultiPolygon):
            self = kwimage.MultiPolygon.from_shapely(data)
        elif isinstance(data, Polygon):
            self = kwimage.Polygon.from_shapely(data)
        else:
            raise TypeError(
                f'Unable to coerce type={type(data)!r} into a segmentation. data={data!r}'
            )
    return self
