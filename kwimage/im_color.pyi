from typing import Iterable
from numpy import ndarray
from typing import Tuple
from numbers import Number
from typing import List
import ubelt as ub
from _typeshed import Incomplete

__todo__: str
BASE_COLORS: Incomplete
TABLEAU_COLORS: Incomplete
XKCD_COLORS: Incomplete
CSS4_COLORS: Incomplete
KITWARE_COLORS: Incomplete


class Color(ub.NiceRepr):
    color01: Incomplete
    space: str

    def __init__(self,
                 color: Color | Iterable[int | float] | str,
                 alpha: float | None = None,
                 space: str | None = None,
                 coerce: bool = True) -> None:
        ...

    @classmethod
    def coerce(cls, data, **kwargs):
        ...

    def __nice__(self):
        ...

    def forimage(self,
                 image: ndarray,
                 space: str = 'auto') -> Tuple[Number, ...]:
        ...

    def ashex(self, space: None | str = None) -> str:
        ...

    def as255(
        self,
        space: None | str = None
    ) -> Tuple[int, int, int] | Tuple[int, int, int, int]:
        ...

    def as01(
        self,
        space: None | str = None
    ) -> Tuple[float, float, float] | Tuple[float, float, float, float]:
        ...

    @classmethod
    def named_colors(cls) -> List[str]:
        ...

    @classmethod
    def distinct(Color,
                 num,
                 existing: Incomplete | None = ...,
                 space: str = ...,
                 legacy: str = ...,
                 exclude_black: bool = ...,
                 exclude_white: bool = ...) -> List[Tuple]:
        ...

    @classmethod
    def random(Color,
               pool: str = ...,
               with_alpha: int = ...,
               rng: Incomplete | None = ...) -> Color:
        ...

    def distance(self, other: Color, space: str = 'lab') -> float:
        ...

    def interpolate(self,
                    other: Color,
                    alpha: float | List[float] = 0.5,
                    ispace: str | None = None,
                    ospace: str | None = None) -> Color | List[Color]:
        ...

    def to_image(self, dsize: Tuple[int, int] = ...):
        ...

    def adjust(self, saturate: float = 0, lighten: float = 0):
        ...
