from typing import List
import ubelt as ub
from _typeshed import Incomplete


class Color(ub.NiceRepr):
    color01: Incomplete
    space: Incomplete

    def __init__(self,
                 color,
                 alpha: Incomplete | None = ...,
                 space: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    def ashex(self, space: Incomplete | None = ...):
        ...

    def as255(self, space: Incomplete | None = ...):
        ...

    def as01(self, space: Incomplete | None = ...):
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
                 exclude_white: bool = ...):
        ...

    @classmethod
    def random(Color, pool: str = ...):
        ...

    def distance(self, other, space: str = ...):
        ...


BASE_COLORS: Incomplete
TABLEAU_COLORS: Incomplete
XKCD_COLORS: Incomplete
CSS4_COLORS: Incomplete
KITWARE_COLORS: Incomplete
