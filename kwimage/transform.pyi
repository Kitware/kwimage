from numpy import ndarray
from typing import Dict
from typing import Union
from typing import Tuple
import ubelt as ub
from _typeshed import Incomplete

profile: Incomplete


class Transform(ub.NiceRepr):
    ...


class Matrix(Transform):
    matrix: Incomplete

    def __init__(self, matrix) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(self):
        ...

    def __json__(self):
        ...

    @classmethod
    def coerce(cls, data: Incomplete | None = ..., **kwargs):
        ...

    def __array__(self):
        ...

    def __imatmul__(self, other) -> None:
        ...

    def __matmul__(self, other):
        ...

    def inv(self) -> Matrix:
        ...

    @property
    def T(self):
        ...

    def det(self) -> float:
        ...

    @classmethod
    def eye(cls, shape: Incomplete | None = ..., rng: Incomplete | None = ...):
        ...

    @classmethod
    def random(cls,
               shape: Incomplete | None = ...,
               rng: Incomplete | None = ...):
        ...

    def __getitem__(self, index):
        ...

    def isclose_identity(self, rtol: float = ..., atol: float = ...):
        ...


class Linear(Matrix):
    ...


class Projective(Linear):

    @classmethod
    def fit(cls, pts1: ndarray, pts2: ndarray) -> Projective:
        ...


class Affine(Projective):

    @property
    def shape(self):
        ...

    def __json__(self):
        ...

    def concise(self) -> Dict:
        ...

    @classmethod
    def coerce(cls, data: Incomplete | None = ..., **kwargs) -> Affine:
        ...

    def eccentricity(self):
        ...

    def to_shapely(self):
        ...

    @classmethod
    def scale(cls, scale: Union[float, Tuple[float, float]]) -> Affine:
        ...

    @classmethod
    def translate(cls, offset: Union[float, Tuple[float, float]]) -> Affine:
        ...

    @classmethod
    def rotate(cls, theta: float) -> Affine:
        ...

    @classmethod
    def random(cls,
               shape: Incomplete | None = ...,
               rng: Incomplete | None = ...,
               **kw) -> Affine:
        ...

    @classmethod
    def random_params(cls, rng: Incomplete | None = ..., **kw) -> Dict:
        ...

    def decompose(self) -> Dict:
        ...

    @classmethod
    def affine(cls,
               scale: Union[float, Tuple[float, float]] = None,
               offset: Union[float, Tuple[float, float]] = None,
               theta: float = None,
               shear: float = None,
               about: Union[float, Tuple[float, float]] = None,
               shearx: float = None,
               **kwargs) -> Affine:
        ...

    @classmethod
    def fit(cls, pts1: ndarray, pts2: ndarray) -> Affine:
        ...
