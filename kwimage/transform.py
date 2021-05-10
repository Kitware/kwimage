"""
Objects for representing and manipulating image transforms.
"""
import ubelt as ub
import numpy as np
import kwarray


class Transform(ub.NiceRepr):
    pass


class Matrix(Transform):
    """
    Base class for matrix-based transform.

    Example:
        >>> from kwimage.transform import *  # NOQA
        >>> ms = {}
        >>> ms['random()'] = Matrix.random()
        >>> ms['eye()'] = Matrix.eye()
        >>> ms['random(3)'] = Matrix.random(3)
        >>> ms['random(4, 4)'] = Matrix.random(4, 4)
        >>> ms['eye(3)'] = Matrix.eye(3)
        >>> ms['explicit'] = Matrix(np.array([[1.618]]))
        >>> for k, m in ms.items():
        >>>     print('----')
        >>>     print(f'{k} = {m}')
        >>>     print(f'{k}.inv() = {m.inv()}')
        >>>     print(f'{k}.T = {m.T}')
        >>>     print(f'{k}.det() = {m.det()}')
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def __nice__(self):
        return repr(self.matrix)

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        if self.matrix is None:
            # Default shape is hard coded here, can be overrided (e.g. Affine)
            return (2, 2)
        return self.matrix.shape

    def __json__(self):
        if self.matrix is None:
            return {'type': 'matrix', 'matrix': None}
        else:
            return {'type': 'matrix', 'matrix': self.matrix.tolist()}

    @classmethod
    def coerce(cls, data=None, **kwargs):
        """
        Example:
            >>> Matrix.coerce({'type': 'matrix', 'matrix': [[1, 0, 0], [0, 1, 0]]})
            >>> Matrix.coerce(np.eye(3))
            >>> Matrix.coerce(None)
        """
        if data is None and not kwargs:
            return cls(matrix=None)
        if data is None:
            data = kwargs
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        elif isinstance(data, cls):
            self = data
        elif data.__class__.__name__ == cls.__name__:
            self = data
        elif isinstance(data, dict):
            keys = set(data.keys())
            if 'matrix' in keys:
                self = cls(matrix=np.array(data['matrix']))
            else:
                raise KeyError(', '.join(list(data.keys())))
        else:
            raise TypeError(type(data))
        return self

    def __array__(self):
        """
        Allow this object to be passed to np.asarray

        References:
            https://numpy.org/doc/stable/user/basics.dispatch.html
        """
        if self.matrix is None:
            return np.eye(*self.shape)
        return self.matrix

    def __imatmul__(self, other):
        if isinstance(other, np.ndarray):
            other_matrix = other
        else:
            other_matrix = other.matrix
        if self.matrix is None:
            self.matrix = other_matrix
        else:
            self.matrix @= other_matrix

    def __matmul__(self, other):
        """
        Example:
            >>> m = {}
            >>> # Works, and returns a Matrix
            >>> m[len(m)] = x = Matrix.random() @ np.eye(2)
            >>> assert isinstance(x, Matrix)
            >>> m[len(m)] = x = Matrix.random() @ None
            >>> assert isinstance(x, Matrix)
            >>> # Works, and returns an ndarray
            >>> m[len(m)] = x = np.eye(3) @ Matrix.random(3)
            >>> assert isinstance(x, np.ndarray)
            >>> # These do not work
            >>> # m[len(m)] = None @ Matrix.random()
            >>> # m[len(m)] = np.eye(3) @ None
            >>> print('m = {}'.format(ub.repr2(m)))
        """
        if other is None:
            return self
        if self.matrix is None:
            return self.__class__.coerce(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.matrix @ other)
        elif other.matrix is None:
            return self
        elif isinstance(other, self.__class__):
            # Prefer using the type of the left-hand-side, but try
            # not to break group rules.
            return self.__class__(self.matrix @ other.matrix)
        elif isinstance(self, other.__class__):
            return other.__class__(self.matrix @ other.matrix)
        else:
            raise TypeError('{} @ {}'.format(type(self), type(other)))

    def inv(self):
        if self.matrix is None:
            return self.__class__(None)
        else:
            return self.__class__(np.linalg.inv(self.matrix))

    @property
    def T(self):
        if self.matrix is None:
            return self
        else:
            return self.__class__(self.matrix.T)

    def det(self):
        if self.matrix is None:
            return 1
        else:
            return np.linalg.det(self.matrix)

    @classmethod
    def eye(cls, shape=None, rng=None):
        self = cls(None)
        if isinstance(shape, int):
            shape = (shape, shape)
        if shape is None:
            shape = self.shape
        self.matrix = np.eye(*shape)
        return self

    @classmethod
    def random(cls, shape=None, rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)
        self = cls(None)
        if isinstance(shape, int):
            shape = (shape, shape)
        if shape is None:
            shape = self.shape
        self.matrix = rng.rand(*shape)
        return self


class Linear(Matrix):
    pass


class Projective(Linear):
    """
    Currently just a stub class that may be used to implement projective /
    homography transforms in the future.
    """
    pass


class Affine(Projective):
    """
    Helper for making affine transform matrices.

    Example:
        >>> self = Affine(np.eye(3))
        >>> m1 = np.eye(3) @ self
        >>> m2 = self @ np.eye(3)

    Example:
        >>> from kwimage.transform import *  # NOQA
        >>> m = {}
        >>> # Works, and returns a Affine
        >>> m[len(m)] = x = Affine.random() @ np.eye(3)
        >>> assert isinstance(x, Affine)
        >>> m[len(m)] = x = Affine.random() @ None
        >>> assert isinstance(x, Affine)
        >>> # Works, and returns an ndarray
        >>> m[len(m)] = x = np.eye(3) @ Affine.random(3)
        >>> assert isinstance(x, np.ndarray)
        >>> # Works, and returns an Matrix
        >>> m[len(m)] = x = Affine.random() @ Matrix.random(3)
        >>> assert isinstance(x, Matrix)
        >>> m[len(m)] = x = Matrix.random(3) @ Affine.random()
        >>> assert isinstance(x, Matrix)
        >>> print('m = {}'.format(ub.repr2(m)))
    """
    @property
    def shape(self):
        return (3, 3)

    def __getitem__(self, index):
        if self.matrix is None:
            return np.asarray(self)[index]
        return self.matrix[index]

    def __json__(self):
        if self.matrix is None:
            return {'type': 'affine', 'matrix': None}
        else:
            return {'type': 'affine', 'matrix': self.matrix.tolist()}

    @classmethod
    def coerce(cls, data=None, **kwargs):
        """
        Example:
            >>> Affine.coerce({'type': 'affine', 'matrix': [[1, 0, 0], [0, 1, 0]]})
            >>> Affine.coerce({'scale': 2})
            >>> Affine.coerce({'offset': 3})
            >>> Affine.coerce(np.eye(3))
            >>> Affine.coerce(None)
        """
        if data is None and not kwargs:
            return cls(matrix=None)
        if data is None:
            data = kwargs
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        elif isinstance(data, cls):
            self = data
        elif data.__class__.__name__ == cls.__name__:
            self = data
        elif isinstance(data, dict):
            keys = set(data.keys())
            known_params = {'scale', 'shear', 'offset', 'theta'}
            params = ub.dict_isect(data, known_params)
            if 'matrix' in keys:
                self = cls(matrix=np.array(data['matrix']))
            elif len(known_params & keys):
                # data.pop('type', None)
                self = cls.affine(**params)
            else:
                raise KeyError(', '.join(list(data.keys())))
        else:
            raise TypeError(type(data))
        return self

    def decompose(self):
        raise NotImplementedError

    @classmethod
    def scale(cls, scale):
        return cls.affine(scale=scale)

    @classmethod
    def translate(cls, offset):
        return cls.affine(offset=offset)

    @classmethod
    def rotate(cls, theta):
        return cls.affine(theta=theta)

    @classmethod
    def random(cls, rng=None, **kw):
        params = cls.random_params(rng=rng, **kw)
        self = cls.affine(**params)
        return self

    @classmethod
    def random_params(cls, rng=None, **kw):
        """
        TODO: improve kwarg parameterization
        """
        from kwarray import distributions
        TN = distributions.TruncNormal
        rng = kwarray.ensure_rng(rng)

        # scale_kw = dict(mean=1, std=1, low=0, high=2)
        # offset_kw = dict(mean=0, std=1, low=-1, high=1)
        # theta_kw = dict(mean=0, std=1, low=-6.28, high=6.28)
        scale_kw = dict(mean=1, std=1, low=1, high=2)
        offset_kw = dict(mean=0, std=1, low=-1, high=1)
        theta_kw = dict(mean=0, std=1, low=-np.pi / 8, high=np.pi / 8)

        scale_dist = TN(**scale_kw, rng=rng)
        offset_dist = TN(**offset_kw, rng=rng)
        theta_dist = TN(**theta_kw, rng=rng)

        # TODO: distributions.Distribution.coerce()
        # offset_dist = distributions.Constant(0)
        # theta_dist = distributions.Constant(0)

        # todo better parametarization
        params = dict(
            scale=scale_dist.sample(2),
            offset=offset_dist.sample(2),
            theta=theta_dist.sample(),
            shear=0,
            about=0,
        )
        return params

    @classmethod
    def affine(cls, scale=None, offset=None, theta=None, shear=None,
               about=None):
        """
        Create an affine matrix from high-level parameters

        Args:
            scale (float | Tuple[float, float]): x, y scale factor
            offset (float | Tuple[float, float]): x, y translation factor
            theta (float): counter-clockwise rotation angle in radians
            shear (float): counter-clockwise shear angle in radians
            about (float | Tuple[float, float]): x, y location of the origin

        Example:
            >>> rng = kwarray.ensure_rng(None)
            >>> scale = rng.randn(2) * 10
            >>> offset = rng.randn(2) * 10
            >>> about = rng.randn(2) * 10
            >>> theta = rng.randn() * 10
            >>> shear = rng.randn() * 10
            >>> # Create combined matrix from all params
            >>> F = Affine.affine(
            >>>     scale=scale, offset=offset, theta=theta, shear=shear,
            >>>     about=about)
            >>> # Test that combining components matches
            >>> S = Affine.affine(scale=scale)
            >>> T = Affine.affine(offset=offset)
            >>> R = Affine.affine(theta=theta)
            >>> H = Affine.affine(shear=shear)
            >>> O = Affine.affine(offset=about)
            >>> # combine (note shear must be on the RHS of rotation)
            >>> alt  = O @ T @ R @ H @ S @ O.inv()
            >>> print('F    = {}'.format(ub.repr2(F.matrix.tolist(), nl=1)))
            >>> print('alt  = {}'.format(ub.repr2(alt.matrix.tolist(), nl=1)))
            >>> assert np.all(np.isclose(alt.matrix, F.matrix))
            >>> pt = np.vstack([np.random.rand(2, 1), [[1]]])
            >>> warp_pt1 = (F.matrix @ pt)
            >>> warp_pt2 = (alt.matrix @ pt)
            >>> assert np.allclose(warp_pt2, warp_pt1)

        Sympy:
            >>> # xdoctest: +SKIP
            >>> import sympy
            >>> # Shows the symbolic construction of the code
            >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
            >>> from sympy.abc import theta
            >>> x0, y0, sx, sy, theta, shear, tx, ty = sympy.symbols(
            >>>     'x0, y0, sx, sy, theta, shear, tx, ty')
            >>> # move the center to 0, 0
            >>> tr1_ = np.array([[1, 0,  -x0],
            >>>                  [0, 1,  -y0],
            >>>                  [0, 0,    1]])
            >>> # Define core components of the affine transform
            >>> S = np.array([  # scale
            >>>     [sx,  0, 0],
            >>>     [ 0, sy, 0],
            >>>     [ 0,  0, 1]])
            >>> H = np.array([  # shear
            >>>     [1, -sympy.sin(shear), 0],
            >>>     [0,  sympy.cos(shear), 0],
            >>>     [0,                 0, 1]])
            >>> R = np.array([  # rotation
            >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
            >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
            >>>     [               0,                 0, 1]])
            >>> T = np.array([  # translation
            >>>     [ 1,  0, tx],
            >>>     [ 0,  1, ty],
            >>>     [ 0,  0,  1]])
            >>> # Contruct the affine 3x3 about the origin
            >>> aff0 = np.array(sympy.simplify(T @ R @ H @ S))
            >>> # move 0, 0 back to the specified origin
            >>> tr2_ = np.array([[1, 0,  x0],
            >>>                  [0, 1,  y0],
            >>>                  [0, 0,   1]])
            >>> # combine transformations
            >>> aff = tr2_ @ aff0 @ tr1_
            >>> print('aff = {}'.format(ub.repr2(aff.tolist(), nl=1)))
        """
        scale_ = 1 if scale is None else scale
        offset_ = 0 if offset is None else offset
        shear_ = 0 if shear is None else shear
        theta_ = 0 if theta is None else theta
        about_ = 0 if about is None else about
        sx, sy = _ensure_iterablen(scale_, 2)
        tx, ty = _ensure_iterablen(offset_, 2)
        x0, y0 = _ensure_iterablen(about_, 2)

        # Make auxially varables to reduce the number of sin/cos calls
        cos_theta = np.cos(theta_)
        sin_theta = np.sin(theta_)
        cos_shear_p_theta = np.cos(shear_ + theta_)
        sin_shear_p_theta = np.sin(shear_ + theta_)
        sx_cos_theta = sx * cos_theta
        sx_sin_theta = sx * sin_theta
        sy_sin_shear_p_theta = sy * sin_shear_p_theta
        sy_cos_shear_p_theta = sy * cos_shear_p_theta
        tx_ = tx + x0 - (x0 * sx_cos_theta) + (y0 * sy_sin_shear_p_theta)
        ty_ = ty + y0 - (x0 * sx_sin_theta) - (y0 * sy_cos_shear_p_theta)
        # Sympy simplified expression
        mat = np.array([[sx_cos_theta, -sy_sin_shear_p_theta, tx_],
                        [sx_sin_theta,  sy_cos_shear_p_theta, ty_],
                        [           0,                     0,  1]])
        self = cls(mat)
        return self


def _ensure_iterablen(scalar, n):
    try:
        iter(scalar)
    except TypeError:
        return [scalar] * n
    return scalar
