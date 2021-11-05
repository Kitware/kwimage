"""
Objects for representing and manipulating image transforms.
"""
import ubelt as ub
import numpy as np
import kwarray
import skimage.transform
import math


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


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
        """
        Returns the inverse of this matrix

        Returns:
            Matrix
        """
        if self.matrix is None:
            return self.__class__(None)
        else:
            return self.__class__(np.linalg.inv(self.matrix))

    @property
    def T(self):
        """
        Transpose the underlying matrix
        """
        if self.matrix is None:
            return self
        else:
            return self.__class__(self.matrix.T)

    def det(self):
        """
        Compute the determinant of the underlying matrix

        Returns:
            float
        """
        if self.matrix is None:
            return 1.
        else:
            return np.linalg.det(self.matrix)

    @classmethod
    def eye(cls, shape=None, rng=None):
        """
        Construct an identity
        """
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

    @profile
    def concise(self):
        """
        Return a concise coercable dictionary representation of this matrix

        Returns:
            Dict[str, object]: a small serializable dict that can be passed
                to :func:`Affine.coerce` to reconstruct this object.

        Returns:
            Dict: dictionary with consise parameters

        Example:
            >>> import kwimage
            >>> self = kwimage.Affine.random(rng=0, scale=1)
            >>> params = self.concise()
            >>> assert np.allclose(Affine.coerce(params).matrix, self.matrix)
            >>> print('params = {}'.format(ub.repr2(params, nl=1, precision=2)))
            params = {
                'offset': (0.08, 0.38),
                'theta': 0.08,
                'type': 'affine',
            }

        Example:
            >>> import kwimage
            >>> self = kwimage.Affine.random(rng=0, scale=2, offset=0)
            >>> params = self.concise()
            >>> assert np.allclose(Affine.coerce(params).matrix, self.matrix)
            >>> print('params = {}'.format(ub.repr2(params, nl=1, precision=2)))
            params = {
                'scale': 2.00,
                'theta': 0.04,
                'type': 'affine',
            }
        """
        params = self.decompose()
        params['type'] = 'affine'
        # New much faster impl
        tx, ty = params['offset']
        sx, sy = params['scale']
        math.isclose(params['shear'], 0)
        math.isclose(params['shear'], 0)
        if math.isclose(tx, 0) and math.isclose(ty, 0):
            params.pop('offset')
        elif tx == ty:
            params['offset'] = tx
        if math.isclose(sy, 1) and math.isclose(sy, 1):
            params.pop('scale')
        elif sx == sy:
            params['scale'] = sx
        if math.isclose(params['shear'], 0):
            params.pop('shear')
        if math.isclose(params['theta'], 0):
            params.pop('theta')
        # else:
        #     if np.allclose(params['offset'], (0, 0)):
        #         params.pop('offset')
        #     elif ub.allsame(params['offset']):
        #         params['offset'] = params['offset'][0]
        #     if np.allclose(params['scale'], (1, 1)):
        #         params.pop('scale')
        #     elif ub.allsame(params['scale']):
        #         params['scale'] = params['scale'][0]
        #     if np.allclose(params['shear'], 0):
        #         params.pop('shear')
        #     if np.isclose(params['theta'], 0):
        #         params.pop('theta')
        return params

    @classmethod
    @profile
    def coerce(cls, data=None, **kwargs):
        """
        Attempt to coerce the data into an affine object

        Args:
            data : some data we attempt to coerce to an Affine matrix
            **kwargs : some data we attempt to coerce to an Affine matrix,
                mutually exclusive with `data`.

        Returns:
            Affine

        Example:
            >>> import kwimage
            >>> kwimage.Affine.coerce({'type': 'affine', 'matrix': [[1, 0, 0], [0, 1, 0]]})
            >>> kwimage.Affine.coerce({'scale': 2})
            >>> kwimage.Affine.coerce({'offset': 3})
            >>> kwimage.Affine.coerce(np.eye(3))
            >>> kwimage.Affine.coerce(None)
            >>> kwimage.Affine.coerce(skimage.transform.AffineTransform(scale=30))
        """
        if data is None and not kwargs:
            return cls(matrix=None)
        if data is None:
            data = kwargs
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        elif isinstance(data, cls):
            self = data
        elif isinstance(data, skimage.transform.AffineTransform):
            self = cls(matrix=data.params)
        elif data.__class__.__name__ == cls.__name__:
            self = data
        elif isinstance(data, dict):
            keys = set(data.keys())
            if 'matrix' in keys:
                self = cls(matrix=np.array(data['matrix']))
            else:
                known_params = {'scale', 'shear', 'offset', 'theta', 'type'}
                params = {key: data[key] for key in known_params if key in data}
                if len(known_params & keys):
                    params.pop('type', None)
                    self = cls.affine(**params)
                else:
                    raise KeyError(', '.join(list(data.keys())))
        else:
            raise TypeError(type(data))
        return self

    def eccentricity(self):
        """
        Eccentricity of the ellipse formed by this affine matrix

        References:
            https://en.wikipedia.org/wiki/Conic_section
            https://github.com/rasterio/affine/blob/78c20a0cfbb5ec/affine/__init__.py#L368
        """
        # Ignore the translation part
        M = self.matrix[0:2, 0:2]

        MMt = M @ M.T
        trace = np.trace(MMt)
        det = np.linalg.det(MMt)

        root_delta = np.sqrt((trace * trace) / 4 - det)

        # scaling defined via affine.Affine.
        ell1 = np.sqrt(trace / 2 + root_delta)
        ell2 = np.sqrt(trace / 2 - root_delta)

        ecc = np.sqrt(ell1 * ell1 - ell2 * ell2) / ell1
        return ecc

    @profile
    def decompose(self):
        """
        Decompose the affine matrix into its individual scale, translation,
        rotation, and skew parameters.

        Returns:
            Dict: decomposed offset, scale, theta, and shear params

        References:
            https://math.stackexchange.com/questions/612006/decompose-affine

        Example:
            >>> self = Affine.random()
            >>> params = self.decompose()
            >>> recon = Affine.coerce(**params)
            >>> params2 = recon.decompose()
            >>> pt = np.vstack([np.random.rand(2, 1), [1]])
            >>> result1 = self.matrix[0:2] @ pt
            >>> result2 = recon.matrix[0:2] @ pt
            >>> assert np.allclose(result1, result2)

            >>> self = Affine.scale(0.001) @ Affine.random()
            >>> params = self.decompose()
            >>> self.det()

        Ignore:
            import affine
            self = Affine.random()
            a, b, c, d, e, f = self.matrix.ravel()[0:6]
            aff = affine.Affine(a, b, c, d, e, f)
            assert np.isclose(self.det(), aff.determinant)

            params = self.decompose()
            assert np.isclose(params['theta'], np.deg2rad(aff.rotation_angle))

            print(params['scale'])
            print(aff._scaling)
            print(self.eccentricity())
            print(aff.eccentricity)

            pass

        Ignore:
            import timerit
            ti = timerit.Timerit(100, bestof=10, verbose=2)
            for timer in ti.reset('time'):
                self = Affine.random()
                with timer:
                    self.decompose()

            # Wow: using math instead of numpy for scalars is much faster!

            for timer in ti.reset('time'):
                with timer:
                    math.sqrt(a11 * a11 + a21 * a21)

            for timer in ti.reset('time'):
                with timer:
                    np.arctan2(a21, a11)

            for timer in ti.reset('time'):
                with timer:
                    math.atan2(a21, a11)
        """
        a11, a12, a13, a21, a22, a23 = self.matrix.ravel()[0:6]
        sx = math.sqrt(a11 * a11 + a21 * a21)
        theta = math.atan2(a21, a11)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        msy = a12 * cos_t + a22 * sin_t
        if abs(cos_t) < abs(sin_t):
            sy = (msy * cos_t - a12) / sin_t
        else:
            sy = (a22 - msy * sin_t) / cos_t
        m = msy / sy
        tx, ty = a13, a23
        params = {
            'offset': (tx, ty),
            'scale': (sx, sy),
            'shear': m,
            'theta': theta,
        }

        # a11, a12, a13, a21, a22, a23 = self.matrix.ravel()[0:6]
        # sx = np.sqrt(a11 ** 2 + a21 ** 2)
        # theta = np.arctan2(a21, a11)
        # sin_t = np.sin(theta)
        # cos_t = np.cos(theta)
        # msy = a12 * cos_t + a22 * sin_t
        # if abs(cos_t) < abs(sin_t):
        #     sy = (msy * cos_t - a12) / sin_t
        # else:
        #     sy = (a22 - msy * sin_t) / cos_t
        # m = msy / sy
        # tx, ty = a13, a23
        # params = {
        #     'offset': (tx, ty),
        #     'scale': (sx, sy),
        #     'shear': m,
        #     'theta': theta,
        # }
        return params

    @classmethod
    def scale(cls, scale):
        """
        Create a scale Affine object

        Args:
            scale (float | Tuple[float, float]): x, y scale factor

        Returns:
            Affine
        """
        scale_ = 1 if scale is None else scale
        sx, sy = _ensure_iterable2(scale_)
        # Sympy simplified expression
        mat = np.array([sx , 0.0, 0.0,
                        0.0,  sy, 0.0,
                        0.0, 0.0, 1.0])
        mat = mat.reshape(3, 3)  # Faster to make a flat array and reshape
        self = cls(mat)
        return self

    @classmethod
    def translate(cls, offset):
        """
        Create a translation Affine object

        Args:
            offset (float | Tuple[float, float]): x, y translation factor

        Returns:
            Affine

        Benchmark:
            >>> # xdoctest: +REQUIRES(--benchmark)
            >>> # It is ~3x faster to use the more specific method
            >>> import timerit
            >>> import kwimage
            >>> #
            >>> offset = np.random.rand(2)
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('time'):
            >>>     with timer:
            >>>         kwimage.Affine.translate(offset)
            >>> #
            >>> for timer in ti.reset('time'):
            >>>     with timer:
            >>>         kwimage.Affine.affine(offset=offset)

        """
        offset_ = 0 if offset is None else offset
        tx, ty = _ensure_iterable2(offset_)
        # Sympy simplified expression
        mat = np.array([1.0, 0.0, tx,
                        0.0, 1.0, ty,
                        0.0, 0.0, 1.0])
        mat = mat.reshape(3, 3)  # Faster to make a flat array and reshape
        self = cls(mat)
        return self

    @classmethod
    def rotate(cls, theta):
        """
        Create a rotation Affine object

        Args:
            theta (float): counter-clockwise rotation angle in radians

        Returns:
            Affine
        """
        return cls.affine(theta=theta)

    @classmethod
    def random(cls, rng=None, **kw):
        """
        Create a random Affine object

        Args:
            rng : random number generator
            **kw: passed to :func:`Affine.random_params`.
                can contain coercable random distributions for scale, offset,
                about, theta, and shear.

        Returns:
            Affine
        """
        params = cls.random_params(rng=rng, **kw)
        self = cls.affine(**params)
        return self

    @classmethod
    def random_params(cls, rng=None, **kw):
        """
        Args:
            rng : random number generator
            **kw: can contain coercable random distributions for
                scale, offset, about, theta, and shear.

        Returns:
            Dict: affine parameters suitable to be passed to Affine.affine

        TODO:
            - [ ] improve kwargs parameterization
        """
        from kwarray import distributions
        import numbers
        TN = distributions.TruncNormal
        rng = kwarray.ensure_rng(rng)

        def _coerce_distri(arg):
            if isinstance(arg, numbers.Number):
                dist = distributions.Constant(arg, rng=rng)
            elif isinstance(arg, tuple) and len(arg) == 2:
                lo, hi = arg
                dist = distributions.Uniform(lo, hi, rng=rng)
            else:
                raise NotImplementedError
            return dist

        if 'scale' in kw:
            if ub.iterable(kw['scale']):
                raise NotImplementedError
            else:
                xscale_dist = _coerce_distri(kw['scale'])
                yscale_dist = xscale_dist
        else:
            scale_kw = dict(mean=1, std=1, low=1, high=2)
            xscale_dist = TN(**scale_kw, rng=rng)
            yscale_dist = TN(**scale_kw, rng=rng)

        if 'offset' in kw:
            if ub.iterable(kw['offset']):
                raise NotImplementedError
            else:
                xoffset_dist = _coerce_distri(kw['offset'])
                yoffset_dist = xoffset_dist
        else:
            offset_kw = dict(mean=0, std=1, low=-1, high=1)
            xoffset_dist = TN(**offset_kw, rng=rng)
            yoffset_dist = TN(**offset_kw, rng=rng)

        if 'about' in kw:
            if ub.iterable(kw['about']):
                raise NotImplementedError
            else:
                xabout_dist = _coerce_distri(kw['about'])
                yabout_dist = xabout_dist
        else:
            xabout_dist = distributions.Constant(0, rng=rng)
            yabout_dist = distributions.Constant(0, rng=rng)

        if 'theta' in kw:
            theta_dist = _coerce_distri(kw['theta'])
        else:
            theta_kw = dict(mean=0, std=1, low=-np.pi / 8, high=np.pi / 8)
            theta_dist = TN(**theta_kw, rng=rng)

        if 'shear' in kw:
            shear_dist = _coerce_distri(kw['shear'])
        else:
            shear_dist = distributions.Constant(0, rng=rng)

        # scale_kw = dict(mean=1, std=1, low=0, high=2)
        # offset_kw = dict(mean=0, std=1, low=-1, high=1)
        # theta_kw = dict(mean=0, std=1, low=-6.28, high=6.28)

        # TODO: distributions.Distribution.coerce()
        # offset_dist = distributions.Constant(0)
        # theta_dist = distributions.Constant(0)

        # todo better parametarization
        params = dict(
            scale=(xscale_dist.sample(), yscale_dist.sample()),
            offset=(xoffset_dist.sample(), yoffset_dist.sample()),
            theta=theta_dist.sample(),
            shear=shear_dist.sample(),
            about=(xabout_dist.sample(), yabout_dist.sample()),
        )
        return params

    @classmethod
    @profile
    def affine(cls, scale=None, offset=None, theta=None, shear=None,
               about=None, **kwargs):
        """
        Create an affine matrix from high-level parameters

        Args:
            scale (float | Tuple[float, float]):
                x, y scale factor

            offset (float | Tuple[float, float]):
                x, y translation factor

            theta (float):
                counter-clockwise rotation angle in radians

            shear (float):
                counter-clockwise shear angle in radians

            about (float | Tuple[float, float]):
                x, y location of the origin

        TODO:
            - [ ] Add aliases -
                origin : alias for about
                rotation : alias for theta
                translation : alias for offset

        Returns:
            Affine: the constructed Affine object

        Example:
            >>> from kwimage.transform import *  # NOQA
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

        Bench:
            import timerit
            ti = timerit.Timerit(10000, bestof=10, verbose=2)
            for timer in ti.reset('time'):
                with timer:
                    self = kwimage.Affine.affine(scale=3, offset=2, theta=np.random.rand(), shear=np.random.rand())

        """
        scale_ = 1 if scale is None else scale
        offset_ = 0 if offset is None else offset
        shear_ = 0 if shear is None else shear
        theta_ = 0 if theta is None else theta
        about_ = 0 if about is None else about
        sx, sy = _ensure_iterable2(scale_)
        tx, ty = _ensure_iterable2(offset_)
        x0, y0 = _ensure_iterable2(about_)

        # Make auxially varables to reduce the number of sin/cos calls
        shear_p_theta = shear_ + theta_

        cos_theta = math.cos(theta_)
        sin_theta = math.sin(theta_)
        cos_shear_p_theta = math.cos(shear_p_theta)
        sin_shear_p_theta = math.sin(shear_p_theta)

        sx_cos_theta = sx * cos_theta
        sx_sin_theta = sx * sin_theta
        sy_sin_shear_p_theta = sy * sin_shear_p_theta
        sy_cos_shear_p_theta = sy * cos_shear_p_theta
        tx_ = tx + x0 - (x0 * sx_cos_theta) + (y0 * sy_sin_shear_p_theta)
        ty_ = ty + y0 - (x0 * sx_sin_theta) - (y0 * sy_cos_shear_p_theta)
        # Sympy simplified expression
        mat = np.array([sx_cos_theta, -sy_sin_shear_p_theta, tx_,
                        sx_sin_theta,  sy_cos_shear_p_theta, ty_,
                                   0,                     0,  1])
        mat = mat.reshape(3, 3)  # Faster to make a flat array and reshape
        self = cls(mat)
        return self


# def _ensure_iterablen(scalar, n):
#     try:
#         iter(scalar)
#     except TypeError:
#         return [scalar] * n
#     return scalar


def _ensure_iterable2(scalar):
    try:
        a, b = scalar
    except TypeError:
        a = b = scalar
    return a, b

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwimage.transform all --profile
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
