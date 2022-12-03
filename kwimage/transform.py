"""
Objects for representing and manipulating image transforms.
"""
import ubelt as ub
import numpy as np
import kwarray
import skimage.transform
import math
from kwimage import _internal

# try:
#     import xdev
#     profile = xdev.profile
# except Exception:
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
        prefix = '<{}('.format(self.__class__.__name__)
        if isinstance(self.matrix, np.ndarray):
            return np.array2string(self.matrix, separator=', ', prefix=prefix)
        else:
            return ub.repr2(self.matrix.tolist(), nl=1)

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
        Allow this object to be passed to np.asarray. See [NumpyDispatch]_ for
        details.

        References:
            ..[NumpyDispatch] https://numpy.org/doc/stable/user/basics.dispatch.html
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

        Example:
            >>> # Test with rationals
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> import kwimage
            >>> a = kwimage.Matrix.random((3, 3)).rationalize()
            >>> b = kwimage.Matrix.random((3, 3)).rationalize()
            >>> c = kwimage.Matrix.random((3, 3))
            >>> assert not c.is_rational()
            >>> assert (a @ c).is_rational()
            >>> assert (c @ a).is_rational()

        Ignore:
            %timeit c.inv()
            %timeit a.inv()
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

    def is_rational(self):
        """
        TODO: rename to "is_symbolic"
        """
        if sympy is None:
            return False
        return isinstance(self.matrix, sympy.Matrix) and isinstance(self.matrix[0, 0], (sympy.Rational, sympy.core.symbol.Symbol, sympy.core.basic.Basic))

    def inv(self):
        """
        Returns the inverse of this matrix

        Returns:
            Matrix

        Example:
            >>> # Test with rationals
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> import kwimage
            >>> self = kwimage.Matrix.random((3, 3)).rationalize()
            >>> inv = self.inv()
            >>> eye = self @ inv
            >>> eye.isclose_identity(0, 0)
        """
        if self.matrix is None:
            return self.__class__(None)
        else:
            try:
                inv_mat = np.linalg.inv(self.matrix)
            except (np.linalg.LinAlgError, np.core._exceptions.UFuncTypeError):
                if self.is_rational():
                    # inv_mat = mp.inverse(self.matrix)
                    # handle object arrays (rationals)
                    inv_mat = self.matrix.inv()
                else:
                    raise
            return self.__class__(inv_mat)

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
            try:
                det = np.linalg.det(self.matrix)
            except np.core._exceptions.UFuncTypeError:
                # handle object arrays (rationals)
                det = self.matrix.det()
            return det

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

    def __getitem__(self, index):
        if self.matrix is None:
            return np.asarray(self)[index]
        return self.matrix[index]

    def rationalize(self):
        """
        Convert the underlying matrix to a rational type to avoid floating
        point errors. This does decrease efficiency.

        TODO:
            - [ ] mpmath for arbitrary precision? It doesn't seem to do
            inverses correct whereas sympy does.

        Ignore:
            from sympy import Rational
            from fractions import Fraction
            float_mat = np.random.rand(3, 3)
            float_num = float_mat[0, 0]
            frac_num = Fraction(float_num)
            rat_num = Rational(float_num)

            rat_mat_v1 = float_mat.astype(Rational, subok=False)
            frac_mat_v1 = float_mat.astype(Fraction, subok=False)
            rat_num_v1 = rat_mat_v1[0, 0]
            frac_num_v1 = frac_mat_v1[0, 0]
            print(f'{type(float_num)=}')
            print(f'{type(rat_num)=}')
            print(f'{type(frac_num)=}')
            print(f'{type(rat_num_v1)=}')
            print(f'{type(frac_num_v1)=}')

            flat_rat = list(map(Rational, float_mat.ravel().tolist()))
            from sympy import Matrix
            rat_mat_v2 = Matrix(flat_rat).reshape(*float_mat.shape)

        Example:
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> import kwimage
            >>> self = kwimage.Matrix.random((3, 3))
            >>> mat = self.rationalize()
            >>> mat2 = kwimage.Matrix.random((3, 3))
            >>> mat3 = mat @ mat2
            >>> #assert 'sympy' in mat3.matrix.__class__.__module__
            >>> mat3 = mat2 @ mat
            >>> #assert 'sympy' in mat3.matrix.__class__.__module__
            >>> assert not mat.isclose_identity()
            >>> assert (mat @ mat.inv()).isclose_identity(rtol=0, atol=0)
        """
        if self.matrix is None:
            new_mat = self.matrix
        else:
            new_mat = _RationalNDArray.from_numpy(self.matrix)
        new = self.__class__(new_mat)
        return new

    def astype(self, dtype):
        """
        Convert the underlying matrix to a rational type to avoid floating
        point errors. This does decrease efficiency.

        Args:
            dtype (type):
        """
        if self.matrix is None:
            new_mat = self.matrix
        else:
            new_mat = self.matrix.astype(dtype)
        new = self.__class__(new_mat)
        return new

    def isclose_identity(self, rtol=1e-05, atol=1e-08):
        """
        Returns true if the matrix is nearly the identity.
        """
        if self.matrix is None:
            return True
        else:
            eye = np.eye(*self.matrix.shape)
            try:
                return np.allclose(self.matrix, eye, rtol=rtol, atol=atol)
            except TypeError:
                if self.is_rational():
                    residual = np.array(self.matrix - eye).astype(float)
                    return np.allclose(residual, 0, rtol=rtol, atol=atol)
                else:
                    raise


class Linear(Matrix):
    pass


class Projective(Linear):
    """
    A thin wraper around a 3x3 matrix that represent a projective transform

    Implements methods for:
        * creating random projective transforms
        * decomposing the matrix
        * finding a best-fit transform between corresponding points
        * TODO: - [ ] fully rational transform

    Example:
        >>> import kwimage
        >>> import math
        >>> image = kwimage.grab_test_image()
        >>> theta = 0.123 * math.tau
        >>> components = {
        >>>     'rotate': kwimage.Projective.projective(theta=theta),
        >>>     'scale': kwimage.Projective.projective(scale=0.5),
        >>>     'shear': kwimage.Projective.projective(shearx=0.2),
        >>>     'translation': kwimage.Projective.projective(offset=(100, 200)),
        >>>     'rotate+translate': kwimage.Projective.projective(theta=0.123 * math.tau, about=(256, 256)),
        >>>     'perspective': kwimage.Projective.projective(uv=(0.0003, 0.0007)),
        >>>     'random-composed': kwimage.Projective.random(scale=(0.5, 1.5), translate=(-20, 20), theta=(-theta, theta), shearx=(0, .4), rng=900558176210808600),
        >>> }
        >>> warp_stack = []
        >>> for key, mat in components.items():
        ...     warp = kwimage.warp_projective(image, mat)
        ...     warp = kwimage.draw_text_on_image(
        ...        warp,
        ...        ub.repr2(mat.matrix, nl=1, nobr=1, precision=4, si=1, sv=1, with_dtype=0),
        ...        org=(1, 1),
        ...        valign='top', halign='left',
        ...        fontScale=0.8, color='kw_green',
        ...        border={'thickness': 3},
        ...        )
        ...     warp = kwimage.draw_header_text(warp, key, color='kw_blue')
        ...     warp_stack.append(warp)
        >>> warp_canvas = kwimage.stack_images_grid(warp_stack, chunksize=4, pad=10, bg_value='kitware_gray')
        >>> # xdoctest: +REQUIRES(module:sympy)
        >>> import sympy
        >>> # Shows the symbolic construction of the code
        >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        >>> from sympy.abc import theta
        >>> params = x0, y0, sx, sy, theta, shearx, tx, ty, u, v = sympy.symbols(
        >>>     'x0, y0, sx, sy, theta, ex, tx, ty, u, v')
        >>> # move the center to 0, 0
        >>> tr1_ = sympy.Matrix([[1, 0,  -x0],
        >>>                      [0, 1,  -y0],
        >>>                      [0, 0,    1]])
        >>> P = sympy.Matrix([  # projective part
        >>>     [ 1,  0,  0],
        >>>     [ 0,  1,  0],
        >>>     [ u,  v,  1]])
        >>> # Define core components of the affine transform
        >>> S = sympy.Matrix([  # scale
        >>>     [sx,  0, 0],
        >>>     [ 0, sy, 0],
        >>>     [ 0,  0, 1]])
        >>> E = sympy.Matrix([  # x-shear
        >>>     [1,  shearx, 0],
        >>>     [0,  1, 0],
        >>>     [0,  0, 1]])
        >>> R = sympy.Matrix([  # rotation
        >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
        >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
        >>>     [               0,                 0, 1]])
        >>> T = sympy.Matrix([  # translation
        >>>     [ 1,  0, tx],
        >>>     [ 0,  1, ty],
        >>>     [ 0,  0,  1]])
        >>> # move 0, 0 back to the specified origin
        >>> tr2_ = sympy.Matrix([[1, 0,  x0],
        >>>                      [0, 1,  y0],
        >>>                      [0, 0,   1]])
        >>> # combine transformations
        >>> homog_ = sympy.MatMul(tr2_, T, R, E, S, P, tr1_)
        >>> #with sympy.evaluate(False):
        >>> #    homog_ = sympy.MatMul(tr2_, T, R, E, S, P, tr1_)
        >>> #    sympy.pprint(homog_)
        >>> homog = homog_.doit()
        >>> #sympy.pprint(homog)
        >>> print('homog = {}'.format(ub.repr2(homog.tolist(), nl=1)))
        >>> # This could be prettier
        >>> texts = {
        >>>     'Translation': sympy.pretty(R, use_unicode=0),
        >>>     'Rotation': sympy.pretty(R, use_unicode=0),
        >>>     'shEar-X': sympy.pretty(E, use_unicode=0),
        >>>     'Scale': sympy.pretty(S, use_unicode=0),
        >>>     'Perspective': sympy.pretty(P, use_unicode=0),
        >>> }
        >>> print(ub.repr2(texts, nl=2, sv=1))
        >>> equation_stack = []
        >>> for text, m in texts.items():
        >>>     render_canvas = kwimage.draw_text_on_image(None, m, color='kw_green', fontScale=1.0)
        >>>     render_canvas = kwimage.draw_header_text(render_canvas, text, color='kw_blue')
        >>>     render_canvas = kwimage.imresize(render_canvas, scale=1.3)
        >>>     equation_stack.append(render_canvas)
        >>> equation_canvas = kwimage.stack_images(equation_stack, pad=10, axis=1, bg_value='kitware_gray')
        >>> render_canvas = kwimage.draw_text_on_image(None, sympy.pretty(homog, use_unicode=0), color='kw_green', fontScale=1.0)
        >>> render_canvas = kwimage.draw_header_text(render_canvas, 'Full Equation With Pre-Shift', color='kw_blue')
        >>> # xdoctest: -REQUIRES(module:sympy)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> canvas = kwimage.stack_images([warp_canvas, equation_canvas, render_canvas], pad=20, axis=0, bg_value='kitware_gray', resize='larger')
        >>> canvas = kwimage.draw_header_text(canvas, 'Projective matrixes can represent', color='kw_blue')
        >>> kwplot.imshow(canvas)
        >>> fig = plt.gcf()
        >>> fig.set_size_inches(13, 13)
    """
    # References:
    #     .. [AffineDecompColab] https://colab.research.google.com/drive/1ImBB-N6P9zlNMCBH9evHD6tjk0dzvy1_

    @classmethod
    def fit(cls, pts1, pts2):
        """
        Fit an projective transformation between a set of corresponding points.

        See [HomogEst]_ [SzeleskiBook]_ and [RansacDummies]_ for references on
        the subject.

        Args:
            pts1 (ndarray): An Nx2 array of points in "space 1".
            pts2 (ndarray): A corresponding Nx2 array of points in "space 2"

        Returns:
            Projective : a transform that warps from "space1" to "space2".

        Note:
            A projective matrix has 8 degrees of freedome, so at least 8 point
            pairs are needed.

        References:
            .. [HomogEst] http://dip.sun.ac.za/~stefan/TW793/attach/notes/homography_estimation.pdf
            .. [SzeleskiBook] http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf Page 317
            .. [RansacDummies] http://vision.ece.ucsb.edu/~zuliani/Research/RANSAC/docs/RANSAC4Dummies.pdf page 53

        Example:
            >>> # Create a set of points, warp them, then recover the warp
            >>> import kwimage
            >>> points = kwimage.Points.random(9).scale(64)
            >>> A1 = kwimage.Affine.affine(scale=0.9, theta=-3.2, offset=(2, 3), about=(32, 32), skew=2.3)
            >>> A2 = kwimage.Affine.affine(scale=0.8, theta=0.8, offset=(2, 0), about=(32, 32))
            >>> A12_real = A2 @ A1.inv()
            >>> points1 = points.warp(A1)
            >>> points2 = points.warp(A2)
            >>> # Make the correspondence non-affine
            >>> points2.data['xy'].data[0, 0] += 3.5
            >>> points2.data['xy'].data[3, 1] += 8.5
            >>> # Recover the warp
            >>> pts1, pts2 = points1.xy, points2.xy
            >>> A_recovered = kwimage.Projective.fit(pts1, pts2)
            >>> #assert np.all(np.isclose(A_recovered.matrix, A12_real.matrix))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import cv2
            >>> import kwplot
            >>> kwplot.autompl()
            >>> base1 = np.zeros((96, 96, 3))
            >>> base1[32:-32, 5:-5] = 0.5
            >>> base2 = np.zeros((96, 96, 3))
            >>> img1 = points1.draw_on(base1, radius=3, color='blue')
            >>> img2 = points2.draw_on(base2, radius=3, color='green')
            >>> img1_warp = kwimage.warp_projective(img1, A_recovered.matrix, dsize=img1.shape[0:2][::-1])
            >>> canvas = kwimage.stack_images([img1, img2, img1_warp], pad=10, axis=1, bg_value=(1., 1., 1.))
            >>> kwplot.imshow(canvas)
        """
        if 0:
            import cv2
            inlier_method = 'all'
            inlier_method_lut = {
                'all': 0,
                'lmeds': cv2.LMEDS,
                'ransac': cv2.RANSAC,
                'prosac': cv2.RHO,
            }
            cv2_method = inlier_method_lut[inlier_method]
            # This probably does the point normaliztion internally,
            # but I'm not sure
            H, mask = cv2.findHomography(pts1, pts2, method=cv2_method)
            return Projective(H)
        else:
            def whiten_xy_points(xy_m):
                """
                whitens points to mean=0, stddev=1 and returns transformation
                """
                mu_xy  = xy_m.mean(axis=1)  # center of mass
                std_xy = xy_m.std(axis=1)
                std_xy[std_xy == 0] = 1  # prevent divide by zero
                tx, ty = -mu_xy / std_xy
                sx, sy = 1 / std_xy
                T = np.array([(sx, 0, tx),
                              (0, sy, ty),
                              (0,  0,  1)])
                xy_norm = ((xy_m.T - mu_xy) / std_xy).T
                return xy_norm, T
            # Hartley Precondition (to reduce sensitivity to noise)
            xy1_mn, T1 = whiten_xy_points(pts1.T)
            xy2_mn, T2 = whiten_xy_points(pts2.T)
            # xy1_mn = pts1.T
            # xy2_mn = pts2.T
            x1_mn = xy1_mn[0]
            y1_mn = xy1_mn[1]
            x2_mn = xy2_mn[0]
            y2_mn = xy2_mn[1]
            num_pts = x1_mn.shape[0]
            # Concatenate all 2x9 matrices into an Mx9 matrix
            Mx9 = np.empty((2 * num_pts, 9), dtype=float)
            for ix in range(num_pts):
                u2        = x2_mn[ix]
                v2        = y2_mn[ix]
                x1        = x1_mn[ix]
                y1        = y1_mn[ix]
                (d, e, f) = (     -x1,      -y1,  -1)
                (g, h, i) = ( v2 * x1,  v2 * y1,  v2)
                (j, k, l) = (      x1,       y1,   1)
                (p, q, r) = (-u2 * x1, -u2 * y1, -u2)
                Mx9[ix * 2]     = (0, 0, 0, d, e, f, g, h, i)
                Mx9[ix * 2 + 1] = (j, k, l, 0, 0, 0, p, q, r)
            M = (Mx9.T @ Mx9)
            # M = Mx9
            try:
                # https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem/2173715#2173715
                # http://twistedoakstudios.com/blog/Post7254_visualizing-the-eigenvectors-of-a-rotation
                USVt = np.linalg.svd(M, full_matrices=True, compute_uv=True)
            except MemoryError:
                import scipy.sparse as sps
                import scipy.sparse.linalg as spsl
                M_sparse = sps.lil_matrix(M)
                USVt = spsl.svds(M_sparse)
            except np.linalg.LinAlgError:
                raise
            except Exception:
                raise
            # U is the co-domain unitary matrix
            # V is the domain unitary matrix
            # s contains the singular values
            U, s, Vt = USVt
            # The column of V (row of Vt) corresponding to the lowest singular
            # value is the solution to the least squares problem
            H_prime = Vt[8].reshape(3, 3)

            # Then compute ax = b  [aka: x = npl.solve(a, b)]
            M = np.linalg.inv(T2) @ H_prime @ T1  # Unnormalize
            # homographies that only differ by a scale factor are equivalent
            M /= M[2, 2]
            return Projective(M)

    @classmethod
    def projective(cls, scale=None, offset=None, shearx=None, theta=None,
                   uv=None, about=None):
        """
        Reconstruct from parameters

        Sympy:
            >>> # xdoctest: +SKIP
            >>> import sympy
            >>> # Shows the symbolic construction of the code
            >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
            >>> from sympy.abc import theta
            >>> params = x0, y0, sx, sy, theta, shearx, tx, ty, u, v = sympy.symbols(
            >>>     'x0, y0, sx, sy, theta, ex, tx, ty, u, v')
            >>> # move the center to 0, 0
            >>> tr1_ = sympy.Matrix([[1, 0,  -x0],
            >>>                      [0, 1,  -y0],
            >>>                      [0, 0,    1]])
            >>> P = sympy.Matrix([  # projective part
            >>>     [ 1,  0,  0],
            >>>     [ 0,  1,  0],
            >>>     [ u,  v,  1]])
            >>> # Define core components of the affine transform
            >>> S = sympy.Matrix([  # scale
            >>>     [sx,  0, 0],
            >>>     [ 0, sy, 0],
            >>>     [ 0,  0, 1]])
            >>> E = sympy.Matrix([  # x-shear
            >>>     [1,  shearx, 0],
            >>>     [0,  1, 0],
            >>>     [0,  0, 1]])
            >>> R = sympy.Matrix([  # rotation
            >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
            >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
            >>>     [               0,                 0, 1]])
            >>> T = sympy.Matrix([  # translation
            >>>     [ 1,  0, tx],
            >>>     [ 0,  1, ty],
            >>>     [ 0,  0,  1]])
            >>> # move 0, 0 back to the specified origin
            >>> tr2_ = sympy.Matrix([[1, 0,  x0],
            >>>                      [0, 1,  y0],
            >>>                      [0, 0,   1]])
            >>> # combine transformations
            >>> with sympy.evaluate(False):
            >>>     homog_ = sympy.MatMul(tr2_, T, R, E, S, P, tr1_)
            >>>     sympy.pprint(homog_)
            >>> homog = homog_.doit()
            >>> sympy.pprint(homog)
            >>> print('homog = {}'.format(ub.repr2(homog.tolist(), nl=1)))

        Ignore:
            M = kwimage.Projective.projective(uv=(0, 0.04), about=128)
            img1 = kwimage.ensure_float01(kwimage.grab_test_image('astro', dsize=(228, 228)))
            points = kwimage.Points(xy=kwimage.Coords(np.array([
                (0, 0),
                (1, 1),
                (128, 1),
                (144, 0),
                (288, 288),
                (97, 77),
                (0, 288),
                (128, 128),
            ])))
            img1_warp = kwimage.warp_projective(img1, M.matrix, interpolation='linear')
            warped = points.warp(M.matrix)

            pic_copy = points.draw_on(img1.copy(), radius=10, color='kitware_green')
            warp_copy = warped.draw_on(img1_warp.copy(), radius=10, color='kitware_green')
            stacked, stack_tfs = kwimage.stack_images([pic_copy, warp_copy], return_info=True, axis=1)
            stacked = kwimage.draw_line_segments_on_image(stacked, points.warp(stack_tfs[0]).xy, warped.warp(stack_tfs[1]).xy)

            import kwplot
            kwplot.autompl()
            kwplot.imshow(stacked)
            # M.matrix.dot(np.array([[0, 0, 1]]).T)
        """
        import kwimage
        about_ = 0 if about is None else about
        if uv is None:
            uv = 0, 0
        x0, y0 = _ensure_iterable2(about_)
        # About needs to be wrt to this because the projective and affine parts
        # will be inside it.
        tr1_ = np.array([[1, 0,  -x0],
                         [0, 1,  -y0],
                         [0, 0,    1]])
        tr2_ = np.array([[1, 0,  x0],
                         [0, 1,  y0],
                         [0, 0,   1]])
        # TODO: add sympy optimization
        aff_part = kwimage.Affine.affine(
            scale=scale, offset=offset, shearx=shearx, theta=theta)
        u, v = uv
        proj_part = np.array([
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ u,  v,  1],
        ])
        self = kwimage.Projective(tr2_ @ aff_part.matrix @ proj_part @ tr1_)
        return self

    @classmethod
    @profile
    def coerce(cls, data=None, **kwargs):
        """
        Attempt to coerce the data into an Projective object

        Args:
            data : some data we attempt to coerce to an Projective matrix
            **kwargs : some data we attempt to coerce to an Projective matrix,
                mutually exclusive with `data`.

        Returns:
            Projective

        Example:
            >>> import kwimage
            >>> kwimage.Projective.coerce({'type': 'affine', 'matrix': [[1, 0, 0], [0, 1, 0]]})
            >>> kwimage.Projective.coerce({'type': 'affine', 'scale': 2})
            >>> kwimage.Projective.coerce({'type': 'projective', 'scale': 2})
            >>> kwimage.Projective.coerce({'scale': 2})
            >>> kwimage.Projective.coerce({'offset': 3})
            >>> kwimage.Projective.coerce(np.eye(3))
            >>> kwimage.Projective.coerce(None)
            >>> import skimage
            >>> kwimage.Projective.coerce(skimage.transform.AffineTransform(scale=30))
            >>> kwimage.Projective.coerce(skimage.transform.ProjectiveTransform(matrix=None))
        """
        if data is None and not kwargs:
            return cls(matrix=None)
        if data is None:
            data = kwargs
        if isinstance(data, np.ndarray):
            self = cls(matrix=data)
        elif isinstance(data, cls):
            self = data
        elif isinstance(data, (skimage.transform.AffineTransform, skimage.transform.ProjectiveTransform)):
            self = cls(matrix=data.params)
        elif data.__class__.__name__ == cls.__name__:
            self = data
        elif isinstance(data, dict):
            keys = set(data.keys())
            if 'matrix' in keys:
                matrix = np.array(data['matrix'])
                if matrix.shape[0] == 2:
                    matrix = np.vstack([matrix, [[0, 0, 1.]]])
                self = cls(matrix=matrix)
            else:
                known_params = {'uv', 'scale', 'offset', 'theta', 'type', 'shearx', 'shear', 'about'}
                params = {key: data[key] for key in known_params if key in data}
                if len(known_params & keys):
                    type_ = params.pop('type', None)
                    type_
                    self = cls.projective(**params)
                else:
                    raise KeyError(', '.join(list(data.keys())))
        else:
            raise TypeError(type(data))
        return self

    def is_affine(self):
        """
        If the bottom row is [[0, 0, 1]], then this can be safely turned into
        an affine matrix.

        Returns:
            bool

        Example:
            >>> import kwimage
            >>> kwimage.Projective.coerce(scale=2, uv=[1, 1]).is_affine()
            False
            >>> kwimage.Projective.coerce(scale=2, uv=[0, 0]).is_affine()
            True
        """
        if self.matrix is None:
            return True
        else:
            return np.all(self.matrix[2] == [0, 0, 1])

    def to_skimage(self):
        """
        Returns:
            skimage.transform.AffineTransform

        Example:
            >>> import kwimage
            >>> self = kwimage.Projective.random()
            >>> tf = self.to_skimage()
            >>> # Transform points with kwimage and scikit-image
            >>> kw_poly = kwimage.Polygon.random()
            >>> kw_warp_xy = kw_poly.warp(self.matrix).exterior.data
            >>> sk_warp_xy = tf(kw_poly.exterior.data)
            >>> assert np.allclose(sk_warp_xy, sk_warp_xy)
        """
        return skimage.transform.ProjectiveTransform(matrix=np.asarray(self))

    @classmethod
    def random(cls, shape=None, rng=None, **kw):
        """
        Example/
            >>> import kwimage
            >>> self = kwimage.Projective.random()
            >>> print(f'self={self}')
            >>> params = self.decompose()
            >>> aff_part = kwimage.Affine.affine(**ub.dict_diff(params, ['uv']))
            >>> proj_part = kwimage.Projective.coerce(uv=params['uv'])
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import cv2
            >>> import kwplot
            >>> dsize = (256, 256)
            >>> kwplot.autompl()
            >>> img1 = kwimage.grab_test_image(dsize=dsize)
            >>> img1_affonly = kwimage.warp_projective(img1, aff_part.matrix, dsize=img1.shape[0:2][::-1])
            >>> img1_projonly = kwimage.warp_projective(img1, proj_part.matrix, dsize=img1.shape[0:2][::-1])
            >>> ###
            >>> img2 = kwimage.ensure_uint255(kwimage.atleast_3channels(kwimage.checkerboard(dsize=dsize)))
            >>> img1_fullwarp = kwimage.warp_projective(img1, self.matrix, dsize=img1.shape[0:2][::-1])
            >>> img2_affonly = kwimage.warp_projective(img2, aff_part.matrix, dsize=img2.shape[0:2][::-1])
            >>> img2_projonly = kwimage.warp_projective(img2, proj_part.matrix, dsize=img2.shape[0:2][::-1])
            >>> img2_fullwarp = kwimage.warp_projective(img2, self.matrix, dsize=img2.shape[0:2][::-1])
            >>> canvas1 = kwimage.stack_images([img1, img1_projonly, img1_affonly, img1_fullwarp], pad=10, axis=1, bg_value=(0.5, 0.9, 0.1))
            >>> canvas2 = kwimage.stack_images([img2, img2_projonly, img2_affonly, img2_fullwarp], pad=10, axis=1, bg_value=(0.5, 0.9, 0.1))
            >>> canvas = kwimage.stack_images([canvas1, canvas2], axis=0)
            >>> kwplot.imshow(canvas)
        """
        import kwimage
        rng = kwarray.ensure_rng(rng)
        aff_part = kwimage.Affine.random(shape, rng=rng, **kw)
        # Random projective part
        u = 1 / rng.randint(0, 10000)
        v = 1 / rng.randint(0, 10000)
        proj_part = np.array([
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ u,  v,  1],
        ])
        self = Projective(aff_part.matrix @ proj_part)
        return self

    def decompose(self):
        r"""
        Based on the analysis done in [ME1319680]_.

        Returns:
            Dict:

        References:
            .. [ME1319680] https://math.stackexchange.com/questions/1319680

        Example:
            >>> # Create a set of points, warp them, then recover the warp
            >>> import kwimage
            >>> points = kwimage.Points.random(9).scale(64)
            >>> A1 = kwimage.Affine.affine(scale=0.9, theta=-3.2, offset=(2, 3), about=(32, 32), skew=2.3)
            >>> A2 = kwimage.Affine.affine(scale=0.8, theta=0.8, offset=(2, 0), about=(32, 32))
            >>> A12_real = A2 @ A1.inv()
            >>> points1 = points.warp(A1)
            >>> points2 = points.warp(A2)
            >>> # Make the correspondence non-affine
            >>> points2.data['xy'].data[0, 0] += 3.5
            >>> points2.data['xy'].data[3, 1] += 8.5
            >>> # Recover the warp
            >>> pts1, pts2 = points1.xy, points2.xy
            >>> self = kwimage.Projective.random()
            >>> self.decompose()

        Example:
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> from kwimage.transform import *  # NOQA
            >>> import kwimage
            >>> from kwimage.transform import _RationalNDArray
            >>> self = kwimage.Projective.random().rationalize()
            >>> rat_decomp = self.decompose()
            >>> print('rat_decomp = {}'.format(ub.repr2(rat_decomp, nl=1)))
            >>> ####
            >>> import sympy
            >>> cells = sympy.symbols('h1, h2, h3, h4, h5, h6, h7, h8, h9')
            >>> matrix = _RationalNDArray(cells).reshape(3, 3)
            >>> # Symbolic decomposition. Neat.
            >>> self = kwimage.Projective(matrix)
            >>> self.decompose()

        Ignore:
            >>> from sympy.abc import theta
            >>> import sympy
            >>> h1, h2, h3, h4, h5, h6, h7, h8, h9 = sympy.symbols(
            >>>     'h1, h2, h3, h4, h5, h6, h7, h8, h9')
            >>> H = sympy.Matrix([[h1, h2, h3], [h4, h5, h6], [h7, h8, 1]])
            >>> a1 = h1 - h3 * h7
            >>> a2 = h2 - h3 * h8
            >>> a3 = h3
            >>> a4 = h4 - h6 * h7
            >>> a5 = h5 - h6 * h8
            >>> a6 = h6
            >>> A = sympy.Matrix([[a1, a2, a3], [a4, a5, a6], [0, 0, 1]])
            >>> P = sympy.Matrix([[1, 0, 0], [0, 1, 0], [h7, h8, 1]])
            >>> assert np.all(np.ravel((A @ P - H).tolist()) == 0)

            # TODO: Can we get a more concise ane nice sympy decomposition /
            # recombination
            sympy.printing.pretty_print(sympy.Eq(H, B @ Q))
            bs = b1, b2, b3, b4, b5, b6, b7, b8, b9 = sympy.symbols(
            'b1, b2, b3, b4, b5, b6, b7, b8, b9')
            uvs = u, v = sympy.symbols('u, v')
            B = sympy.Matrix([[b1, b2, b3], [b4, b5, b6], [0, 0, 1]])
            Q = sympy.Matrix([[1, 0, 0], [0, 1, 0], [u, v, 1]])
            # Q = sympy.Matrix([[1, 0, 0], [0, 1, 0], [u, v, 1 / (b3*u + b6*v + 1)]])
            H2_unnorm = Q @ B
            H2_norm = H2_unnorm / H2_unnorm.tolist()[-1][-1]
            expr = sympy.Eq(H, Q @ B)
            sympy.solve(expr, bs + uvs)
            A @ P = H
            A @ P = H =
            A @ P @ A.inv() @ A
            A @ P @ A.inv()

            kwimage.Projective.projective(

        Ignore:
            import kwimage
            import kwplot
            import sympy
            plt = kwplot.autoplt()
            from kwplot.cli import gifify
            import cv2
            check = kwimage.atleast_3channels(kwimage.checkerboard(dsize=(288, 288)))
            pic = kwimage.grab_test_image('astro', dsize=check.shape[0:2][::-1])
            img1 = np.maximum((1 - check) * kwimage.ensure_float01(pic), check)
            ims = []
            # v_coords = np.log(np.logspace(1, 2)) / np.log(10) - 1
            v_coords = np.linspace(0, 1.0, 64) ** 6
            for v in ub.ProgIter(v_coords):
                u = 0
                v = v
                I = kwimage.Affine.eye()
                T = kwimage.Affine.translate(-128)
                # T = kwimage.Affine.translate(0)
                H = kwimage.Projective(np.array([[1, 0, 0], [0, 1, 0], [u, v, 1]]).astype(np.float32))
                H2 = (T.inv() @ H @ T)
                C2 = (I @ I)
                img1_warp = kwimage.warp_projective(img1, H2.matrix, dsize=img1.shape[0:2][::-1]).clip(0, 1)
                img1_pre = kwimage.warp_projective(img1, C2.matrix, dsize=img1.shape[0:2][::-1]).clip(0, 1)
                canvas = kwimage.stack_images([img1_pre, img1_warp], pad=10, axis=1, bg_value=(0., 1., 0.))
                canvas = kwimage.draw_text_on_image(
                    canvas,
                    'u={},{}v={}'.format(round(u, 8), chr(10) * 2, round(v, 8)),
                    # org=tuple(img1.shape[0:2][::-1]),
                    # valign='bottom', halign='right',
                    org=(1, 1),
                    valign='top', halign='left',
                    fontScale=0.8,
                    border=True,
                    )
                ims.append(canvas)
            # Hinge animation
            images = ims
            dpath = ub.Path.appdir('kwcoco/demo').ensuredir()
            output_fpath = dpath / 'hinge-v-t.gif'
            gifify.ffmpeg_animate_images(ims, output_fpath, in_framerate=2)
        """
        import numpy as np
        h1, h2, h3, h4, h5, h6, h7, h8, h9 = self.matrix.ravel()
        # assert h9 == 1

        a1 = h1 - h3 * h7
        a2 = h2 - h3 * h8
        a3 = h3
        a4 = h4 - h6 * h7
        a5 = h5 - h6 * h8
        a6 = h6

        mcls = _RationalNDArray if self.is_rational() else np.array

        affine_part = Affine(mcls([
            [a1, a2, a3],
            [a4, a5, a6],
            [0,   0,  1],
        ]))
        decomp = affine_part.decompose()
        # The line u * x + v * y = 0 is fixed to iteself.
        # I.e. y = -u/v * x + 0

        # The line u * x + v * y = -1 is mapped to the point at infinity
        # I.e. y = -u/v * x - 1/v
        u, v = h7, h8
        # This transform has to happen first when we re-compose
        # TODO: I would love to find a more intuitive name or representation
        # for this. Can I do something to call this a "hinge"? Is there a
        # representation where I can look at this and get a sense of where the
        # "hinge" is?
        decomp['uv'] = (u, v)
        return decomp


class Affine(Projective):
    """
    A thin wraper around a 3x3 matrix that represents an affine transform

    Implements methods for:
        * creating random affine transforms
        * decomposing the matrix
        * finding a best-fit transform between corresponding points
        * TODO: - [ ] fully rational transform

    Example:
        >>> import kwimage
        >>> import math
        >>> image = kwimage.grab_test_image()
        >>> theta = 0.123 * math.tau
        >>> components = {
        >>>     'rotate': kwimage.Affine.affine(theta=theta),
        >>>     'scale': kwimage.Affine.affine(scale=0.5),
        >>>     'shear': kwimage.Affine.affine(shearx=0.2),
        >>>     'translation': kwimage.Affine.affine(offset=(100, 200)),
        >>>     'rotate+translate': kwimage.Affine.affine(theta=0.123 * math.tau, about=(256, 256)),
        >>>     'random composed': kwimage.Affine.random(scale=(0.5, 1.5), translate=(-20, 20), theta=(-theta, theta), shearx=(0, .4), rng=900558176210808600),
        >>> }
        >>> warp_stack = []
        >>> for key, aff in components.items():
        ...     warp = kwimage.warp_affine(image, aff)
        ...     warp = kwimage.draw_text_on_image(
        ...        warp,
        ...        ub.repr2(aff.matrix, nl=1, nobr=1, precision=2, si=1, sv=1, with_dtype=0),
        ...        org=(1, 1),
        ...        valign='top', halign='left',
        ...        fontScale=0.8, color='kw_blue',
        ...        border={'thickness': 3},
        ...        )
        ...     warp = kwimage.draw_header_text(warp, key, color='kw_green')
        ...     warp_stack.append(warp)
        >>> warp_canvas = kwimage.stack_images_grid(warp_stack, chunksize=3, pad=10, bg_value='kitware_gray')
        >>> # xdoctest: +REQUIRES(module:sympy)
        >>> import sympy
        >>> # Shows the symbolic construction of the code
        >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        >>> from sympy.abc import theta
        >>> params = x0, y0, sx, sy, theta, shearx, tx, ty = sympy.symbols(
        >>>     'x0, y0, sx, sy, theta, shearx, tx, ty')
        >>> theta = sympy.symbols('theta')
        >>> # move the center to 0, 0
        >>> tr1_ = np.array([[1, 0,  -x0],
        >>>                  [0, 1,  -y0],
        >>>                  [0, 0,    1]])
        >>> # Define core components of the affine transform
        >>> S = np.array([  # scale
        >>>     [sx,  0, 0],
        >>>     [ 0, sy, 0],
        >>>     [ 0,  0, 1]])
        >>> E = np.array([  # x-shear
        >>>     [1,  shearx, 0],
        >>>     [0,  1, 0],
        >>>     [0,  0, 1]])
        >>> R = np.array([  # rotation
        >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
        >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
        >>>     [               0,                 0, 1]])
        >>> T = np.array([  # translation
        >>>     [ 1,  0, tx],
        >>>     [ 0,  1, ty],
        >>>     [ 0,  0,  1]])
        >>> # Contruct the affine 3x3 about the origin
        >>> aff0 = np.array(sympy.simplify(T @ R @ E @ S))
        >>> # move 0, 0 back to the specified origin
        >>> tr2_ = np.array([[1, 0,  x0],
        >>>                  [0, 1,  y0],
        >>>                  [0, 0,   1]])
        >>> # combine transformations
        >>> aff = tr2_ @ aff0 @ tr1_
        >>> print('aff = {}'.format(ub.repr2(aff.tolist(), nl=1)))
        >>> # This could be prettier
        >>> texts = {
        >>>     'Translation': sympy.pretty(R),
        >>>     'Rotation': sympy.pretty(R),
        >>>     'shEar-X': sympy.pretty(E),
        >>>     'Scale': sympy.pretty(S),
        >>> }
        >>> print(ub.repr2(texts, nl=2, sv=1))
        >>> equation_stack = []
        >>> for text, m in texts.items():
        >>>     render_canvas = kwimage.draw_text_on_image(None, m, color='kw_blue', fontScale=1.0)
        >>>     render_canvas = kwimage.draw_header_text(render_canvas, text, color='kw_green')
        >>>     render_canvas = kwimage.imresize(render_canvas, scale=1.3)
        >>>     equation_stack.append(render_canvas)
        >>> equation_canvas = kwimage.stack_images(equation_stack, pad=10, axis=1, bg_value='kitware_gray')
        >>> render_canvas = kwimage.draw_text_on_image(None, sympy.pretty(aff), color='kw_blue', fontScale=1.0)
        >>> render_canvas = kwimage.draw_header_text(render_canvas, 'Full Equation With Pre-Shift', color='kw_green')
        >>> # xdoctest: -REQUIRES(module:sympy)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> canvas = kwimage.stack_images([warp_canvas, equation_canvas, render_canvas], pad=20, axis=0, bg_value='kitware_gray', resize='larger')
        >>> canvas = kwimage.draw_header_text(canvas, 'Affine matrixes can represent', color='kw_green')
        >>> kwplot.imshow(canvas)
        >>> fig = plt.gcf()
        >>> fig.set_size_inches(13, 13)

    Example:
        >>> import kwimage
        >>> self = kwimage.Affine(np.eye(3))
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
        >>> m[len(m)] = x = np.eye(3) @ Affine.random()
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
        tx, ty = params['offset']
        sx, sy = params['scale']
        if math.isclose(tx, 0) and math.isclose(ty, 0):
            params.pop('offset')
        elif tx == ty:
            params['offset'] = tx
        if math.isclose(sy, 1) and math.isclose(sy, 1):
            params.pop('scale')
        elif sx == sy:
            params['scale'] = sx
        if math.isclose(params['shearx'], 0):
            params.pop('shearx')
        if math.isclose(params['theta'], 0):
            params.pop('theta')
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
                known_params = {'scale', 'offset', 'theta', 'type', 'shearx', 'shear', 'about'}
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

        Returns:
            float: large when there are big scale differences in principle
                directions or skews.

        References:
            https://en.wikipedia.org/wiki/Conic_section
            https://github.com/rasterio/affine/blob/78c20a0cfbb5ec/affine/__init__.py#L368

        Example:
            >>> import kwimage
            >>> kwimage.Affine.random(rng=432).eccentricity()
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

    def to_shapely(self):
        """
        Returns a matrix suitable for shapely.affinity.affine_transform

        Returns:
            Tuple[float, float, float, float, float, float]

        Example:
            >>> import kwimage
            >>> self = kwimage.Affine.random()
            >>> sh_transform = self.to_shapely()
            >>> # Transform points with kwimage and shapley
            >>> import shapely
            >>> from shapely.affinity import affine_transform
            >>> kw_poly = kwimage.Polygon.random()
            >>> kw_warp_poly = kw_poly.warp(self)
            >>> sh_poly = kw_poly.to_shapely()
            >>> sh_warp_poly = affine_transform(sh_poly, sh_transform)
            >>> kw_warp_poly_recon = kwimage.Polygon.from_shapely(sh_warp_poly)
            >>> assert np.allclose(kw_warp_poly_recon.exterior.data, kw_warp_poly_recon.exterior.data)
        """
        # from shapely.affinity import affine_transform
        a, b, x, d, e, y = self.matrix.ravel()[0:6]
        sh_transform = (a, b, d, e, x, y)
        return sh_transform

    def to_skimage(self):
        """
        Returns:
            skimage.transform.AffineTransform

        Example:
            >>> import kwimage
            >>> self = kwimage.Affine.random()
            >>> tf = self.to_skimage()
            >>> # Transform points with kwimage and scikit-image
            >>> kw_poly = kwimage.Polygon.random()
            >>> kw_warp_xy = kw_poly.warp(self.matrix).exterior.data
            >>> sk_warp_xy = tf(kw_poly.exterior.data)
            >>> assert np.allclose(sk_warp_xy, sk_warp_xy)
        """
        return skimage.transform.AffineTransform(matrix=np.asarray(self))

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
    def random(cls, shape=None, rng=None, **kw):
        """
        Create a random Affine object

        Args:
            rng : random number generator
            **kw: passed to :func:`Affine.random_params`.
                can contain coercable random distributions for scale, offset,
                about, theta, and shearx.

        Returns:
            Affine
        """
        if shape is not None:
            raise ValueError('cannot specify shape to Affine.random')
        params = cls.random_params(rng=rng, **kw)
        self = cls.affine(**params)
        return self

    @classmethod
    def random_params(cls, rng=None, **kw):
        """
        Args:
            rng : random number generator
            **kw: can contain coercable random distributions for
                scale, offset, about, theta, and shearx.

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
            if ub.iterable(kw['scale']) and (not isinstance(kw['scale'], tuple) and len(kw['scale']) == 2):
                raise NotImplementedError
            else:
                print(kw['scale'])
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

        if 'shearx' in kw:
            shear_dist = _coerce_distri(kw['shearx'])
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
            shearx=shear_dist.sample(),
            about=(xabout_dist.sample(), yabout_dist.sample()),
        )
        return params

    @profile
    def decompose(self):
        r"""
        Decompose the affine matrix into its individual scale, translation,
        rotation, and skew parameters.

        Returns:
            Dict: decomposed offset, scale, theta, and shearx params

        References:
            https://math.stackexchange.com/questions/612006/decompose-affine
            https://math.stackexchange.com/a/3521141/353527
            https://stackoverflow.com/questions/70357473/how-to-decompose-a-2x2-affine-matrix-with-sympy
            https://en.wikipedia.org/wiki/Transformation_matrix
            https://en.wikipedia.org/wiki/Shear_mapping

        Example:
            >>> from kwimage.transform import *  # NOQA
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

        Example:
            >>> # xdoctest: +REQUIRES(module:sympy)
            >>> # Test decompose with symbolic matrices
            >>> from kwimage.transform import *  # NOQA
            >>> self = Affine.random().rationalize()
            >>> self.decompose()

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> from kwimage.transform import *  # NOQA
            >>> import kwimage
            >>> import pandas as pd
            >>> # Test consistency of decompose + reconstruct
            >>> param_grid = list(ub.named_product({
            >>>     'theta': np.linspace(-4 * np.pi, 4 * np.pi, 3),
            >>>     'shearx': np.linspace(- 10 * np.pi, 10 * np.pi, 4),
            >>> }))
            >>> def normalize_angle(radian):
            >>>     return np.arctan2(np.sin(radian), np.cos(radian))
            >>> for pextra in param_grid:
            >>>     params0 = dict(scale=(3.05, 3.07), offset=(10.5, 12.1), **pextra)
            >>>     self = recon0 = kwimage.Affine.affine(**params0)
            >>>     self.decompose()
            >>>     # Test drift with multiple decompose / reconstructions
            >>>     params_list = [params0]
            >>>     recon_list = [recon0]
            >>>     n = 4
            >>>     for _ in range(n):
            >>>         prev = recon_list[-1]
            >>>         params = prev.decompose()
            >>>         recon = kwimage.Affine.coerce(**params)
            >>>         params_list.append(params)
            >>>         recon_list.append(recon)
            >>>     params_df = pd.DataFrame(params_list)
            >>>     #print('params_list = {}'.format(ub.repr2(params_list, nl=1, precision=5)))
            >>>     print(params_df)
            >>>     assert ub.allsame(normalize_angle(params_df['theta']), eq=np.isclose)
            >>>     assert ub.allsame(params_df['shearx'], eq=np.allclose)
            >>>     assert ub.allsame(params_df['scale'], eq=np.allclose)
            >>>     assert ub.allsame(params_df['offset'], eq=np.allclose)

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
        if self.matrix is None:
            return {'offset': (0., 0.), 'scale': (1., 1.), 'shearx': 0.,
                    'theta': 0., }
        a11, a12, a13, a21, a22, a23 = self.matrix.ravel()[0:6]

        if self.is_rational():
            math_mod = sympy
        else:
            math_mod = math

        sx = math_mod.sqrt(a11 * a11 + a21 * a21)
        theta = math_mod.atan2(a21, a11)
        sin_t = math_mod.sin(theta)
        cos_t = math_mod.cos(theta)

        msy = a12 * cos_t + a22 * sin_t

        try:
            if abs(cos_t) < abs(sin_t):
                sy = (msy * cos_t - a12) / sin_t
            else:
                sy = (a22 - msy * sin_t) / cos_t
        except TypeError:
            # symbolic issue
            sy = sympy.Piecewise(
                ((msy * cos_t - a12) / sin_t, abs(cos_t) < abs(sin_t)),
                ((a22 - msy * sin_t) / cos_t, True)
            )

        shearx = msy / sy
        tx, ty = a13, a23

        params = {
            'offset': (tx, ty),
            'scale': (sx, sy),
            'shearx': shearx,
            'theta': theta,
        }
        return params

    @classmethod
    @profile
    def affine(cls, scale=None, offset=None, theta=None, shear=None,
               about=None, shearx=None, array_cls=None, math_mod=None,
               **kwargs):
        """
        Create an affine matrix from high-level parameters

        Args:
            scale (float | Tuple[float, float]):
                x, y scale factor

            offset (float | Tuple[float, float]):
                x, y translation factor

            theta (float):
                counter-clockwise rotation angle in radians

            shearx (float):
                shear factor parallel to the x-axis.

            about (float | Tuple[float, float]):
                x, y location of the origin

            shear (float):
                BROKEN, dont use.  counter-clockwise shear angle in radians

        TODO:
            - [ ] Add aliases? -
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
            >>> shearx = rng.randn() * 10
            >>> # Create combined matrix from all params
            >>> F = Affine.affine(
            >>>     scale=scale, offset=offset, theta=theta, shearx=shearx,
            >>>     about=about)
            >>> # Test that combining components matches
            >>> S = Affine.affine(scale=scale)
            >>> T = Affine.affine(offset=offset)
            >>> R = Affine.affine(theta=theta)
            >>> E = Affine.affine(shearx=shearx)
            >>> O = Affine.affine(offset=about)
            >>> # combine (note shear must be on the RHS of rotation)
            >>> alt  = O @ T @ R @ E @ S @ O.inv()
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
            >>> params = x0, y0, sx, sy, theta, shearx, tx, ty = sympy.symbols(
            >>>     'x0, y0, sx, sy, theta, shearx, tx, ty')
            >>> # move the center to 0, 0
            >>> tr1_ = np.array([[1, 0,  -x0],
            >>>                  [0, 1,  -y0],
            >>>                  [0, 0,    1]])
            >>> # Define core components of the affine transform
            >>> S = np.array([  # scale
            >>>     [sx,  0, 0],
            >>>     [ 0, sy, 0],
            >>>     [ 0,  0, 1]])
            >>> E = np.array([  # x-shear
            >>>     [1,  shearx, 0],
            >>>     [0,  1, 0],
            >>>     [0,  0, 1]])
            >>> R = np.array([  # rotation
            >>>     [sympy.cos(theta), -sympy.sin(theta), 0],
            >>>     [sympy.sin(theta),  sympy.cos(theta), 0],
            >>>     [               0,                 0, 1]])
            >>> T = np.array([  # translation
            >>>     [ 1,  0, tx],
            >>>     [ 0,  1, ty],
            >>>     [ 0,  0,  1]])
            >>> # Contruct the affine 3x3 about the origin
            >>> aff0 = np.array(sympy.simplify(T @ R @ E @ S))
            >>> # move 0, 0 back to the specified origin
            >>> tr2_ = np.array([[1, 0,  x0],
            >>>                  [0, 1,  y0],
            >>>                  [0, 0,   1]])
            >>> # combine transformations
            >>> aff = tr2_ @ aff0 @ tr1_
            >>> print('aff = {}'.format(ub.repr2(aff.tolist(), nl=1)))

        Ignore:
            import timerit
            ti = timerit.Timerit(10000, bestof=10, verbose=2)
            for timer in ti.reset('time'):
                with timer:
                    self = kwimage.Affine.affine(scale=3, offset=2, theta=np.random.rand(), shearx=np.random.rand())
        """
        if shear is not None and shearx is None:
            # Hack so old data is readable (this should be ok as long as the
            # data wasnt reserialized)
            if not _internal.KWIMAGE_DISABLE_TRANSFORM_WARNINGS:
                ub.schedule_deprecation(
                    modname='kwimage', name='shear', type='parameter',
                    migration=ub.paragraph(
                        '''
                        The `shear` parameter is deprecated and will be removed because
                        of a serious bug. Use `shearx` instead. See Issue #8 on
                        https://gitlab.kitware.com/computer-vision/kwimage/-/issues/8
                        for more details. To ease the impact of this bug we will
                        interpret `shear` as `shearx`, which should result in a correct
                        reconstruction, as long as the data was never reserialized.
                        '''
                    ), deprecate='0.9.0', error='0.10.0', remove='0.11.0', warncls=UserWarning)
            shearx = shear
            shear = None

        if array_cls is None:
            array_cls = np.array

        if math_mod is None:
            math_mod = math

        scale_ = 1 if scale is None else scale
        offset_ = 0 if offset is None else offset
        xshear_ = 0 if shearx is None else shearx
        theta_ = 0 if theta is None else theta
        about_ = 0 if about is None else about
        sx, sy = _ensure_iterable2(scale_)
        tx, ty = _ensure_iterable2(offset_)
        x0, y0 = _ensure_iterable2(about_)

        cos_theta = math_mod.cos(theta_)
        sin_theta = math_mod.sin(theta_)

        sx_cos_theta = sx * cos_theta
        sx_sin_theta = sx * sin_theta

        sy_cos_theta = sy * cos_theta
        sy_sin_theta = sy * sin_theta

        m_sy_cos_theta = xshear_ * sy_cos_theta
        m_sy_sin_theta = xshear_ * sy_sin_theta

        a12 = m_sy_cos_theta - sy_sin_theta
        a22 = m_sy_sin_theta + sy_cos_theta

        tx_ = tx + x0 - (x0 * sx_cos_theta) - (y0 * a12)
        ty_ = ty + y0 - (x0 * sx_sin_theta) - (y0 * a22)

        mat = array_cls([sx_cos_theta, a12, tx_,
                         sx_sin_theta, a22, ty_,
                                    0,   0,  1])
        mat = mat.reshape(3, 3)  # Faster to make a flat array and reshape
        self = cls(mat)
        return self

    @classmethod
    def fit(cls, pts1, pts2):
        """
        Fit an affine transformation between a set of corresponding points

        Args:
            pts1 (ndarray): An Nx2 array of points in "space 1".
            pts2 (ndarray): A corresponding Nx2 array of points in "space 2"

        Returns:
            Affine : a transform that warps from "space1" to "space2".

        Note:
            An affine matrix has 6 degrees of freedom, so at least 3
            non-colinear xy-point pairs are needed.

        References:
            https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf page 22

        Example:
            >>> # Create a set of points, warp them, then recover the warp
            >>> import kwimage
            >>> points = kwimage.Points.random(6).scale(64)
            >>> #A1 = kwimage.Affine.affine(scale=0.9, theta=-3.2, offset=(2, 3), about=(32, 32), skew=2.3)
            >>> #A2 = kwimage.Affine.affine(scale=0.8, theta=0.8, offset=(2, 0), about=(32, 32))
            >>> A1 = kwimage.Affine.random()
            >>> A2 = kwimage.Affine.random()
            >>> A12_real = A2 @ A1.inv()
            >>> points1 = points.warp(A1)
            >>> points2 = points.warp(A2)
            >>> # Recover the warp
            >>> pts1, pts2 = points1.xy, points2.xy
            >>> A_recovered = kwimage.Affine.fit(pts1, pts2)
            >>> assert np.all(np.isclose(A_recovered.matrix, A12_real.matrix))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> base1 = np.zeros((96, 96, 3))
            >>> base1[32:-32, 5:-5] = 0.5
            >>> base2 = np.zeros((96, 96, 3))
            >>> img1 = points1.draw_on(base1, radius=3, color='blue')
            >>> img2 = points2.draw_on(base2, radius=3, color='green')
            >>> img1_warp = kwimage.warp_affine(img1, A_recovered)
            >>> canvas = kwimage.stack_images([img1, img2, img1_warp], pad=10, axis=1, bg_value=(1., 1., 1.))
            >>> kwplot.imshow(canvas)
        """
        if 0:
            # Not sure if cv2 has this variant of the affine matrix calc
            import cv2
            inlier_method = 'ransac'
            inlier_method_lut = {
                'lmeds': cv2.LMEDS,
                'ransac': cv2.RANSAC,
            }
            cv2_method = inlier_method_lut[inlier_method]
            A, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2_method)

        x1_mn = pts1[:, 0]
        y1_mn = pts1[:, 1]
        x2_mn = pts2[:, 0]
        y2_mn = pts2[:, 1]
        num_pts = x1_mn.shape[0]
        Mx6 = np.empty((2 * num_pts, 6), dtype=float)
        b = np.empty((2 * num_pts, 1), dtype=float)
        for ix in range(num_pts):  # Loop over inliers
            # Concatenate all 2x9 matrices into an Mx6 matrix
            x1 = x1_mn[ix]
            x2 = x2_mn[ix]
            y1 = y1_mn[ix]
            y2 = y2_mn[ix]
            Mx6[ix * 2]     = (x1, y1, 0, 0, 1, 0)
            Mx6[ix * 2 + 1] = ( 0, 0, x1, y1, 0, 1)
            b[ix * 2] = x2
            b[ix * 2 + 1] = y2

        M = Mx6
        try:
            USVt = np.linalg.svd(M, full_matrices=True, compute_uv=True)
        except MemoryError:
            import scipy.sparse as sps
            import scipy.sparse.linalg as spsl
            M_sparse = sps.lil_matrix(M)
            USVt = spsl.svds(M_sparse)
        except np.linalg.LinAlgError:
            raise
        except Exception:
            raise

        U, s, Vt = USVt

        # Inefficient, but the math works
        # We want to solve Ax=b (where A is the Mx6 in this case)
        # Ax = b
        # (U S V.T) x = b
        # x = (U.T inv(S) V) b
        Sinv = np.zeros((len(Vt), len(U)))
        Sinv[np.diag_indices(len(s))] = 1 / s
        a = Vt.T.dot(Sinv).dot(U.T).dot(b).T[0]
        mat = np.array([
            [a[0], a[1], a[4]],
            [a[2], a[3], a[5]],
            [   0, 0, 1],
        ])
        return Affine(mat)


try:
    import sympy
    _RationalMatrixBase = sympy.Matrix
except Exception:
    sympy = None
    _RationalMatrixBase = object


class _RationalNDArray(_RationalMatrixBase):
    """
    Wraps sympy matrices to make it somewhat more compatible with numpy.

    Example:
        >>> # xdoctest: +REQUIRES(module:sympy)
        >>> from kwimage.transform import *  # NOQA
        >>> from kwimage.transform import _RationalNDArray
        >>> arr = np.random.rand(3, 3)
        >>> a = _RationalNDArray.from_numpy(arr)
        >>> b = np.random.rand(3, 3)
        >>> c = a @ b
        >>> c @ c.inv()
    """

    @classmethod
    def from_numpy(_RationalNDArray, arr):
        flat_rat = list(map(sympy.Rational, arr.ravel().tolist()))
        self = _RationalNDArray(flat_rat).reshape(*arr.shape)
        return self

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            other = _RationalNDArray.from_numpy(other)
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            other = _RationalNDArray.from_numpy(other)
        return super().__matmul__(other)

    def numpy(self):
        return np.array(self.tolist()).astype(float)

    def ravel(self):
        return self.flat()

# Does not seem to be working out
# if 0:
#     try:
#         from mpmath import matrix as mp_matrix_base
#     except ImportError:
#         mp_matrix_base = object
#     class _mpmatrix(mp_matrix_base):
#         """
#         A compatability layer for mpmath matrix

#         Example:
#             >>> # xdoctest: +REQUIRES(module:mpmath)
#             >>> from kwimage.transform import _mpmatrix  # NOQA
#             >>> A = _mpmatrix(np.random.rand(3, 3))
#             >>> B = _mpmatrix(np.random.rand(3, 3))
#             >>> C = np.random.rand(3, 3)
#             >>> A @ B
#             >>> B @ A
#             >>> self = A
#             >>> other = C
#             >>> A.__matmul__(C)
#             >>> # C.__matmul__(A) not sure why this fails
#         """

#         @property
#         def shape(self):
#             return (self.rows, self.cols)

#         def __matmul__(self, other):
#             if isinstance(other, np.ndarray):
#                 other = _mpmatrix(other)
#             return _mpmatrix(mp_matrix_base.__matmul__(self, other))

#         def __rmatmul__(self, other):
#             if isinstance(other, np.ndarray):
#                 other = _mpmatrix(other)
#             return _mpmatrix(mp_matrix_base.__matmul__(other, self))

#         def numpy(self):
#             return np.array(self).reshape(*self.shape)


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
