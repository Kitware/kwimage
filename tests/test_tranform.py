def test_rational_affine():
    from kwimage import Affine
    import numbers
    import ubelt as ub
    import numpy as np
    import kwarray
    import pytest

    try:
        import sympy
    except ImportError:
        pytest.skip()

    rng = kwarray.ensure_rng(None)
    offset = [0, 0]
    about = [0, 0]
    theta = 0
    shearx = 0
    scale = rng.randn(2) * 10
    offset = rng.randn(2) * 10
    about = rng.randn(2) * 10
    theta = rng.randn() * 10
    shearx = rng.randn() * 10

    def as_rational(data):
        if isinstance(data, list):
            rat = list(map(as_rational, data))
        elif isinstance(data, numbers.Number):
            rat = sympy.Rational(data)
        elif isinstance(data, np.ndarray):
            flat_rat = list(map(sympy.Rational, data.ravel().tolist()))
            rat = sympy.Array(flat_rat).reshape(*data.shape)
        elif isinstance(data, sympy.Rational):
            rat = data
        elif isinstance(data, sympy.Array):
            # TODO: check rationality
            rat = data
        else:
            raise TypeError(type(data))
        return rat

    scale = as_rational(scale)
    offset = as_rational(offset)
    about = as_rational(about)
    theta = as_rational(theta)
    shearx = as_rational(shearx)

    math_mod = sympy
    array_cls = sympy.Matrix

    # Create combined matrix from all params
    A = Affine.affine(
        scale=scale, offset=offset, theta=theta, shearx=shearx,
        about=about, array_cls=array_cls, math_mod=math_mod)
    # Test that combining components matches
    S = Affine.affine(scale=scale, array_cls=array_cls, math_mod=math_mod)
    T = Affine.affine(offset=offset, array_cls=array_cls, math_mod=math_mod)
    R = Affine.affine(theta=theta, array_cls=array_cls, math_mod=math_mod)
    H = Affine.affine(shearx=shearx, array_cls=array_cls, math_mod=math_mod)
    F = Affine.affine(offset=about, array_cls=array_cls, math_mod=math_mod)
    # combine (note shear must be on the RHS of rotation)
    alt  = F @ T @ R @ H @ S @ F.inv()

    V2 = np.asarray(alt.matrix).astype(float)
    V1 = np.asarray(A.matrix).astype(float)
    print(f'V2={V2}')
    print(f'V1={V1}')

    print('A    = {}'.format(ub.repr2(A.matrix.tolist(), nl=1)))
    print('alt  = {}'.format(ub.repr2(alt.matrix.tolist(), nl=1)))
    assert np.all(V1 == V2)
    pt = np.vstack([np.random.rand(2, 1), [[1]]])
    warp_pt1 = (A.matrix @ pt).evalf()
    warp_pt2 = (alt.matrix @ pt).evalf()
    assert warp_pt2 == warp_pt1
