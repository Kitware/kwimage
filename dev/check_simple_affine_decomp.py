#########
# Simplest case

"""
This is now working :)

I'm attempting to show the decomposition of an affine matrix with sympy as
shown in the following stackexchange post:

https://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation


I've setup two matrices `A_params` and `A_matrix`, where the former represents
the raw matrix values and the latter is the matrix constructed from its
underlying parameters.
"""
import sympy
import itertools as it
import ubelt as ub
import numpy as np
# domain = {'real': True, 'negative': False}
# domain = {}
domain = {'real': True}

theta = sympy.symbols('theta', **domain)
sx, sy = sympy.symbols('sx, sy', **domain)
m = sympy.symbols('m', **domain)
params = [sx, theta, sy, m]

S = sympy.Matrix([  # scale
    [sx,  0],
    [ 0, sy]])

H = sympy.Matrix([  # shear
    [1, m],
    [0, 1]])

R = sympy.Matrix((  # rotation
    [sympy.cos(theta), -sympy.sin(theta)],
    [sympy.sin(theta),  sympy.cos(theta)]))
# R = sympy.matrices.dense.rot_axis3(theta)[0:2, 0:2]


A_params = sympy.simplify((R @ H @ S))
a11, a12, a21, a22 = sympy.symbols(
    'a11, a12, a21, a22', real=True)
A_matrix = sympy.Matrix(((a11, a12), (a21, a22)))
elements = list(it.chain.from_iterable(A_matrix.tolist()))


print(ub.hzcat(['A_matrix = ', sympy.pretty(A_matrix)]))
print(ub.hzcat(['A_params = ', sympy.pretty(A_params)]))


# A_matrix.singular_values()
# A_params.singular_values()
# A_matrix.singular_value_decomposition()


"""
A_matrix = ⎡a₁₁  a₁₂⎤
           ⎢        ⎥
           ⎣a₂₁  a₂₂⎦
A_params = ⎡sx⋅cos(θ)  sy⋅(m⋅cos(θ) - sin(θ))⎤
           ⎢                                 ⎥
           ⎣sx⋅sin(θ)  sy⋅(m⋅sin(θ) + cos(θ))⎦
"""


if 0:
    # The skimage definition of shear is different than this "m" definition
    # scikit-image code is:
    # self.params = np.array([
    #     [sx * math.cos(rotation), -sy * math.sin(rotation + shear), 0],
    #     [sx * math.sin(rotation),  sy * math.cos(rotation + shear), 0],
    #     [                      0,                                0, 1]
    # ])
    shear = sympy.symbols('shear', **domain)
    shear_equations = [
        sympy.Eq(A_params[0, 1], -sy * sympy.sin(theta + shear)),
        sympy.Eq(A_params[1, 1],  sy * sympy.cos(theta + shear))
    ]
    sympy.solve(shear_equations, shear)
    sympy.solvers.solveset(shear_equations, shear)
    """
    sympy.solve(sympy.Eq(A_params[0, 1], -sy * sympy.sin(theta + shear)), shear)
    sympy.solve(sympy.Eq(A_params[1, 1],  sy * sympy.cos(theta + shear)), shear)

    [(-theta - asin(m*cos(theta) - sin(theta)),),
     (-theta + asin(m*cos(theta) - sin(theta)) + pi,)]

    """

"""
From what I understand I should simply be able to set these two matrices to be
equal and then solve for the parameters of interest. However, I'm getting
unexpected results.

First, if I just try to solve for "sx", I get no result.
"""

mat_equation = sympy.Eq(A_matrix, A_params)
if 0:
    ## Option 1: Matrix equality
    soln_sx = sympy.solve(mat_equation, sx, exclude=[theta, sy, m], manual=True, dict=True)
    print('soln_sx = {!r}'.format(soln_sx))

    ## Option 2: List of equations
    lhs_iter = it.chain.from_iterable(A_matrix.tolist())
    rhs_iter = it.chain.from_iterable(A_params.tolist())
    equations = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(lhs_iter, rhs_iter)]
    soln_sx = sympy.solve(equations, sx, theta, manual=True)
    print('soln_sx = {!r}'.format(soln_sx))

    if 0:
        sympy.solveset(mat_equation, sx, sympy.Reals)
        sympy.nonlinsolve(equations, sx)
        sympy.solve(equations, (m, sx, sy, theta), particular=True, manual=True, quintics=False)

    """
    soln_sx = []
    soln_sx = []
    """


"""
But if I try to solve for all variables simultaniously, I get a result although
it does not match what I would expect. However, it does appear to work.
"""

solve_for = (sx, theta, sy, m)
solutions = sympy.solve(mat_equation, *solve_for, dict=True)
solved = {}
# minimal=True, quick=True, cubics=False, quartics=False, quintics=False, check=False)
for sol in solutions:
    for sym, symsol0 in sol.items():
        symsol = sympy.radsimp(symsol0)
        symsol = sympy.trigsimp(symsol)
        symsol = sympy.simplify(symsol)
        symsol = sympy.radsimp(symsol)
        print('\n=====')
        print('sym = {!r}'.format(sym))
        print('symsol  = {!r}'.format(symsol))
        print('--')
        sympy.pretty_print(symsol, wrap_line=False)
        solved[sym] = symsol
        print('--')
        print('=====\n')

    A_matrix[0, :].dot(A_matrix[1, :]) / A_matrix.det()

S = sympy.Matrix([  # scale
    [sx,  0],
    [ 0, sy]])

H = sympy.Matrix([  # shear
    [1, m],
    [0, 1]])

R = sympy.Matrix((  # rotation
    [sympy.cos(theta), -sympy.sin(theta)],
    [sympy.sin(theta),  sympy.cos(theta)]))

A_solved_recon = sympy.simplify(A_params.subs(solved))

print(ub.hzcat(['A_solved_recon = ', sympy.pretty(A_solved_recon)]))


"""
=====
sym = sx
symsol  = -sqrt(a11**2 + a21**2)
--
    _____________
   ╱    2      2
-╲╱  a₁₁  + a₂₁
--
=====


=====
sym = theta
symsol  = -2*atan((a11 + sqrt(a11**2 + a21**2))/a21)
--
       ⎛         _____________⎞
       ⎜        ╱    2      2 ⎟
       ⎜a₁₁ + ╲╱  a₁₁  + a₂₁  ⎟
-2⋅atan⎜──────────────────────⎟
       ⎝         a₂₁          ⎠
--
=====


=====
sym = m
symsol  = (a11*a12 + a21*a22)/(a11*a22 - a12*a21)
--
a₁₁⋅a₁₂ + a₂₁⋅a₂₂
─────────────────
a₁₁⋅a₂₂ - a₁₂⋅a₂₁
--
=====


=====
sym = sy
symsol  = (-a11*a22*sqrt(a11**2 + a21**2) + a12*a21*sqrt(a11**2 + a21**2))/(a11**2 + a21**2)
--
             _____________              _____________
            ╱    2      2              ╱    2      2
- a₁₁⋅a₂₂⋅╲╱  a₁₁  + a₂₁   + a₁₂⋅a₂₁⋅╲╱  a₁₁  + a₂₁
─────────────────────────────────────────────────────
                        2      2
                     a₁₁  + a₂₁
--
=====

A_solved_recon = ⎡a₁₁  a₁₂⎤
                 ⎢        ⎥
                 ⎣a₂₁  a₂₂⎦

"""

"""
The above solution does seem correct in that it reconstructs correctly.

After having a hard time getting the above to work, I wanted to see if I could
at least verify the solution from stackexchange. So I coded that up
symbolically:
"""

# This is the guided solution by Stéphane Laurent
recon_sx = sympy.sqrt(a11 * a11 + a21 * a21)
recon_theta = sympy.atan2(a21, a11)
recon_sin_t = sympy.sin(recon_theta)
recon_cos_t = sympy.cos(recon_theta)

recon_msy = a12 * recon_cos_t + a22 * recon_sin_t

if 0:
    recon_m = (a11 * a21 + a12 * a22) / (a11 * a22 - a12 * a21)
    recon_sy = sympy.simplify(recon_msy / recon_m)
else:
    condition2 = sympy.simplify(sympy.Eq(recon_sin_t, 0))
    condition1 = sympy.simplify(sympy.Not(condition2))
    # condition1 = sympy.Gt(recon_sin_t ** 2, recon_cos_t ** 2)
    # condition2 = sympy.Le(recon_sin_t ** 2, recon_cos_t ** 2)
    sy_cond1 = (recon_msy * recon_cos_t - a12) / recon_sin_t
    sy_cond2 = (a22 - recon_msy * recon_sin_t) / recon_cos_t
    recon_sy = sympy.Piecewise((sy_cond1, condition1), (sy_cond2, condition2))
    recon_m = sympy.simplify(recon_msy / recon_sy)

recon_symbols = {sx: recon_sx, theta: recon_theta, m: recon_m, sy: recon_sy}

for sym, symval in recon_symbols.items():
    # symval = sympy.radsimp(symval)
    symval = sympy.trigsimp(symval)
    symval = sympy.simplify(symval)
    if not isinstance(symval, sympy.Piecewise):
        symval = sympy.radsimp(symval)
    print('\n=====')
    print('sym = {!r}'.format(sym))
    print('symval  = {!r}'.format(symval))
    print('--')
    sympy.pretty_print(symval)
    print('=====\n')

A_recon = A_params.subs(recon_symbols)
# A_recon[1, 1]
# print(ub.hzcat(['A_recon = ', sympy.pretty(A_recon)]))
A_recon = sympy.simplify(A_recon)
print(ub.hzcat(['A_recon = ', sympy.pretty(A_recon)]))


"""
This gives the desired result. Thank you @Michael Albright

=====
sym = sx
symval  = sqrt(a11**2 + a21**2)
--
   _____________
  ╱    2      2
╲╱  a₁₁  + a₂₁
=====


=====
sym = theta
symval  = atan2(a21, a11)
--
atan2(a₂₁, a₁₁)
=====


=====
sym = m
symval  = (a11*a12 + a21*a22)/(a11*a22 - a12*a21)
--
a₁₁⋅a₁₂ + a₂₁⋅a₂₂
─────────────────
a₁₁⋅a₂₂ - a₁₂⋅a₂₁
=====


=====
sym = sy
symval  = Piecewise((sqrt(a11**2 + a21**2)*(a11*(a11*a12/sqrt(a11**2 + a21**2) + a21*a22/sqrt(a11**2 + a21**2))/sqrt(a11**2 + a21**2) - a12)/a21, Ne(a21/sqrt(a11**2 + a21**2), 0)), (sqrt(a11**2 + a21**2)*(-a21*(a11*a12/sqrt(a11**2 + a21**2) + a21*a22/sqrt(a11**2 + a21**2))/sqrt(a11**2 + a21**2) + a22)/a11, True))
--
⎧                  ⎛    ⎛    a₁₁⋅a₁₂            a₂₁⋅a₂₂     ⎞      ⎞
⎪                  ⎜a₁₁⋅⎜──────────────── + ────────────────⎟      ⎟
⎪                  ⎜    ⎜   _____________      _____________⎟      ⎟
⎪    _____________ ⎜    ⎜  ╱    2      2      ╱    2      2 ⎟      ⎟
⎪   ╱    2      2  ⎜    ⎝╲╱  a₁₁  + a₂₁     ╲╱  a₁₁  + a₂₁  ⎠      ⎟
⎪ ╲╱  a₁₁  + a₂₁  ⋅⎜───────────────────────────────────────── - a₁₂⎟
⎪                  ⎜                _____________                  ⎟
⎪                  ⎜               ╱    2      2                   ⎟
⎪                  ⎝             ╲╱  a₁₁  + a₂₁                    ⎠             a₂₁
⎪ ──────────────────────────────────────────────────────────────────   for ──────────────── ≠ 0
⎪                                a₂₁                                          _____________
⎪                                                                            ╱    2      2
⎨                                                                          ╲╱  a₁₁  + a₂₁
⎪
⎪                 ⎛      ⎛    a₁₁⋅a₁₂            a₂₁⋅a₂₂     ⎞      ⎞
⎪                 ⎜  a₂₁⋅⎜──────────────── + ────────────────⎟      ⎟
⎪                 ⎜      ⎜   _____________      _____________⎟      ⎟
⎪   _____________ ⎜      ⎜  ╱    2      2      ╱    2      2 ⎟      ⎟
⎪  ╱    2      2  ⎜      ⎝╲╱  a₁₁  + a₂₁     ╲╱  a₁₁  + a₂₁  ⎠      ⎟
⎪╲╱  a₁₁  + a₂₁  ⋅⎜- ───────────────────────────────────────── + a₂₂⎟
⎪                 ⎜                  _____________                  ⎟
⎪                 ⎜                 ╱    2      2                   ⎟
⎪                 ⎝               ╲╱  a₁₁  + a₂₁                    ⎠
⎪────────────────────────────────────────────────────────────────────         otherwise
⎩                                a₁₁
=====



A_recon = ⎡a₁₁  a₁₂⎤
          ⎢        ⎥
          ⎣a₂₁  a₂₂⎦
"""


#### Check numerically

params = [sx, theta, sy, m]
params_rand = {p: np.random.rand() for p in params}
A_params_rand = A_params.subs(params_rand)
matrix_rand = {lhs: rhs for lhs, rhs in zip(elements, ub.flatten(A_params_rand.tolist()))}
A_matrix_rand = A_matrix.subs(matrix_rand)
A_solved_rand = A_solved_recon.subs(matrix_rand)
A_recon_rand = A_recon.subs(matrix_rand)

mat1 = np.array(A_matrix_rand.tolist()).astype(float)
mat2 = np.array(A_params_rand.tolist()).astype(float)
mat3 = np.array(A_recon_rand.tolist()).astype(float)
assert np.all(np.isclose(mat1, mat2))

print(mat2 - mat3)

mat4 = np.array(A_solved_rand.tolist()).astype(float)

# # print('A_params_rand = {!r}'.format(A_params_rand))
# print('A_matrix_rand = {!r}'.format(A_matrix_rand))
# print('A_recon_rand = {!r}'.format(A_recon_rand))


# print('A_solved_rand = {!r}'.format(A_solved_rand))
