#########
# Simplest case

"""
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
domain = {'real': True}

theta = sympy.symbols('theta', **domain)
sx, sy = sympy.symbols('sx, sy', **domain)
m = sympy.symbols('m', **domain)

S = sympy.Matrix([  # scale
    [sx,  0],
    [ 0, sy]])

H = sympy.Matrix([  # shear
    [1, m],
    [0, 1]])

R = sympy.Matrix([  # rotation
    [sympy.cos(theta), -sympy.sin(theta)],
    [sympy.sin(theta),  sympy.cos(theta)]])


A_params = sympy.simplify((R @ H @ S))
a11, a12, a21, a22 = sympy.symbols(
    'a11, a12, a21, a22', real=True)
A_matrix = sympy.Matrix([[a11, a12], [a21, a22]])


print(ub.hzcat(['A_matrix = ', sympy.pretty(A_matrix)]))
print(ub.hzcat(['A_params = ', sympy.pretty(A_params)]))


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

## Option 1: Matrix equality
mat_equation = sympy.Eq(A_matrix, A_params)
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
But if I try to solve for all variables simultaniously, I get a result
but it does not agree with what I would expect
"""

solve_for = (sx, theta, sy, m)
solutions = sympy.solve(mat_equation, *solve_for, dict=True, minimal=True, quick=True, cubics=False, quartics=False, quintics=False, check=False)
for sol in solutions:
    for sym, symsol0 in sol.items():
        symsol = sympy.radsimp(symsol0)
        symsol = sympy.trigsimp(symsol)
        symsol = sympy.simplify(symsol)
        print('\n=====')
        print('sym = {!r}'.format(sym))
        print('symsol0 = {!r}'.format(symsol0))
        print('symsol  = {!r}'.format(symsol))
        print('--')
        sympy.pretty_print(symsol, wrap_line=False)
        print('--')
        print('=====\n')

"""
sol(sx) = -(a11**2 + a11*sqrt(a11**2 + a21**2) + a21**2)/(a11 + sqrt(a11**2 + a21**2))
sol(theta) = -2*atan((a11 + sqrt(a11**2 + a21**2))/a21)
sol(sy) = (-8*a11**6*a22 + 8*a11**5*a12*a21 - 8*a11**5*a22*sqrt(a11**2 + a21**2) + 8*a11**4*a12*a21*sqrt(a11**2 + a21**2) - 12*a11**4*a21**2*a22 + 12*a11**3*a12*a21**3 - 8*a11**3*a21**2*a22*sqrt(a11**2 + a21**2) + 8*a11**2*a12*a21**3*sqrt(a11**2 + a21**2) - 4*a11**2*a21**4*a22 + 4*a11*a12*a21**5 - a11*a21**4*a22*sqrt(a11**2 + a21**2) + a12*a21**5*sqrt(a11**2 + a21**2))/(8*a11**6 + 8*a11**5*sqrt(a11**2 + a21**2) + 16*a11**4*a21**2 + 12*a11**3*a21**2*sqrt(a11**2 + a21**2) + 9*a11**2*a21**4 + 4*a11*a21**4*sqrt(a11**2 + a21**2) + a21**6)
sol(m) = (a11*a12 + a21*a22)/(a11*a22 - a12*a21)



expr1 = -2*sympy.atan((a11 + sympy.sqrt(a11**2 + a21**2))/a21)
expr2 = sympy.atan2(a21, a11)

subs = {s: np.random.rand() for s in expr1.free_symbols}
expr1.subs(subs)
expr2.subs(subs)



a11, a12, a22, a21 = np.random.rand(4)
atan2 = np.arctan2
atan = np.arctan
sqrt = np.sqrt
theta_alt1 = -2*atan((a11 + sqrt(a11**2 + a21**2))/a21)
theta_alt2 = -2*atan((a11 + sqrt(a11**2 + a21**2)) / a21)
print('theta_alt1 = {!r}'.format(theta_alt1))
print('theta_alt2 = {!r}'.format(theta_alt2))
theta_main = atan2(a21, a11)
"""

"""
After having a hard time getting the above to work, I wanted to see
if I could at least verify the solution from stackexchange. So I coded that up
symbolically:
"""

# This is the guided solution by Stéphane Laurent
recon_sx = sympy.sqrt(a11 * a11 + a21 * a21)
recon_theta = sympy.atan2(a21, a11)
recon_sin_t = sympy.sin(recon_theta)
recon_cos_t = sympy.cos(recon_theta)

recon_msy = a12 * recon_sin_t + a22 * recon_cos_t

condition2 = sympy.simplify(sympy.Eq(recon_sin_t, 0))
condition1 = sympy.simplify(sympy.Not(condition2))
# condition1 = sympy.Gt(recon_sin_t ** 2, recon_cos_t ** 2)
# condition2 = sympy.Le(recon_sin_t ** 2, recon_cos_t ** 2)
sy_cond1 = (recon_msy * recon_cos_t - a12) / recon_sin_t
sy_cond2 = (a22 - recon_msy * recon_sin_t) / recon_cos_t

recon_sy = sympy.Piecewise((sy_cond1, condition1), (sy_cond2, condition2))

recon_m = recon_msy / recon_sy

recon_S = sympy.Matrix([  # scale
    [recon_sx,  0],
    [ 0, recon_sy]])

recon_H = sympy.Matrix([  # shear
    [1, recon_m],
    [0, 1]])

recon_R = sympy.Matrix([  # rotation
    [sympy.cos(recon_theta), -sympy.sin(recon_theta)],
    [sympy.sin(recon_theta),  sympy.cos(recon_theta)]])

# Recombine the components
A_recon = sympy.simplify((recon_R @ recon_H @ recon_S))
print(ub.hzcat(['A_recon = ', sympy.pretty(A_recon)]))


"""
That results in something quite like what I would expect, but it doesn't
seem to simplify all the way down to the point where it can be programatically
validated.


A_recon = ⎡     ⎧                                       a₂₁            ⎤
          ⎢     ⎪            a₁₂              for ──────────────── ≠ 0 ⎥
          ⎢     ⎪                                    _____________     ⎥
          ⎢     ⎪                                   ╱    2      2      ⎥
          ⎢a₁₁  ⎨                                 ╲╱  a₁₁  + a₂₁       ⎥
          ⎢     ⎪                                                      ⎥
          ⎢     ⎪a₁₁⋅a₂₂ + a₁₂⋅a₂₁ - a₂₁⋅a₂₂                           ⎥
          ⎢     ⎪───────────────────────────         otherwise         ⎥
          ⎢     ⎩            a₁₁                                       ⎥
          ⎢                                                            ⎥
          ⎢     ⎧-a₁₁⋅a₁₂ + a₁₁⋅a₂₂ + a₁₂⋅a₂₁            a₂₁           ⎥
          ⎢     ⎪────────────────────────────  for ──────────────── ≠ 0⎥
          ⎢     ⎪            a₂₁                      _____________    ⎥
          ⎢a₂₁  ⎨                                    ╱    2      2     ⎥
          ⎢     ⎪                                  ╲╱  a₁₁  + a₂₁      ⎥
          ⎢     ⎪                                                      ⎥
          ⎣     ⎩            a₂₂                      otherwise        ⎦

"""


"""
My thought is that the conditional is messing is up, so I tried just
using two cases:
"""

(A_recon2, _), (A_recon3, _) = sympy.piecewise_fold(A_recon).args
print('')
print(ub.hzcat(['A_recon2 = ', sympy.pretty(A_recon2)]))
print('')
print(ub.hzcat(['A_recon3 = ', sympy.pretty(A_recon3)]))

"""
A_recon2 = ⎡a₁₁              a₁₂             ⎤
           ⎢                                 ⎥
           ⎢     -a₁₁⋅a₁₂ + a₁₁⋅a₂₂ + a₁₂⋅a₂₁⎥
           ⎢a₂₁  ────────────────────────────⎥
           ⎣                 a₂₁             ⎦

A_recon3 = ⎡     a₁₁⋅a₂₂ + a₁₂⋅a₂₁ - a₂₁⋅a₂₂⎤
           ⎢a₁₁  ───────────────────────────⎥
           ⎢                 a₁₁            ⎥
           ⎢                                ⎥
           ⎣a₂₁              a₂₂            ⎦


But that doesn't seem to allow any further simplification.

I'm not quite seeing how a22/a12 pops out of the top/bottom equations
respectively, but they should if this decomposition is correct, but these
results are making me worried that it is not.
"""

# checks

lhs_iter = it.chain.from_iterable(A_matrix.tolist())
rhs_iter = it.chain.from_iterable(A_params.tolist())
subs = {lhs: rhs for lhs, rhs in zip(lhs_iter, rhs_iter)}
# subs.pop(A_matrix[0, 1])

recon_12 = A_recon[0, 1]
recon_12_a = A_recon2[1, 1]
recon_12_b = A_recon3[0, 1]

sympy.pretty_print(sympy.simplify(recon_12.subs(subs)))

sympy.simplify(recon_12_b.subs(subs))
sympy.simplify(recon_12_a.subs(subs))

z = A_recon[0, 1].subs(subs)
sympy.solve(z, A_matrix[0, 1])

# sympy.Eq(A_recon[0, 1], )
