import ubelt as ub
import sympy
import sympy as sym  # NOQA
# Shows the symbolic construction of the code
# https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
# G
# aff2 = np.array(sympy.symbols('a0:9')).reshape((3, 3))
# eqn = sympy.Eq(sympy.Matrix(aff2), sympy.Matrix(aff))
# for p in params:
#     print('p = {!r}'.format(p))
#     soln = sympy.solve(eqn, p)
#     print('soln = {!r}'.format(soln))
# sympy.solve(eqn, sx)
# sympy.solve(eqn, aff2[2, 0])


# params = sx, sy, theta, m = sympy.symbols('sx, sy, theta, m')
# # Simplified
# S = np.array([[sx,  0], [ 0, sy]])
# H = np.array([[1, m], [0, 1]])
# R = np.array([
#     [sympy.cos(theta), -sympy.sin(theta)],
#     [sympy.sin(theta),  sympy.cos(theta)]])
# aff1 = np.array(sympy.simplify(R @ H @ S))
# # Solve for the decomposition too
# aff2 = np.array(sympy.symbols('a0:4')).reshape((2, 2))
# eqn = sympy.Eq(sympy.Matrix(aff1), sympy.Matrix(aff2))

# theta_soln1 = sympy.solve(eqn, theta)[1][0]
# eqn2 = eqn.subs(theta, theta_soln1)

# sx_soln1 = sympy.solve(eqn2, sx)[0][0]
# eqn3 = eqn2.subs(sx, sx_soln1)

# for p in params:
#     print('p = {!r}'.format(p))
#     soln = sympy.solve(eqn3, p)
#     print('soln = {!r}'.format(soln))

# ## --------- TRY AGAIN

tau = sympy.pi * 2

domain = {
    'real': True,
}

theta = sympy.symbols('theta', **domain)
sx, sy = sympy.symbols('sx, sy', nonzero=True, **domain)
shear = sympy.symbols('shear', **domain)
m = sympy.symbols('m', **domain)

USE_SHEAR = False

S = sympy.Matrix([  # scale
    [sx,  0, 0],
    [ 0, sy, 0],
    [ 0,  0, 1]])

if USE_SHEAR:
    H = sympy.Matrix([  # shear
        [1, -sympy.sin(shear), 0],
        [0,  sympy.cos(shear), 0],
        [0,                 0, 1]])
else:
    H = sympy.Matrix([  # shear
        [1, m, 0],
        [0, 1, 0],
        [0, 0, 1]])
R = sympy.Matrix([  # rotation
    [sympy.cos(theta), -sympy.sin(theta), 0],
    [sympy.sin(theta),  sympy.cos(theta), 0],
    [               0,                 0, 1]])


A_params = sympy.simplify((R @ H @ S))
a11, a12, a13, a21, a22, a23, a31, a32, a33 = sympy.symbols(
    'a11, a12, a13, a21, a22, a23, a31, a32, a33', real=True)
A_matrix = sympy.Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


# This is the guided solution by Stéphane Laurent
recon_sx = sympy.sqrt(a11 * a11 + a21 * a21)
recon_theta = sympy.atan2(a21, a11)
recon_sin_t = sympy.sin(recon_theta)
recon_cos_t = sympy.cos(recon_theta)


# Check that recons work?


print('recon_sx = {!r}'.format(recon_sx))
print('recon_theta = {!r}'.format(recon_theta))

recon_msy = a12 * recon_sin_t + a22 * recon_cos_t
# condition1 = sympy.Lt(abs(recon_cos_t), abs(recon_sin_t)).simplify()
condition1 = sympy.simplify(sympy.Ne(recon_sin_t, 0))
condition2 = sympy.simplify(sympy.Not(condition1))
sy_cond1 = (recon_msy * recon_cos_t - a12) / recon_sin_t
sy_cond2 = (a22 - recon_msy * recon_sin_t) / recon_cos_t
if True:
    recon_sy = sympy.Piecewise((sy_cond1, condition1), (sy_cond2, condition2))
else:
    recon_sy = sy_cond1

recon_m = recon_msy / recon_sy


equation = sympy.Eq(A_params, A_matrix)

sympy.solve(equation, sx)


recon_S = sympy.Matrix([  # scale
    [recon_sx,  0, 0],
    [ 0, recon_sy, 0],
    [ 0,  0, 1]])

if USE_SHEAR:
    recon_shear = None
    recon_H = sympy.Matrix([  # shear
        [1, -sympy.sin(recon_shear), 0],
        [0,  sympy.cos(recon_shear), 0],
        [0,                 0, 1]])
else:
    recon_H = sympy.Matrix([  # shear
        [1, recon_m, 0],
        [0, 1, 0],
        [0, 0, 1]])

recon_R = sympy.Matrix([  # rotation
    [sympy.cos(recon_theta), -sympy.sin(recon_theta), 0],
    [sympy.sin(recon_theta),  sympy.cos(recon_theta), 0],
    [               0,                 0, 1]])

A_recon1 = (recon_R @ recon_H @ recon_S)
A_recon2 = sympy.simplify(A_recon1)

sympy.pretty_print(A_matrix)
sympy.pretty_print(A_params)
sympy.pretty_print(A_recon2)


#### Try and make sympy solve it

lhs_iter = ub.flatten(A_matrix.tolist())
rhs_iter = ub.flatten(A_params.tolist())
tocheck = {}
for lhs, rhs in zip(lhs_iter, rhs_iter):
    tocheck[lhs] = rhs


equality = sympy.Eq(A_matrix, A_params)

sympy.solvers.linsolve(equality, sx)
sympy.solveset(equality, sx, domain=sympy.Reals)

theta_ranges = [sympy.pi, sympy.pi / 2, sympy.pi / 3, sympy.pi / 4]
theta_ranges += [-sympy.pi, -sympy.pi / 2, -sympy.pi / 3, -sympy.pi / 4]
theta_ranges += [0]

for theta_val in theta_ranges:
    equations_theta_inst = sympy.simplify(equality.subs(theta, theta_val))
    soln_part = sympy.solve(equations_theta_inst, sx)
    print('soln_part = {!r}'.format(soln_part))

equations_theta_neg = sympy.simplify(equality.subs(theta, -sympy.pi / 2))
equations_theta_zer = sympy.simplify(equality.subs(theta, 0))

equations = [
    sympy.Eq(A_matrix[0, 0], A_params[0, 0]),
    sympy.Eq(A_matrix[0, 1], A_params[0, 1]),
    sympy.Eq(A_matrix[1, 0], A_params[1, 0]),
    sympy.Eq(A_matrix[1, 1], A_params[1, 1]),
]

sympy.solve(equations, (sx, theta))

for e in equations:
    sympy.pretty_print(e)
print('equations = {}'.format(ub.urepr(equations, nl=1)))

check_sx = sympy.simplify(recon_sx.subs(tocheck))
sympy.pretty_print(check_sx)
check_sy = sympy.simplify(recon_sy.subs(tocheck))
sympy.pretty_print(check_sy)
check_theta = sympy.simplify(recon_theta.subs(tocheck))
sympy.pretty_print(check_theta)
sympy.pretty_print(check_sy)

sympy.solve(equations, sx)
sympy.solve(equations, sy)
sympy.solve(equations, m)

soln_theta0 = sympy.solve(equations, theta)[1][0]
soln_theta1 = soln_theta0.subs(sx, recon_sx)

# sympy.simplify(sympy.Eq(recon_theta - soln_theta1, 0))


# Can we get the skimage formulation to work?
if 0:
    # Note: shearx is not shear, but we can use it to solve for it
    # shear = sympy.symbols('shear', **domain)
    # shear_equations = [
    #     sympy.Eq(sy * (shearx * sympy.cos(theta) - sympy.sin(theta)), -sy * sympy.sin(shear + theta)),
    #     sympy.Eq(sy * (shearx * sympy.sin(theta) + sympy.cos(theta)),  sy * sympy.cos(shear + theta))
    # ]
    # sympy.solve(shear_equations[0], shear)
    # sympy.solve(shear_equations[1], shear)
    # [-theta - asin(shearx*cos(theta) - sin(theta)),
    #  -theta + asin(shearx*cos(theta) - sin(theta)) + pi]
    # [-theta + acos(shearx*sin(theta) + cos(theta)),
    #  -theta - acos(shearx*sin(theta) + cos(theta)) + 2*pi]
    mc_sub_s = shearx * np.cos(theta) - np.sin(theta)
    if abs(mc_sub_s) <= 1:
        shear0 = -theta - mc_sub_s
        shear1 = -theta + mc_sub_s + np.pi
    else:
        ms_add_c = shearx * np.sin(theta) + np.cos(theta)
        shear0 = -theta + ms_add_c
        shear1 = -theta - ms_add_c + 2 * np.pi

    def normalize_angle(radian):
        return np.arctan2(np.sin(radian), np.cos(radian))
    shear0 = normalize_angle(shear0)
    shear1 = normalize_angle(shear1)
    # sklearn def
    # if 0:
    #     rot = math.atan2(a21, a11)
    #     beta = math.atan2(-a12, a11)
    #     sklearn_shear = beta - rot
