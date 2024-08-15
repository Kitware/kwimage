def main():
    """
    Posted to: https://github.com/opencv/opencv/issues/11784
    """
    import ubelt as ub
    import sympy
    # Shows the symbolic construction of the code
    # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
    x0, y0, sx, sy, theta, shearx, tx, ty, u, v = sympy.symbols(
        'x0, y0, s_x, s_y, theta, e_x, t_x, t_y, u, v')

    a, b, c, d, e, f, g, h, i = sympy.symbols('a, b, c, d, e, f, g, h, i')
    M = sympy.Matrix([[a, b, c],
                      [d, e, f],
                      [g, h, i]])

    tr1_ = sympy.Matrix([[1, 0,  -x0],
                         [0, 1,  -y0],
                         [0, 0,    1]])
    P = sympy.Matrix([  # projective part
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ u,  v,  1]])
    # Define core components of the affine transform
    S = sympy.Matrix([  # scale
        [sx,  0, 0],
        [ 0, sy, 0],
        [ 0,  0, 1]])
    E = sympy.Matrix([  # x-shear
        [1,  shearx, 0],
        [0,  1, 0],
        [0,  0, 1]])
    R = sympy.Matrix([  # rotation
        [sympy.cos(theta), -sympy.sin(theta), 0],
        [sympy.sin(theta),  sympy.cos(theta), 0],
        [               0,                 0, 1]])
    T = sympy.Matrix([  # translation
        [ 1,  0, tx],
        [ 0,  1, ty],
        [ 0,  0,  1]])
    # move 0, 0 back to the specified origin
    tr2_ = sympy.Matrix([[1, 0,  x0],
                         [0, 1,  y0],
                         [0, 0,   1]])

    # combine transformations
    affine_components = sympy.MatMul(tr2_, T, R, E, S, tr1_)
    affine_noshift_components = sympy.MatMul(T, R, E, S)
    affine = affine_components.doit()
    affine_noshift = affine_noshift_components.doit()
    affine_components_tex = sympy.latex(affine_components)
    affine_tex = sympy.latex(affine)
    affine_noshift_components_tex = sympy.latex(affine_noshift_components)
    affine_noshift_tex = sympy.latex(affine_noshift)

    homog_components = sympy.MatMul(tr2_, T, R, E, S, P, tr1_)
    homog_noshift_components = sympy.MatMul(T, R, E, S, P)
    homog = homog_components.doit()
    homog_noshift = homog_noshift_components.doit()
    homog_components_tex = sympy.latex(homog_components)
    homog_tex = sympy.latex(homog)
    homog_noshift_components_tex = sympy.latex(homog_noshift_components)
    homog_noshift_tex = sympy.latex(homog_noshift)

    import pylatex

    # Dont use fontenc, lmodern, or textcomp
    # https://tex.stackexchange.com/questions/179778/xelatex-under-ubuntu
    doc = pylatex.Document('matrix_doc', inputenc=None,
                           page_numbers=False, indent=False, fontenc=None,
                           lmodern=1,
                           textcomp=False,
                           geometry_options='paperheight=10in,paperwidth=18in,margin=.1in',
                           )
    doc.preamble.append(pylatex.Package('hyperref'))  # For PNG images

    doc.append(pylatex.Section('Transformation Ingredients'))
    doc.append(pylatex.NoEscape(
        fr'''
        The following matrices represent simple linear transformations, which
        can be used to build more complex linear transforms. These matrices are
        minimal in that you can always decompose a 3x3 matrix into some product
        of these matrices.

        \begin{{itemize}}

            \item A projection matrix $P = {sympy.latex(P)}$ which "hinges" the image about
                the origin $(0, 0)$ with a x-hinge-magnitude of $u$ and a y-hinge-magnitude
                of $v$. More precisely it sends the line $ux+vy=-1$ to infinity.
                For more explanation and a visualization of what this means see \\\\
                \url{{https://math.stackexchange.com/questions/1319680/what-is-the-difference-between-affine-and-projective-transformations}}

            \item A scale matrix $S = {sympy.latex(S)}$ which scales by $s_x$ in the x direction and
                $s_y$ in the y direction.

            \item A shear matrix $E = {sympy.latex(E)}$ which shears by $e_x$ in the x direction.
                Note that a shear in the y direction can be achieved by combining a
                rotation with a shear in x direction. As such it is always possible
                to decompose a transformation matrix into one where there all
                shear is the x direction, thus we do not explicitly use
                a $e_y$ in this document.

            \item A rotation matrix $R = {sympy.latex(R)}$ which rotates by $\theta$ radians in
                the counter-clockwise direction around the origin $(0, 0)$.

            \item A translation matrix $T = {sympy.latex(T)}$ which shifts by $t_x$ in the x
                direction and $t_y$ in the y direction.

        \end{{itemize}}
        '''
        ))

    doc.append(pylatex.NoEscape(ub.paragraph(
        f'''
        It is also useful to be able to modify which point we are considering
        the origin. By default it is $(0, 0)$, but if we have some arbitrary
        linear transform $M = {sympy.latex(M)}$, we can always transform about a new origin
        $(x_0, y_0)$ by "wrapping" the matrix two translation matrices, one that
        shifts the desired origin to $(0, 0)$ and a final translation that
        shifts it back.
        ''')))

    shift_components = sympy.MatMul(tr2_, M, tr1_)
    shift = shift_components.doit()

    doc.append(pylatex.Math(data=[sympy.latex(shift_components), '=', sympy.latex(shift)], escape=False))

    doc.append(pylatex.Section('Affine Matrix Components'))
    doc.append(pylatex.NoEscape('An affine matrix $A$ about the origin can be constructed as a scale, shear, rotation, and translation. $T R E S = A$. Expanded out this looks like:'))
    doc.append(pylatex.Math(data=[affine_noshift_components_tex, '=', affine_noshift_tex], escape=False))
    doc.append('And adding an origin shift the general form is:')
    doc.append(pylatex.Math(data=[affine_components_tex, '=', affine_tex], escape=False))

    doc.append(pylatex.Section('Homography Matrix Components'))
    doc.append(pylatex.NoEscape('An projective (or homography) matrix $H$ about the origin is generalizes and affine transform adding a projection as the first operation. Explicitly, homography matrix can be constructed as a projection, scale, shear, rotation, and translation. $T R E S P = H$. Expanded out this looks like:'))
    doc.append(pylatex.Math(data=[homog_noshift_components_tex, '=', homog_noshift_tex], escape=False))
    doc.append('And adding an origin shift the general form is:')
    doc.append(pylatex.Math(data=[homog_components_tex, '=', homog_tex], escape=False))

    print(doc.dumps())
    print('generate pdf')

    pdf_fpath = ub.Path('~/code/kwimage/docs/temp/projective_maths/projective_maths.pdf').expand()
    pdf_fpath.parent.ensuredir()
    doc.generate_pdf(pdf_fpath.augment(ext=''), clean_tex=True)

    import xdev
    xdev.startfile(pdf_fpath)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/devcheck/projective_script.py
        python projective_script.py
    """
    main()
