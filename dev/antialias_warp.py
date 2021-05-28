

def warp_image_test(image, transform, dsize=None):
    """

    from kwimage.transform import Affine
    import kwimage
    image = kwimage.grab_test_image('checkerboard', dsize=(2048, 2048)).astype(np.float32)
    image = kwimage.grab_test_image('astro', dsize=(2048, 2048))
    transform = Affine.random() @ Affine.scale(0.01)

    """
    from kwimage.transform import Affine
    import kwimage
    import numpy as np
    import ubelt as ub

    # Choose a random affine transform that probably has a small scale
    # transform = Affine.random() @ Affine.scale((0.3, 2))
    # transform = Affine.scale((0.1, 1.2))
    # transform = Affine.scale(0.05)
    transform = Affine.random() @ Affine.scale(0.01)
    # transform = Affine.random()

    image = kwimage.grab_test_image('astro')
    image = kwimage.grab_test_image('checkerboard')

    image = kwimage.ensure_float01(image)

    from kwimage import im_cv2
    import kwarray
    import cv2
    transform = Affine.coerce(transform)

    if 1 or dsize is None:
        h, w = image.shape[0:2]

        boxes = kwimage.Boxes(np.array([[0, 0, w, h]]), 'xywh')
        poly = boxes.to_polygons()[0]
        warped_poly = poly.warp(transform.matrix)
        warped_box = warped_poly.to_boxes().to_ltrb().quantize()
        dsize = tuple(map(int, warped_box.data[0, 2:4]))

    import timerit
    ti = timerit.Timerit(10, bestof=3, verbose=2)

    def _full_gauss_kernel(k0, sigma0, scale):
        num_downscales = np.log2(1 / scale)
        if num_downscales < 0:
            return 1, 0

        # Define b0 = kernel size for one downsample operation
        b0 = 5
        # Define sigma0 = sigma for one downsample operation
        sigma0 = 1

        # The kernel size and sigma doubles for each 2x downsample
        k = int(np.ceil(b0 * (2 ** (num_downscales - 1))))
        sigma = sigma0 * (2 ** (num_downscales - 1))

        if k % 2 == 0:
            k += 1
        return k, sigma

    def pyrDownK(a, k=1):
        assert k >= 0
        for _ in range(k):
            a = cv2.pyrDown(a)
        return a

    for timer in ti.reset('naive'):
        with timer:
            interpolation = 'nearest'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v5 = cv2.warpAffine(image, transform.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 1
    #
    for timer in ti.reset('resize+warp'):
        with timer:
            params = transform.decompose()

            sx, sy = params['scale']
            noscale_params = ub.dict_diff(params, {'scale'})
            noscale_warp = Affine.affine(**noscale_params)

            h, w = image.shape[0:2]
            resize_dsize = (int(np.ceil(sx * w)), int(np.ceil(sy * h)))

            downsampled = cv2.resize(image, dsize=resize_dsize, fx=sx, fy=sy,
                                     interpolation=cv2.INTER_AREA)

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v1 = cv2.warpAffine(downsampled, noscale_warp.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 2
    for timer in ti.reset('fullblur+warp'):
        with timer:
            k_x, sigma_x = _full_gauss_kernel(k0=5, sigma0=1, scale=sx)
            k_y, sigma_y = _full_gauss_kernel(k0=5, sigma0=1, scale=sy)
            image_ = image.copy()
            image_ = cv2.GaussianBlur(image_, (k_x, k_y), sigma_x, sigma_y)
            image_ = kwarray.atleast_nd(image_, 3)
            # image_ = image_.clip(0, 1)

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v2 = cv2.warpAffine(image_, transform.matrix[0:2], dsize=dsize, flags=flags)

    # --------------------
    # METHOD 3

    for timer in ti.reset('pyrDown+blur+warp'):
        with timer:
            temp = image.copy()
            params = transform.decompose()
            sx, sy = params['scale']

            biggest_scale = max(sx, sy)
            # The -2 allows the gaussian to be a little bigger. This
            # seems to help with border effects at only a small runtime cost
            num_downscales = max(int(np.log2(1 / biggest_scale)) - 2, 0)
            pyr_scale = 1 / (2 ** num_downscales)

            # Does the gaussian downsampling
            temp = pyrDownK(image, num_downscales)

            rest_sx = sx / pyr_scale
            rest_sy = sy / pyr_scale

            partial_scale = Affine.scale((rest_sx, rest_sy))
            rest_warp = noscale_warp @ partial_scale

            k_x, sigma_x = _full_gauss_kernel(k0=5, sigma0=1, scale=rest_sx)
            k_y, sigma_y = _full_gauss_kernel(k0=5, sigma0=1, scale=rest_sy)
            temp = cv2.GaussianBlur(temp, (k_x, k_y), sigma_x, sigma_y)
            temp = kwarray.atleast_nd(temp, 3)

            interpolation = 'cubic'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v3 = cv2.warpAffine(temp, rest_warp.matrix[0:2], dsize=dsize,
                                      flags=flags)

    # --------------------
    # METHOD 4 - dont do the final blur

    for timer in ti.reset('pyrDown+warp'):
        with timer:
            temp = image.copy()
            params = transform.decompose()
            sx, sy = params['scale']

            biggest_scale = max(sx, sy)
            num_downscales = max(int(np.log2(1 / biggest_scale)), 0)
            pyr_scale = 1 / (2 ** num_downscales)

            # Does the gaussian downsampling
            temp = pyrDownK(image, num_downscales)

            rest_sx = sx / pyr_scale
            rest_sy = sy / pyr_scale

            partial_scale = Affine.scale((rest_sx, rest_sy))
            rest_warp = noscale_warp @ partial_scale

            interpolation = 'linear'
            flags = im_cv2._coerce_interpolation(interpolation)
            final_v4 = cv2.warpAffine(temp, rest_warp.matrix[0:2], dsize=dsize, flags=flags)

    if 1:

        def get_title(key):
            from ubelt.timerit import _choose_unit
            value = ti.measures['mean'][key]
            suffix, mag = _choose_unit(value)
            unit_val = value / mag

            return key + ' ' + ub.repr2(unit_val, precision=2) + ' ' + suffix

        final_v2 = final_v2.clip(0, 1)
        final_v1 = final_v1.clip(0, 1)
        final_v3 = final_v3.clip(0, 1)
        final_v4 = final_v4.clip(0, 1)
        final_v5 = final_v5.clip(0, 1)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(final_v5, pnum=(1, 5, 1), title=get_title('naive'))
        kwplot.imshow(final_v2, pnum=(1, 5, 2), title=get_title('fullblur+warp'))
        kwplot.imshow(final_v1, pnum=(1, 5, 3), title=get_title('resize+warp'))
        kwplot.imshow(final_v3, pnum=(1, 5, 4), title=get_title('pyrDown+blur+warp'))
        kwplot.imshow(final_v4, pnum=(1, 5, 5), title=get_title('pyrDown+warp'))
        # kwplot.imshow(np.abs(final_v2 - final_v1), pnum=(1, 4, 4))


def warp_affine(image, transform, dsize=None, antialias=True,
                interpolation='linear'):
    """
    Applies an affine transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image

        transform (ndarray | Affine): a coercable affine matrix

        dsize (Tuple[int, int] | None | str):
            width and height of the resulting image. If "auto", it is computed
            such that the positive coordinates of the warped image will fit in
            the new canvas. If None, then the image size will not change.

        antialias (bool, default=True):
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur.

    TODO:
        - [ ] This will be moved to kwimage.im_cv2

    Example:
        >>> import kwimage
        >>> image = kwimage.grab_test_image('astro')
        >>> image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale(0.05)
        >>> transform = Affine.scale(0.02)
        >>> warped1 = warp_affine(image, transform, dsize='auto', antialias=1, interpolation='nearest')
        >>> warped2 = warp_affine(image, transform, dsize='auto', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()

    Example:
        >>> import kwimage
        >>> image = kwimage.grab_test_image('astro')
        >>> image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale((.1, 1.2))
        >>> warped1 = warp_affine(image, transform, dsize='auto', antialias=1)
        >>> warped2 = warp_affine(image, transform, dsize='auto', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()
    """
    from kwimage import im_cv2
    from kwimage.transform import Affine
    import kwimage
    import numpy as np
    import cv2
    import ubelt as ub
    transform = Affine.coerce(transform)
    flags = im_cv2._coerce_interpolation(interpolation)

    # TODO: expose these params
    # borderMode = cv2.BORDER_DEFAULT
    # borderMode = cv2.BORDER_CONSTANT
    borderMode = None
    borderValue = None

    """
    Variations that could change in the future:

        * In _gauss_params I'm not sure if we want to compute integer or
            fractional "number of downsamples".

        * The fudge factor bothers me, but seems necessary
    """

    def _gauss_params(scale, k0=5, sigma0=1, fractional=True):
        # Compute a gaussian to mitigate aliasing for a requested downsample
        # Args:
        # scale: requested downsample factor
        # k0 (int): kernel size for one downsample operation
        # sigma0 (float): sigma for one downsample operation
        # fractional (bool): controls if we compute params for integer downsample
        # ops
        num_downs = np.log2(1 / scale)
        if not fractional:
            num_downs = max(int(num_downs), 0)
        if num_downs <= 0:
            k = 1
            sigma = 0
        else:
            # The kernel size and sigma doubles for each 2x downsample
            sigma = sigma0 * (2 ** (num_downs - 1))
            k = int(np.ceil(k0 * (2 ** (num_downs - 1))))
            k = k + int(k % 2 == 0)
        return k, sigma

    def _pyrDownK(a, k=1):
        # Downsamples by (2 ** k)x with antialiasing
        if k == 0:
            a = a.copy()
        for _ in range(k):
            a = cv2.pyrDown(a)
        return a

    if dsize is None:
        dsize = tuple(image.shape[0:2][::-1])
    elif dsize == 'auto':
        h, w = image.shape[0:2]
        boxes = kwimage.Boxes(np.array([[0, 0, w, h]]), 'xywh')
        poly = boxes.to_polygons()[0]
        warped_poly = poly.warp(transform.matrix)
        warped_box = warped_poly.to_boxes().to_ltrb().quantize()
        dsize = tuple(map(int, warped_box.data[0, 2:4]))

    if not antialias:
        M = np.asarray(transform)
        result = cv2.warpAffine(image, M[0:2],
                                dsize=dsize, flags=flags,
                                borderMode=borderMode,
                                borderValue=borderValue)
    else:
        # Decompose the affine matrix into its 6 core parameters
        params = transform.decompose()
        sx, sy = params['scale']

        if sx >= 1 and sy > 1:
            # No downsampling detected, no need to antialias
            M = np.asarray(transform)
            result = cv2.warpAffine(image, M[0:2], dsize=dsize, flags=flags,
                                    borderMode=borderMode,
                                    borderValue=borderValue)
        else:
            # At least one dimension is downsampled

            # Compute the transform with all scaling removed
            noscale_warp = Affine.affine(**ub.dict_diff(params, {'scale'}))

            max_scale = max(sx, sy)
            # The "fudge" factor limits the number of downsampled pyramid
            # operations. A bigger fudge factor means means that the final
            # gaussian kernel for the antialiasing operation will be bigger.
            # It essentials say that at most "fudge" downsampling ops will
            # be handled by the final blur rather than the pyramid downsample.
            # It seems to help with border effects at only a small runtime cost
            # I don't entirely understand why the border artifact is introduced
            # when this is enabled though

            # TODO: should we allow for this fudge factor?
            # TODO: what is the real name of this? num_down_prevent ?
            # skip_final_downs?
            fudge = 2
            # TODO: should final antialiasing be on?
            # Note, if fudge is non-zero it is important to do this.
            do_final_aa = 1
            # TODO: should fractional be True or False by default?
            # If fudge is 0 and fractional=0, then I think is the same as
            # do_final_aa=0.
            fractional = 0

            num_downs = max(int(np.log2(1 / max_scale)) - fudge, 0)
            pyr_scale = 1 / (2 ** num_downs)

            # Downsample iteratively with antialiasing
            downscaled = _pyrDownK(image, num_downs)

            rest_sx = sx / pyr_scale
            rest_sy = sy / pyr_scale

            # Compute the transform from the downsampled image to the destination
            rest_warp = noscale_warp @ Affine.scale((rest_sx, rest_sy))

            # Do a final small blur to acount for the potential aliasing
            # in any remaining scaling operations.
            if do_final_aa:
                # Computed as the closest sigma to the [1, 4, 6, 4, 1] approx
                # used in cv2.pyrDown
                aa_sigma0 = 1.0565137190917149
                aa_k0 = 5
                k_x, sigma_x = _gauss_params(scale=rest_sx, k0=aa_k0,
                                             sigma0=aa_sigma0,
                                             fractional=fractional)
                k_y, sigma_y = _gauss_params(scale=rest_sy, k0=aa_k0,
                                             sigma0=aa_sigma0,
                                             fractional=fractional)

                # Note: when k=1, no blur occurs
                # blurBorderType = cv2.BORDER_REPLICATE
                # blurBorderType = cv2.BORDER_CONSTANT
                blurBorderType = cv2.BORDER_DEFAULT
                downscaled = cv2.GaussianBlur(
                    downscaled, (k_x, k_y), sigma_x, sigma_y,
                    borderType=blurBorderType
                )

            result = cv2.warpAffine(downscaled, rest_warp.matrix[0:2],
                                    dsize=dsize, flags=flags,
                                    borderMode=borderMode,
                                    borderValue=borderValue)

    return result


def _check():
    # Find the sigma closest to the pyrDown op [1, 4, 6, 4, 1] / 16
    import cv2
    import numpy as np
    import scipy
    import ubelt as ub
    def sigma_error(sigma):
        sigma = np.asarray(sigma).ravel()[0]
        got = (cv2.getGaussianKernel(5, sigma) * 16).ravel()
        want = np.array([1, 4, 6, 4, 1])
        loss = ((got - want) ** 2).sum()
        return loss
    result = scipy.optimize.minimize(sigma_error, x0=1.0, method='Nelder-Mead')
    print('result = {}'.format(ub.repr2(result, nl=1)))

    best_loss = float('inf')
    best = None
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
               # 'Newton-CG',
               'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
               # 'trust-constr',
               # 'dogleg',
               # 'trust-ncg', 'trust-exact',
               # 'trust-krylov'
              ]
    # results = {}
    x0 = 1.06
    for i in range(100):
        for method in methods:
            best_method_loss, best_method_sigma, best_method_x0 = results.get(method, (float('inf'), 1.06, x0))
            result = scipy.optimize.minimize(sigma_error, x0=x0, method=method)
            sigma = np.asarray(result.x).ravel()[0]
            loss = sigma_error(sigma)
            if loss <= best_method_loss:
                results[method] = (loss, sigma, x0)

        best_method = ub.argmin(results)
        best_loss, best_sigma = results[best_method][0:2]
        rng = np.random
        if rng.rand() > 0.5:
            x0 = best_sigma
        else:
            x0 = best_sigma + rng.rand() * 0.0001
        print('best_method = {!r}'.format(best_method))
        print('best_loss = {!r}'.format(best_loss))
        print('best_sigma = {!r}'.format(best_sigma))

    print('results = {}'.format(ub.repr2(results, nl=1, align=':')))
    print('best_method = {}'.format(ub.repr2(best_method, nl=1)))
    print('best_method = {!r}'.format(best_method))

    sigma_error(1.0565139268118493)
    sigma_error(1.0565137190917149)
    # scipy.optimize.minimize_scalar(sigma_error, bounds=(1, 1.1))

    import kwarray
    import numpy as np
    a = (kwarray.ensure_rng(0).rand(32, 32) * 256).astype(np.uint8)
    a = kwimage.ensure_float01(a)
    b = cv2.GaussianBlur(a.copy(), (1, 1), 3, 3)
    assert np.all(a == b)

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('time'):
        with timer:
            b = cv2.GaussianBlur(a.copy(), (9, 9), 3, 3)

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('time'):
        with timer:
            c = cv2.GaussianBlur(a.copy(), (1, 9), 3, 3)
            zR= cv2.GaussianBlur(c.copy(), (9, 1), 3, 3)
