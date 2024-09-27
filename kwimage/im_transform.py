"""
Contains functions that used to belong to im_cv2, but have been generalized to
accept different backends.
"""
from kwimage._backend_info import _default_backend


def warp_image(image, transform, dsize=None, antialias=False,
               interpolation='linear', border_mode=None, border_value=0,
               large_warp_dim=None, origin_convention='center',
               return_info=False, backend='auto'):
    """
    Applies an transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image as a numpy array.
            Note: this is passed directly to cv2, so it is best to ensure that
            it is contiguous and using a dtype that cv2 can handle.

        transform (ndarray | dict | kwimage.Matrix): a coercable affine or
            projective matrix.  See :class:`kwimage.Affine` and
            :class:`kwimage.Projective` for details on what can be coerced.

        dsize (Tuple[int, int] | None | str):
            A integer width and height tuple of the resulting "canvas" image.
            If None, then the input image size is used.

            If specified as a string, dsize is computed based on the given
            heuristic.

            If 'positive' (or 'auto'), dsize is computed such that the positive
            coordinates of the warped image will fit in the new canvas. In this
            case, any pixel that maps to a negative coordinate will be clipped.
            This has the property that the input transformation is not
            modified.

            If 'content' (or 'max'), the transform is modified with an extra
            translation such that both the positive and negative coordinates of
            the warped image will fit in the new canvas.

        antialias (bool)
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur. Defaults to False

        interpolation (str | int):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area. Defaults to "linear".

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float]):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.

        large_warp_dim (int | None | str):
            If specified, perform the warp piecewise in chunks of the specified
            size. If "auto", it is set to the maximum "short" value in numpy.
            This works around a limitation of cv2.warpAffine, which must have
            image dimensions < SHRT_MAX (=32767 in version 4.5.3)

        origin_convention (str):
            Controls the interpretation of the underlying raster.
            Can be "center" (default opencv behavior), or "corner" (matches
            torchvision / detectron2 behavior).
            If "center", then center of the top left pixel is at (0, 0), and
            the top left corner is at (-0.5, -0.5).
            If "center", then center of the top left pixel is at (0.5, 0.5), and
            the top left corner is at (0, 0).
            Currently defaults to "center", but in the future we may change the
            default to "corner".  For more info see [WhereArePixels]_.

        return_info (bool):
            if True, returns information about the operation. In the case
            where dsize="content", this includes the modified transformation.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            the warped image, or if return info is True, the warped image and
            the info dictionary.

    SeeAlso:
        :func:`kwimage.warp_tensor`
        :func:`kwimage.warp_affine`
        :func:`kwimage.warp_projective`

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import ubelt as ub
        >>> image = kwimage.grab_test_image('paraview')
        >>> tf_homog = kwimage.Projective.random(rng=30342110) @ kwimage.Projective.coerce(uv=[0.001, 0.001])
        >>> tf_aff = kwimage.Affine.coerce(ub.udict(tf_homog.decompose()) - {'uv'})
        >>> tf_uv = kwimage.Projective.coerce(ub.udict(tf_homog.decompose()) & {'uv'})
        >>> warped1 = kwimage.warp_image(image, tf_homog, dsize='positive')
        >>> warped2 = kwimage.warp_image(image, tf_aff, dsize='positive')
        >>> warped3 = kwimage.warp_image(image, tf_uv, dsize='positive')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=2, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='projective warp')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='affine warp')
        >>> kwplot.imshow(warped3, pnum=pnum_(), title='projective part')
        >>> kwplot.show_if_requested()
    """
    import kwimage
    transform = kwimage.Projective.coerce(transform)
    kwargs = dict(dsize=dsize, antialias=antialias,
                  interpolation=interpolation, border_mode=border_mode,
                  border_value=border_value, large_warp_dim=large_warp_dim,
                  origin_convention=origin_convention,
                  return_info=return_info, backend=backend)
    if transform.is_affine():
        return kwimage.warp_affine(image, transform, **kwargs)
    else:
        return kwimage.warp_projective(image, transform, **kwargs)


def warp_projective(image, transform, dsize=None, antialias=False,
                    interpolation='linear', border_mode=None, border_value=0,
                    large_warp_dim=None, origin_convention='center',
                    return_info=False, backend='auto'):
    """
    Applies an projective transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image as a numpy array.
            Note: this is passed directly to cv2, so it is best to ensure that
            it is contiguous and using a dtype that cv2 can handle.

        transform (ndarray | dict | kwimage.Projective): a coercable projective matrix.
            See :class:`kwimage.Projective` for details on what can be coerced.

        dsize (Tuple[int, int] | None | str):
            A integer width and height tuple of the resulting "canvas" image.
            If None, then the input image size is used.

            If specified as a string, dsize is computed based on the given
            heuristic.

            If 'positive' (or 'auto'), dsize is computed such that the positive
            coordinates of the warped image will fit in the new canvas. In this
            case, any pixel that maps to a negative coordinate will be clipped.
            This has the property that the input transformation is not
            modified.

            If 'content' (or 'max'), the transform is modified with an extra
            translation such that both the positive and negative coordinates of
            the warped image will fit in the new canvas.

        antialias (bool)
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur. Defaults to False

        interpolation (str | int):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area. Defaults to "linear".

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float]):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.

        large_warp_dim (int | None | str):
            If specified, perform the warp piecewise in chunks of the specified
            size. If "auto", it is set to the maximum "short" value in numpy.
            This works around a limitation of cv2.warpAffine, which must have
            image dimensions < SHRT_MAX (=32767 in version 4.5.3)

        origin_convention (str):
            Controls the interpretation of the underlying raster.
            Can be "center" (default opencv behavior), or "corner" (matches
            torchvision / detectron2 behavior).
            If "center", then center of the top left pixel is at (0, 0), and
            the top left corner is at (-0.5, -0.5).
            If "center", then center of the top left pixel is at (0.5, 0.5), and
            the top left corner is at (0, 0).
            Currently defaults to "center", but in the future we may change the
            default to "corner".  For more info see [WhereArePixels]_.

        return_info (bool):
            if True, returns information about the operation. In the case
            where dsize="content", this includes the modified transformation.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            the warped image, or if return info is True, the warped image and
            the info dictionary.
    """
    if backend == 'auto':
        backend = _default_backend()

    if backend == 'cv2':
        from kwimage.im_cv2 import _cv2_warp_projective
        return _cv2_warp_projective(
            image=image,
            transform=transform,
            dsize=dsize,
            antialias=antialias,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
            large_warp_dim=large_warp_dim,
            origin_convention=origin_convention,
            return_info=return_info,
        )
    elif backend == 'itk':
        raise NotImplementedError(f'no kwimage backend={backend} for warp_projective')
    elif backend == 'torch':
        raise NotImplementedError(f'no kwimage backend={backend} for warp_projective')
    elif backend == 'skimage':
        raise NotImplementedError(f'no kwimage backend={backend} for warp_projective')
    else:
        raise KeyError(f'no kwimage backend={backend} for warp_projective')


def warp_affine(image, transform, dsize=None, antialias=False,
                interpolation='linear', border_mode=None, border_value=0,
                large_warp_dim=None, origin_convention='center',
                return_info=False, backend='auto'):
    """
    Applies an affine transformation to an image with optional antialiasing.

    Args:
        image (ndarray): the input image as a numpy array.
            Note: this is passed directly to cv2, so it is best to ensure that
            it is contiguous and using a dtype that cv2 can handle.

        transform (ndarray | dict | kwimage.Affine): a coercable affine matrix.
            See :class:`kwimage.Affine` for details on what can be coerced.

        dsize (Tuple[int, int] | None | str):
            A integer width and height tuple of the resulting "canvas" image.

            If None, then the input image size is used.

            If specified as a string, dsize is computed based on the given
            heuristic.

            If 'positive' (or 'auto'), dsize is computed such that the positive
            coordinates of the warped image will fit in the new canvas. In this
            case, any pixel that maps to a negative coordinate will be clipped.
            This has the property that the input transformation is not
            modified. NOTE: there are issues with this when the transformation
            includes rotation of reflections.

            If 'content' (or 'max'), the transform is modified with an extra
            translation such that both the positive and negative coordinates of
            the warped image will fit in the new canvas.

        antialias (bool)
            if True determines if the transform is downsampling and applies
            antialiasing via gaussian a blur. Defaults to False

        interpolation (str | int):
            interpolation code or cv2 integer. Interpolation codes are linear,
            nearest, cubic, lancsoz, and area. Defaults to "linear".

        border_mode (str | int):
            Border code or cv2 integer. Border codes are constant (default)
            replicate, reflect, wrap, reflect101, and transparent.

        border_value (int | float | Iterable[int | float] | str):
            Used as the fill value if border_mode is constant. Otherwise this
            is ignored. Defaults to 0, but can also be defaulted to nan.
            if border_value is a scalar and there are multiple channels, the
            value is applied to all channels. More than 4 unique border values
            for individual channels will cause an error. See OpenCV #22283 for
            details.  In the future we may accept np.ma and return a masked
            array, but for now that is not implemented.
            If a string, it indicates a border_mode that is not constant.
            In this case border mode should not be given.

        large_warp_dim (int | None | str):
            If specified, perform the warp piecewise in chunks of the specified
            size. If "auto", it is set to the maximum "short" value in numpy.
            This works around a limitation of cv2.warpAffine, which must have
            image dimensions < SHRT_MAX (=32767 in version 4.5.3)

        return_info (bool):
            if True, returns information about the operation. In the case
            where dsize="content", this includes the modified transformation.

        origin_convention (str):
            Controls the interpretation of the underlying raster.
            Can be "center" (default opencv behavior), or "corner" (matches
            torchvision / detectron2 behavior).
            If "center", then center of the top left pixel is at (0, 0), and
            the top left corner is at (-0.5, -0.5).
            If "center", then center of the top left pixel is at (0.5, 0.5), and
            the top left corner is at (0, 0).
            Currently defaults to "center", but in the future we may change the
            default to "corner".  For more info see [WhereArePixels]_.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            the warped image, or if return info is True, the warped image and
            the info dictionary.

    TODO:
        - [ ] When dsize='positive' but the transform contains an axis flip,
              the width / height of the box will become negative.  Should we
              adjust for this?

    References:
        .. [WhereArePixels] https://ppwwyyxx.com/blog/2021/Where-are-Pixels/

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> #image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale(0.05)
        >>> transform = Affine.scale(0.02)
        >>> warped1 = kwimage.warp_affine(image, transform, dsize='positive', antialias=1, interpolation='nearest')
        >>> warped2 = kwimage.warp_affine(image, transform, dsize='positive', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> image = kwimage.grab_test_image('checkerboard')
        >>> transform = Affine.random() @ Affine.scale((.1, 1.2))
        >>> warped1 = kwimage.warp_affine(image, transform, dsize='positive', antialias=1)
        >>> warped2 = kwimage.warp_affine(image, transform, dsize='positive', antialias=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Test the case where the input data is empty or the target canvas
        >>> # is empty, this should be handled like boundary effects
        >>> import kwimage
        >>> import numpy as np
        >>> image = np.random.rand(1, 1, 3)
        >>> transform = kwimage.Affine.random()
        >>> result = kwimage.warp_affine(image, transform, dsize=(0, 0))
        >>> assert result.shape == (0, 0, 3)
        >>> #
        >>> empty_image = np.random.rand(0, 1, 3)
        >>> result = kwimage.warp_affine(empty_image, transform, dsize=(10, 10))
        >>> assert result.shape == (10, 10, 3)
        >>> #
        >>> empty_image = np.random.rand(0, 1, 3)
        >>> result = kwimage.warp_affine(empty_image, transform, dsize=(10, 0))
        >>> assert result.shape == (0, 10, 3)

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Demo difference between positive and content dsize
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro', dsize=(512, 512))
        >>> transform = Affine.coerce(offset=(-100, -50), scale=2, theta=0.1)
        >>> # When warping other images or geometry along with this image
        >>> # it is important to account for the modified transform when
        >>> # setting dsize='content'. If dsize='positive', the transform
        >>> # will remain unchanged wrt other aligned images / geometries.
        >>> poly = kwimage.Boxes([[350, 5, 130, 290]], 'xywh').to_polygons()[0]
        >>> # Apply the warping to the images
        >>> warped_pos, info_pos = kwimage.warp_affine(image, transform, dsize='positive', return_info=True)
        >>> warped_con, info_con = kwimage.warp_affine(image, transform, dsize='content', return_info=True)
        >>> assert info_pos['dsize'] == (919, 1072)
        >>> assert info_con['dsize'] == (1122, 1122)
        >>> assert info_pos['transform'] == transform
        >>> # Demo the correct and incorrect way to apply transforms
        >>> poly_pos = poly.warp(transform)
        >>> poly_con = poly.warp(info_con['transform'])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # show original
        >>> kwplot.imshow(image, pnum=(1, 3, 1), title='original')
        >>> poly.draw(color='green', alpha=0.5, border=True)
        >>> # show positive warped
        >>> kwplot.imshow(warped_pos, pnum=(1, 3, 2), title='dsize=positive')
        >>> poly_pos.draw(color='purple', alpha=0.5, border=True)
        >>> # show content warped
        >>> ax = kwplot.imshow(warped_con, pnum=(1, 3, 3), title='dsize=content')[1]
        >>> poly_con.draw(color='dodgerblue', alpha=0.5, border=True)   # correct
        >>> poly_pos.draw(color='orangered', alpha=0.5, border=True)  # incorrect
        >>> cc = poly_con.to_shapely().centroid
        >>> cp = poly_pos.to_shapely().centroid
        >>> ax.text(cc.x, cc.y + 250, 'correctly transformed', color='dodgerblue',
        >>>         backgroundcolor=(0, 0, 0, 0.7), horizontalalignment='center')
        >>> ax.text(cp.x, cp.y - 250, 'incorrectly transformed', color='orangered',
        >>>         backgroundcolor=(0, 0, 0, 0.7), horizontalalignment='center')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Demo piecewise transform
        >>> import kwimage
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('pm5644')
        >>> transform = Affine.coerce(offset=(-100, -50), scale=2, theta=0.1)
        >>> warped_piecewise, info = kwimage.warp_affine(image, transform, dsize='positive', return_info=True, large_warp_dim=32)
        >>> warped_normal, info = kwimage.warp_affine(image, transform, dsize='positive', return_info=True, large_warp_dim=None)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image, pnum=(1, 3, 1), title='original')
        >>> kwplot.imshow(warped_normal, pnum=(1, 3, 2), title='normal warp')
        >>> kwplot.imshow(warped_piecewise, pnum=(1, 3, 3), title='piecewise warp')

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import numpy as np
        >>> # TODO: Explain why the bottom left is interpolated with 0's
        >>> # And not 2s, probably has to do with interpretation of pixels
        >>> # as points and not areas.
        >>> image = np.full((6, 6), fill_value=3, dtype=np.uint8)
        >>> transform = kwimage.Affine.eye()
        >>> transform = kwimage.Affine.coerce(offset=.5) @ transform
        >>> transform = kwimage.Affine.coerce(scale=2) @ transform
        >>> warped = kwimage.warp_affine(image, transform, dsize=(12, 12))

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Demo how nans are handled
        >>> import kwimage
        >>> import numpy as np
        >>> image = kwimage.grab_test_image('pm5644')
        >>> image = kwimage.ensure_float01(image)
        >>> image[100:300, 400:700] = np.nan
        >>> transform = kwimage.Affine.coerce(scale=0.05, offset=10.5, theta=0.3, shearx=0.2)
        >>> warped1 = kwimage.warp_affine(image, transform, dsize='positive', antialias=1, interpolation='linear', border_value=0)
        >>> warped2 = kwimage.warp_affine(image, transform, dsize='positive', antialias=0, border_value=np.nan)
        >>> assert np.isnan(warped1).any()
        >>> assert np.isnan(warped2).any()
        >>> assert warped1[np.isnan(warped1).any(axis=2)].all()
        >>> assert warped2[np.isnan(warped2).any(axis=2)].all()
        >>> print('warped1.shape = {!r}'.format(warped1.shape))
        >>> print('warped2.shape = {!r}'.format(warped2.shape))
        >>> assert warped2.shape == warped1.shape
        >>> warped2[np.isnan(warped2).any(axis=2)]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=3)
        >>> image_canvas = kwimage.fill_nans_with_checkers(image)
        >>> warped1_canvas = kwimage.fill_nans_with_checkers(warped1)
        >>> warped2_canvas = kwimage.fill_nans_with_checkers(warped2)
        >>> kwplot.imshow(image_canvas, pnum=pnum_(), title='original')
        >>> kwplot.imshow(warped1_canvas, pnum=pnum_(), title='antialias=True, border=0')
        >>> kwplot.imshow(warped2_canvas, pnum=pnum_(), title='antialias=False, border=nan')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Demo how of how we also handle masked arrays
        >>> import kwimage
        >>> import numpy as np
        >>> _image = kwimage.grab_test_image('pm5644')
        >>> _image = kwimage.ensure_float01(_image)
        >>> _image[100:200, 400:700] = np.nan
        >>> mask = np.isnan(_image)
        >>> data = np.nan_to_num(_image)
        >>> image = np.ma.MaskedArray(data=data, mask=mask)
        >>> transform = kwimage.Affine.coerce(scale=0.05, offset=10.5, theta=0.3, shearx=0.2)
        >>> warped1 = kwimage.warp_affine(image, transform, dsize='positive', antialias=1, interpolation='linear')
        >>> assert isinstance(warped1, np.ma.MaskedArray)
        >>> warped2 = kwimage.warp_affine(image, transform, dsize='positive', antialias=0)
        >>> print('warped1.shape = {!r}'.format(warped1.shape))
        >>> print('warped2.shape = {!r}'.format(warped2.shape))
        >>> assert warped2.shape == warped1.shape
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nRows=1, nCols=2)
        >>> kwplot.imshow(warped1, pnum=pnum_(), title='antialias=True')
        >>> kwplot.imshow(warped2, pnum=pnum_(), title='antialias=False')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import ubelt as ub
        >>> import numpy as np
        >>> #image = kwimage.checkerboard(dsize=(8, 8), num_squares=4, on_value='kitware_blue', off_value='kitware_green')
        >>> image = kwimage.checkerboard(dsize=(32, 32), num_squares=4, on_value='kitware_blue', off_value='kitware_green')
        >>> #image = kwimage.checkerboard(dsize=(4, 4), num_squares=4, on_value='kitware_blue', off_value='kitware_green')
        >>> grid = list(ub.named_product({
        >>>     'border_value': [
        >>>         0,
        >>>         np.nan,
        >>>         #'replicate'
        >>>    ],
        >>>     'interpolation': [
        >>>         'nearest',
        >>>         'linear',
        >>>         #'cubic',
        >>>         #'lanczos',
        >>>         #'area'
        >>>     ],
        >>>     'origin_convention': ['center', 'corner'],
        >>> }))
        >>> results = []
        >>> results += [('input', image)]
        >>> for kwargs in grid:
        >>>     warped_image = kwimage.warp_affine(image, {'scale': 0.5}, dsize='auto', **kwargs)
        >>>     title = ub.urepr(kwargs, compact=1, nl=1)
        >>>     results.append((title, warped_image))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for title, canvas in results:
        >>>     canvas = kwimage.fill_nans_with_checkers(canvas).clip(0, 1)
        >>>     kwplot.imshow(canvas, pnum=pnum_(), origin_convention='corner', title=title, show_ticks=True)
        >>> fig = kwplot.plt.gcf()
        >>> fig.subplots_adjust(hspace=0.37, wspace=0.25)
        >>> fig.set_size_inches([18.11, 11.07])
        >>> kwplot.show_if_requested()
    """
    if backend == 'auto':
        backend = _default_backend()

    if backend == 'cv2':
        from kwimage.im_cv2 import _cv2_warp_affine
        return _cv2_warp_affine(
            image=image,
            transform=transform,
            dsize=dsize,
            antialias=antialias,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
            large_warp_dim=large_warp_dim,
            origin_convention=origin_convention,
            return_info=return_info,
        )
    elif backend == 'itk':
        from kwimage.im_itk import _itk_warp_affine
        if large_warp_dim is not None:
            raise ValueError('itk backend does not support large_warp_dim')
        return _itk_warp_affine(
            image=image,
            transform=transform,
            dsize=dsize,
            antialias=antialias,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
            origin_convention=origin_convention,
            return_info=return_info,
        )
    elif backend == 'torch':
        raise NotImplementedError(f'no kwimage backend={backend} for warp_affine')
    elif backend == 'skimage':
        raise NotImplementedError(f'no kwimage backend={backend} for warp_affine')
    else:
        raise KeyError(f'no kwimage backend={backend} for warp_affine')


def imresize(img, scale=None, dsize=None, max_dim=None, min_dim=None,
             interpolation=None, grow_interpolation=None, letterbox=False,
             return_info=False, antialias=False, border_value=0, backend='auto'):
    """
    Resize an image via a scale factor, final size, or size and aspect ratio.

    Wraps and generalizes cv2.resize, allows for specification of either a
    scale factor, a final size, or the final size for a particular dimension.

    Note:
        As described in [ResizeConfusion]_, this each entry in the image array
        as representing the center of a pixel. This is the pixels_are='area'
        approach, or align_corners=False in pytorch. It is equivalent to a
        shift and scale in warp_affine (which by default uses align corners).

    Note:
        The border mode cannot be specified here and seems to always be reflect
        in the underlying cv2 implementation.

    Args:
        img (ndarray): image to resize

        scale (float | Tuple[float, float]):
            Desired floating point scale factor. If a tuple, the dimension
            ordering is x,y.  Mutually exclusive with dsize, min_dim, max_dim.

        dsize (Tuple[int | None, int | None] | None):
            The desired width and height of the new image. If a dimension is
            None, then it is automatically computed to preserve aspect ratio.
            Mutually exclusive with scale, min_dim, max_dim.

        max_dim (int):
            New size of the maximum dimension, the other dimension is scaled to
            maintain aspect ratio. Mutually exclusive with scale, dsize,
            min_dim.

        min_dim (int):
            New size of the minimum dimension, the other dimension is scaled to
            maintain aspect ratio. Mutually exclusive with scale, dsize,
            max_dim.

        interpolation (str | int):
            The interpolation key or code (e.g. linear lanczos). By default
            "area" is used if the image is shrinking and "lanczos" is used if
            the image is growing. Note, if this is explicitly set, then it will
            be used regardless of if the image is growing or shrinking. Set
            ``grow_interpolation`` to change the default for an enlarging
            interpolation.

        grow_interpolation (str | int):
            The interpolation key or code to use when the image is being
            enlarged. Does nothing if "interpolation" is explicitly given. If
            "interpolation" is not specified "area" is used when shrinking.
            Defaults to "lanczos".

        letterbox (bool):
            If used in conjunction with dsize, then the image is scaled and
            translated to fit in the center of the new image while maintaining
            aspect ratio. Border padding is added if necessary.  Defaults to
            False.

        return_info (bool):
            if True returns information about the final transformation in a
            dictionary. If there is an offset, the scale is applied before the
            offset when transforming to the new resized space.  Defaults to
            False.

        antialias (bool):
            if True blurs to anti-alias before downsampling.
            Defaults to False.

        border_value (int | float | Iterable[int | float]):
            if letterbox is True, this is used as the constant fill value.

    Returns:
        ndarray | Tuple[ndarray, Dict] :
            the new image and optionally an info dictionary if
            `return_info=True`

    References:
        .. [ResizeConfusion] https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import numpy as np
        >>> # Test scale
        >>> img = np.zeros((16, 10, 3), dtype=np.uint8)
        >>> new_img, info = kwimage.imresize(img, scale=.85,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [.8, 0.875]
        >>> # Test dsize without None
        >>> new_img, info = kwimage.imresize(img, dsize=(5, 12),
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.5 , 0.75]
        >>> # Test dsize with None
        >>> new_img, info = kwimage.imresize(img, dsize=(6, None),
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.6, 0.625]
        >>> # Test max_dim
        >>> new_img, info = kwimage.imresize(img, max_dim=6,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.4  , 0.375]
        >>> # Test min_dim
        >>> new_img, info = kwimage.imresize(img, min_dim=6,
        >>>                                  interpolation='area',
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['scale'].tolist() == [0.6  , 0.625]

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> import kwimage
        >>> import numpy as np
        >>> # Test letterbox resize
        >>> img = np.ones((5, 10, 3), dtype=np.float32)
        >>> new_img, info = kwimage.imresize(img, dsize=(19, 19),
        >>>                                  letterbox=True,
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['offset'].tolist() == [0, 4]
        >>> img = np.ones((10, 5, 3), dtype=np.float32)
        >>> new_img, info = kwimage.imresize(img, dsize=(19, 19),
        >>>                                  letterbox=True,
        >>>                                  return_info=True)
        >>> print('info = {!r}'.format(info))
        >>> assert info['offset'].tolist() == [4, 0]

        >>> import kwimage
        >>> import numpy as np
        >>> # Test letterbox resize
        >>> img = np.random.rand(100, 200)
        >>> new_img, info = kwimage.imresize(img, dsize=(300, 300), letterbox=True, return_info=True)

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Check aliasing
        >>> import numpy as np
        >>> import kwimage
        >>> #img = kwimage.grab_test_image('checkerboard')
        >>> img = kwimage.grab_test_image('pm5644')
        >>> # test with nans
        >>> img = kwimage.ensure_float01(img)
        >>> img[100:200, 400:700] = np.nan
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dsize = (14, 14)
        >>> dsize = (64, 64)
        >>> # When we set "grow_interpolation" for a "shrinking" resize it should
        >>> # still do the "area" interpolation to antialias the results. But if we
        >>> # use explicit interpolation it should alias.
        >>> pnum_ = kwplot.PlotNums(nSubplots=12, nCols=4)
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True,  interpolation='area'), pnum=pnum_(), title='resize aa area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='linear'), pnum=pnum_(), title='resize aa linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='nearest'), pnum=pnum_(), title='resize aa nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, interpolation='cubic'), pnum=pnum_(), title='resize aa cubic')

        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='area'), pnum=pnum_(), title='resize aa grow area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='linear'), pnum=pnum_(), title='resize aa grow linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='nearest'), pnum=pnum_(), title='resize aa grow nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=True, grow_interpolation='cubic'), pnum=pnum_(), title='resize aa grow cubic')

        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='area'), pnum=pnum_(), title='resize no-aa area')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='linear'), pnum=pnum_(), title='resize no-aa linear')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='nearest'), pnum=pnum_(), title='resize no-aa nearest')
        >>> kwplot.imshow(kwimage.imresize(img, dsize=dsize, antialias=False, interpolation='cubic'), pnum=pnum_(), title='resize no-aa cubic')

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # Test single pixel resize
        >>> import kwimage
        >>> import numpy as np
        >>> assert kwimage.imresize(np.random.rand(1, 1, 3), scale=3).shape == (3, 3, 3)
        >>> assert kwimage.imresize(np.random.rand(1, 1), scale=3).shape == (3, 3)

        # cv2.resize(np.random.rand(1, 1, 3), (3, 3))


    TODO:
        - [X] When interpolation is area and the number of channels > 4 cv2.resize will error but it is fine for linear interpolation
        - [ ] TODO: add padding options when letterbox=True

        - [ ] Allow for pre-clipping when letterbox=True
    """
    if backend == 'auto':
        backend = _default_backend()

    if backend == 'cv2':
        from kwimage.im_cv2 import _cv2_imresize
        return _cv2_imresize(
            img=img, scale=scale, dsize=dsize, max_dim=max_dim, min_dim=min_dim,
            interpolation=interpolation, grow_interpolation=grow_interpolation,
            letterbox=letterbox, return_info=return_info, antialias=antialias,
        )
    elif backend == 'skimage':
        return _skimage_resize(
            img=img, scale=scale, dsize=dsize, max_dim=max_dim, min_dim=min_dim,
            interpolation=interpolation, grow_interpolation=grow_interpolation,
            letterbox=letterbox, return_info=return_info, antialias=antialias,
        )
    elif backend == 'torch':
        raise NotImplementedError(backend)
    else:
        raise KeyError(backend)


def _skimage_resize(img, scale=None, dsize=None, max_dim=None, min_dim=None,
                    interpolation=None, grow_interpolation=None, letterbox=False,
                    return_info=False, antialias=False, border_value=0):
    """
    Example:
        >>> import numpy as np
        >>> from kwimage.im_transform import _skimage_resize
        >>> img = np.zeros((16, 10, 3), dtype=np.uint8)
        >>> new_img = _skimage_resize(img, dsize=(20, 21))
        >>> assert new_img.shape[0:2] == (21, 20)
    """
    from skimage.transform import resize
    not_implemented_params = [
        letterbox, min_dim, max_dim, scale, grow_interpolation,
        return_info, antialias
    ]
    for p in not_implemented_params:
        if p:
            raise NotImplementedError('skimage imresize params')

    order = _coerce_skimage_interpolation_order(interpolation)

    new_w, new_h = dsize
    output_shape = (new_h, new_w)
    new_img = resize(img, output_shape=output_shape, order=order)
    return new_img


def _coerce_skimage_interpolation_order(interpolation):
    """
    The order of interpolation. The order has to be in the range 0-5:
     - 0: Nearest-neighbor
     - 1: Bi-linear (default)
     - 2: Bi-quadratic
     - 3: Bi-cubic
     - 4: Bi-quartic
     - 5: Bi-quintic
    """
    if interpolation is None:
        interpolation = 'linear'   # TODO: make consistent with the cv2 default interpolation, which is currently lanczos

    if interpolation == 'nearest':
        order = 0
    elif interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    else:
        raise NotImplementedError(interpolation)
    return order
