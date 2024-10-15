

def _itk_warp_affine(image, transform, dsize=None, antialias=False,
                     interpolation='linear', border_mode=None, border_value=0,
                     origin_convention='center', return_info=False):
    """
    ITK backend for warp affine

    References:
        https://examples.itk.org/src/filtering/imagegrid/resampleanimage/documentation
        https://examples.itk.org/src/core/transform/translateimage/documentation
        https://github.com/SimpleITK/TUTORIAL/blob/main/01_spatial_transformations.ipynb
        https://github.com/hinerm/ITK/blob/master/Wrapping/Generators/Python/Tests/ResampleImageFilter.py
        https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html

    TODO:
        What is the itk package we need to add as a dependency?

    CommandLine:
        xdoctest -m kwimage.im_itk _itk_warp_affine:0 --show
        xdoctest -m kwimage.im_itk _itk_warp_affine:1 --show

    Example:
        >>> # xdoctest: +REQUIRES(module:itk)
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> from kwimage.im_itk import *  # NOQA
        >>> from kwimage.im_itk import _itk_warp_affine
        >>> import numpy as np
        >>> import kwarray
        >>> import kwarray.distributions
        >>> import kwimage
        >>> rng = kwarray.ensure_rng(None)
        >>> image = kwimage.grab_test_image('astro')
        >>> #image = kwimage.ensure_float01(image)
        >>> transform = kwimage.Affine.random(offset=rng.randint(-100, 100))
        >>> # Compare Warp Methods
        >>> #interpolation = 'nearest'
        >>> interpolation = 'linear'
        >>> result1 = kwimage.warp_affine(image, transform, interpolation=interpolation, origin_convention='corner', backend='itk')
        >>> result2 = kwimage.warp_affine(image, transform, interpolation=interpolation, origin_convention='corner', backend='cv2')
        >>> # Test Results
        >>> import kwarray
        >>> print(kwarray.stats_dict(result1))
        >>> print(kwarray.stats_dict(result2))
        >>> result1 = kwimage.ensure_float01(result1)
        >>> result2 = kwimage.ensure_float01(result2)
        >>> difference = (result1 - result2)
        >>> abs_difference = np.abs(difference)
        >>> difference = kwarray.normalize(abs_difference)
        >>> #
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image, pnum=(1, 4, 1), title='input image', fnum=1, doclf=1)
        >>> kwplot.imshow(result1, pnum=(1, 4, 2), title='itk transform')
        >>> kwplot.imshow(result2, pnum=(1, 4, 3), title='cv2 transform')
        >>> kwplot.imshow(abs_difference, pnum=(1, 4, 4), title='difference')
        >>> #kwplot.imshow((abs_difference > 0).astype(np.float32), pnum=(1, 4, 4), title='difference')
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +SKIP(todo)
        >>> # xdoctest: +REQUIRES(module:itk)
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
        >>>     'backend': ['itk'],
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

    Example:
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> # xdoctest: +REQUIRES(module:itk)
        >>> import kwimage
        >>> from kwimage.im_itk import _itk_warp_affine
        >>> from kwimage.transform import Affine
        >>> image = kwimage.grab_test_image('astro')
        >>> transform = Affine.random() @ Affine.translate(300)
        >>> from kwimage.im_cv2 import _cv2_warp_affine
        >>> warped = _itk_warp_affine(image, transform, dsize='auto', interpolation='nearest')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(warped)
        >>> kwplot.show_if_requested()
    """
    import itk
    import numpy as np
    from kwimage._common import _coerce_warp_dsize_inputs
    if 0:
        [n for n in dir(itk) if 'resample' in n.lower()]
        [n for n in dir(itk) if 'affine' in n.lower()]
        [n for n in dir(itk) if '_from_' in n.lower()]
        [n for n in dir(itk) if 'type' in n.lower()]
        [n for n in dir(itk) if 'RGB' in n.upper()]
        [n for n in dir(itk) if 'UC' in n.upper()]

    # ttype = itk.Image[ itk.UC,  2 ]
    # itk_image = itk.image_from_array(image.T, ttype=ttype)
    itk_image = itk.image_from_array(image.T, is_vector=True)

    # TODO: handle origin convention
    if origin_convention == 'center':
        ...
    elif origin_convention == 'corner':
        itk_image.SetOrigin([0.5, 0.5])
    else:
        raise KeyError(origin_convention)

    # ITK is explcit about image coordinate assumptions.
    output_size = itk.size(itk_image)
    output_spacing = itk.spacing(itk_image)
    output_origin = itk.origin(itk_image)

    h, w = image.shape[0:2]
    input_dsize = (w, h)
    require_warped_info = False
    dsize_inputs = _coerce_warp_dsize_inputs(
        dsize=dsize,
        input_dsize=input_dsize,
        transform=transform,
        require_warped_info=require_warped_info
    )
    # max_dsize = dsize_inputs['max_dsize']
    # new_origin = dsize_inputs['new_origin']
    transform_ = dsize_inputs['transform']
    dsize = dsize_inputs['dsize']
    info = {
        'transform': transform_,
        'dsize': dsize,
        'antialias_info': None,
    }

    numpy_matrix = np.asarray(transform_)
    output_size = dsize

    # Is the above or this perferred?
    # itk_image.UpdateOutputInformation()
    # output_size = itk_image.GetLargestPossibleRegion().GetSize()
    # output_spacing = itk_image.GetSpacing()
    # output_origin = itk_image.GetOrigin()

    # Setup the ITK affine matrix
    itk_transform = itk.AffineTransform.D2.New()
    numpy_matrix = numpy_matrix.astype(np.float64)
    mat_2x2 = numpy_matrix[0:2, 0:2]
    itk_matrix = itk.matrix_from_array(mat_2x2)
    itk_transform.SetMatrix(itk_matrix)
    offset = numpy_matrix[0:2, 2]
    itk_transform.SetTranslation(offset)

    # Setup ITK interpolation
    if interpolation == 'nearest':
        itk_interpolator = itk.NearestNeighborInterpolateImageFunction.New(itk_image)
    elif interpolation == 'linear':
        itk_interpolator = itk.LinearInterpolateImageFunction.New(itk_image)
    else:
        raise NotImplementedError(interpolation)

    if antialias:
        affine_params = transform_.decompose()
        sx, sy = affine_params['scale']
        if sx < 1 or sy < 1:
            # Requires antialias
            raise NotImplementedError(
                'TODO: need to implement antialiasing in ITK backend')
            # AntiAliasFilterType = itk.AntiAliasBinaryImageFilter[ImageType, ImageType]
            # antialiasfilter = AntiAliasFilterType.New()
            # antialiasfilter.SetInput(reader.GetOutput())
            # antialiasfilter.SetMaximumRMSError(args.maximum_RMS_error)
            # antialiasfilter.SetNumberOfIterations(args.number_of_iterations)
            # antialiasfilter.SetNumberOfLayers(args.number_of_layers)

    # Execute transformation
    if 0:
        resampled = itk.resample_image_filter(
            itk_image,
            transform=itk_transform.GetInverseTransform(),
            interpolator=itk_interpolator,
            size=output_size,
            output_spacing=output_spacing,
            output_origin=output_origin,
        )
    else:
        resample_filter = itk.ResampleImageFilter.New(
            itk_image,
            transform=itk_transform.GetInverseTransform(),
            interpolator=itk_interpolator,
            size=output_size,
            output_spacing=output_spacing,
            output_origin=output_origin,
            # default_pixel_value=...,
            # output_direction=...,
            # output_start_index=...,
            # reference_image=...,
            # use_reference_image=...,
        )
        resample_filter.Update()
        resampled = resample_filter.GetOutput()

    # Convert back to numpy
    result = itk.array_from_image(resampled)

    if return_info:
        return result, info
    else:
        return result
