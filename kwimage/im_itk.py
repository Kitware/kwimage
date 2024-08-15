

def warp_affine_itk(image, transform, dsize=None, antialias=False,
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

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(module:itk)
        >>> from kwimage.im_itk import *  # NOQA
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
        >>> result1 = warp_affine_itk(image, transform, interpolation=interpolation, origin_convention='corner')
        >>> result2 = kwimage.warp_affine(image, transform, interpolation=interpolation, origin_convention='corner')
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
    """
    import itk
    import numpy as np
    if 0:
        [n for n in dir(itk) if 'resample' in n.lower()]
        [n for n in dir(itk) if 'affine' in n.lower()]
        [n for n in dir(itk) if '_from_' in n.lower()]
        [n for n in dir(itk) if 'type' in n.lower()]
        [n for n in dir(itk) if 'RGB' in n.upper()]
        [n for n in dir(itk) if 'UC' in n.upper()]

    numpy_matrix = np.asarray(transform)

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

    if dsize is not None:
        raise NotImplementedError('TODO')

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
        raise NotImplementedError('TODO: implement antialiasing')

    # Execute transformation
    resampled = itk.resample_image_filter(
        itk_image,
        transform=itk_transform.GetInverseTransform(),
        interpolator=itk_interpolator,
        size=output_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
    )

    # Convert back to numpy
    result = itk.array_from_image(resampled)
    return result
