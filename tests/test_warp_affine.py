
def test_warp_affine():
    # Demo how of how we also handle masked arrays
    import kwimage
    import numpy as np
    _image = kwimage.grab_test_image('pm5644')
    _image = kwimage.ensure_float01(_image)
    _image[100:200, 400:700] = np.nan
    mask = np.isnan(_image)
    data = np.nan_to_num(_image)
    image = np.ma.MaskedArray(data=data, mask=mask)
    transform = kwimage.Affine.coerce(scale=0.05, offset=10.5, theta=0.3, shearx=0.2)
    warped1 = kwimage.warp_affine(image, transform, dsize='positive', antialias=1, interpolation='linear')
    assert isinstance(warped1, np.ma.MaskedArray)
    warped2 = kwimage.warp_affine(image, transform, dsize='positive', antialias=0)
    print('warped1.shape = {!r}'.format(warped1.shape))
    print('warped2.shape = {!r}'.format(warped2.shape))
    assert warped2.shape == warped1.shape

    if 0:
        # xdoctest: +REQUIRES(--show)
        import kwplot
        kwplot.autompl()
        pnum_ = kwplot.PlotNums(nRows=1, nCols=2)

        canvas1 = kwimage.nodata_checkerboard(warped1, square_shape=1)
        canvas2 = kwimage.nodata_checkerboard(warped2, square_shape=1)
        kwplot.imshow(canvas1, pnum=pnum_(), title='antialias=True')
        kwplot.imshow(canvas2, pnum=pnum_(), title='antialias=False')
        kwplot.show_if_requested()


def test_warp_affine_with_nan_border():
    import kwimage
    import numpy as np
    img = kwimage.ensure_float01(kwimage.grab_test_image())
    M = kwimage.Affine.affine(theta=np.pi / 8)

    # TODO: ported to kwarray, use that later
    def equal_with_nan(a1, a2):
        """
        Numpy has array_equal with ``equal_nan=True``, but this is elementwise

        Args:
            a1 (ArrayLike): input array
            a2 (ArrayLike): input array
        """
        a1, a2 = np.asarray(a1), np.asarray(a2)
        a1nan, a2nan = np.isnan(a1), np.isnan(a2)
        nan_sameness = a1nan == a2nan
        value_sameness = (a1 == a2)
        # If they are actually the same, they should be value same xor nansame.
        flags = value_sameness ^ nan_sameness
        return flags

    # Explicit tuple should be for each channel
    border_value = (np.nan, 0, np.nan)
    warped = kwimage.warp_affine(img, M, border_value=border_value)
    nanned_pixels = warped[np.isnan(warped).any(axis=2)]
    assert nanned_pixels.size
    assert not warped[np.isnan(warped).any(axis=2)].all()
    assert equal_with_nan(nanned_pixels, border_value).all(axis=1).any()

    border_value = (np.nan, 0, 0)
    warped = kwimage.warp_affine(img, M, border_value=border_value)
    nanned_pixels = warped[np.isnan(warped).any(axis=2)]
    assert nanned_pixels.size
    assert not warped[np.isnan(warped).any(axis=2)].all()
    assert equal_with_nan(nanned_pixels, border_value).all(axis=1).any()

    border_value = np.nan
    warped = kwimage.warp_affine(img, M, border_value=border_value)
    nanned_pixels = warped[np.isnan(warped).any(axis=2)]
    assert nanned_pixels.size
    assert warped[np.isnan(warped).any(axis=2)].all()

    # Case with a scalar and a 2D image.
    border_value = np.nan
    img0 = img[:, :, 0]
    warped = kwimage.warp_affine(img0, M, border_value=border_value)
    nanned_pixels = warped[np.isnan(warped)]
    assert nanned_pixels.size


def test_warp_affine_with_many_chans():
    import kwimage
    import numpy as np
    img = np.random.rand(5, 5, 4)
    M = kwimage.Affine.affine(theta=np.pi / 8)

    img = np.random.rand(8, 8, 5)
    M = kwimage.Affine.affine(theta=np.pi / 8)
    warped = kwimage.warp_affine(img, M, border_value=np.nan)
