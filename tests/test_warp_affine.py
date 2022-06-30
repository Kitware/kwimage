
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
