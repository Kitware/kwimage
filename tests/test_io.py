
def test_interlaced():
    """
    sudo apt install imagemagick
    """
    import ubelt as ub
    import cv2
    import kwimage
    from os.path import join

    import pytest
    pytest.skip('This is an exploration of an issue, not a test that we should run')

    # https://stackoverflow.com/questions/19742548/how-to-de-interlace-png-files
    convert_ext = ub.find_exe('convert')

    if not convert_ext:
        import pytest
        pytest.skip('need image magick for this test')

    dpath = ub.ensure_app_cache_dir('kwimage/tests/io')
    data = kwimage.grab_test_image('astro')

    orig_fpath = join(dpath, 'orig.png')
    kwimage.imwrite(orig_fpath, data)

    interlace_types = [
        'None',
        'Line',
        'Plane',
        'GIF',
        'PNG',
        'JPEG',
        'Partition',
    ]

    for interlace_type in interlace_types:

        interlaced_fpath = join(dpath, 'interlaced_{}.png'.format(interlace_type))

        info = ub.cmd(
            'convert {orig_fpath} -interlace {interlace_type} {interlaced_fpath}'.format(**locals()),
            cwd=dpath, verbose=2, check=1)

        ret1 = kwimage.imread(interlaced_fpath)
        ret2 = cv2.imread(interlaced_fpath, cv2.IMREAD_UNCHANGED)
        assert ret1 is not None
        assert ret2 is not None
        frac1 = (ret1 == data).sum() / data.size
        frac2 = (ret2 == data).sum() / data.size
        print('frac1 = {!r}'.format(frac1))
        print('frac2 = {!r}'.format(frac2))

        if 0:
            import kwplot
            kwplot.autompl()
            kwplot.imshow(data, pnum=(1, 3, 1))
            kwplot.imshow(ret1, pnum=(1, 3, 2))
            kwplot.imshow(ret1, pnum=(1, 3, 3))

    # # TODO: how to write in interlaced mode?
    # imwrite_params = {
    #     cv2.IMWRITE_PAM_TUPLETYPE: cv2.IMWRITE_PAM_FORMAT_NULL,
    # }
    # params = tuple(ub.flatten(imwrite_params.items()))
    # cv2.imwrite(fpath, data, params=params)
    # ret = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    # assert ret is not None
    # assert np.all(ret == data)


def test_cross_backend_reads():
    import pytest
    pytest.skip('This is a demonstration of an issue, not a test that we should run yet')

    import kwimage
    import ubelt as ub
    from kwimage import im_io
    from os.path import join
    import numpy as np

    dpath = ub.ensure_app_cache_dir('kwimage/tests/io')
    if not im_io._have_gdal():
        import pytest
        pytest.skip()

    data = np.random.rand(128, 128, 13)

    skim_fpath = join(dpath, 'written_skimage.tif')
    gdal_fpath = join(dpath, 'written_gdal.tif')

    kwimage.imwrite(skim_fpath, data, backend='skimage')
    kwimage.imwrite(gdal_fpath, data, backend='gdal')

    results = {}
    results['recon_skim_with_gdal'] = kwimage.imread(skim_fpath, backend='gdal')
    results['recon_gdal_with_gdal'] = kwimage.imread(gdal_fpath, backend='gdal')
    results['recon_skim_with_skim'] = kwimage.imread(skim_fpath, backend='skimage')
    results['recon_gdal_with_skim'] = kwimage.imread(gdal_fpath, backend='skimage')

    shapes = ub.map_vals(lambda x: x.shape, results)
    print('shapes = {}'.format(ub.repr2(shapes, nl=1)))
