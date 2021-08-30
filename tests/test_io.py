
def test_interlaced():
    """
    sudo apt install imagemagick
    """
    import numpy as np
    import ubelt as ub
    import cv2
    import kwimage
    from os.path import join

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
