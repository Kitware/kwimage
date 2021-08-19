def test_demodata_headers():
    """
    Check that the extensions on all of our demo images are accurate
    """
    from PIL import Image
    from os.path import splitext
    import kwimage
    import ubelt as ub

    if not ub.argflag('--network'):
        import pytest
        pytest.skip('requires network')

    for key in kwimage.grab_test_image.keys():
        fpath = kwimage.grab_test_image_fpath(key)
        img = Image.open(fpath)
        try:
            header_format = img.format
            ext_format = splitext(fpath)[-1].upper()[1:]

            if ext_format == 'JPG':
                ext_format = 'JPEG'

            if ext_format != header_format:
                print(img.format)
                print('fpath = {!r}'.format(fpath))
                print('--')

            assert ext_format == header_format, 'format should agree with ext'
        finally:
            img.close()
