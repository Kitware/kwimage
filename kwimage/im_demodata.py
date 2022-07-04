"""
Keep a manifest of demo images used for testing
"""
import ubelt as ub


_TEST_IMAGES = {
    'airport': {
        'fname': 'airport.jpg',
        'sha256': 'bff5f9212d5c77dd47f2b80e5dc1b4409fa7813b08fc39b504294497b3483ffc',
        # seems to hang (2021-08-12), reuploaded to data.kitware.com
        # 'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b6/Fires_and_Deforestation_on_the_Amazon_Frontier%2C_Rondonia%2C_Brazil_-_August_12%2C_2007.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
    },
    'amazon': {
        'fname': 'amazon.jpg',
        'sha256': 'ef352b60f2577692ab3e9da19d09a49fa9da9937f892afc48094988a17c32dc3',
        'url': 'https://data.kitware.com/api/v1/file/611e9f4b2fa25629b9dc0ca2/download',
    },
    # https://en.wikipedia.org/wiki/Eileen_Collins
    'astro': {
        'fname': 'astro.png',
        'sha256': '9f2b4671e868fd51451f03809a694006425eee64ad472f7065da04079be60c53',
        'url': 'https://i.imgur.com/KXhKM72.png',
    },
    'carl': {
        'fname': 'carl.jpg',
        'sha256': '595056e142951bbdc19d79009cb443e29e8a0148597629dbd16fbd7207063f20',
        'url': 'https://i.imgur.com/flTHWFD.png',   # imgur thinks this is a PNG for some reason
    },
    'lowcontrast': {
        'fname': 'lowcontrast.jpg',
        'sha256': '532572a245d2b401583488ccf9f033e5960dc9f3f56b8ca6b933a8986ec9e95e',
        'url': 'https://i.imgur.com/dyC68Bi.jpg',
    },
    'paraview': {
        'fname': 'paraview.png',
        'sha256': '859423aefce1037b2b6959b74d7b137a4104acd6db95a9247abb26c2d0aa93b8',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
    },
    'parrot': {
        'fname': 'parrot.png',
        'sha256': 'fadd4cdddc46e43185999421dcb1ae9d3ba6d13b5b6d0acc05268fc7246f3e59',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
    },
    'stars': {
        'fname': 'stars.png',
        'sha256': '36391b4d36b4b5e2597c53f9465951910542fbec82f5a0213715759d1de9714f',
        'url': 'https://i.imgur.com/kCi7C1r.png',
    },
    'pm5644': {
        # Test pattern
        'fname': 'Philips_Pattern_pm5644.png',
        'sha256': 'eba500341492649d4fa4e83b5200abbffa6673de5de4c20ed669dedeb00d3941',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Philips_Pattern_pm5644.png'
    },
    'tsukuba_l': {
        'fname': 'tsukuba_l.png',
        'sha256': 'e29144841f4e2200e88eb6ad928cfa3ee0c55ccac0a28532c9293c4a5e0b284d',
        'url': 'https://i.imgur.com/DhIKgGx.png',
    },
    'tsukuba_r': {
        'fname': 'tsukuba_r.png',
        'sha256': 'fb4e8b1561c177a9aba23693bd576d0e06f5778b8d44e1c1cc5c5dd35d5fd1d4',
        'url': 'https://i.imgur.com/38RST9H.png',
    },
}


# Note: tsukuba images are mirrored from the following urls:
# 'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imL.png',
# 'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imR.png',


def _update_hashes():
    """
    for dev use to update hashes of the demo images

    CommandLine:
        xdoctest -m kwimage.im_demodata _update_hashes
        xdoctest -m kwimage.im_demodata _update_hashes --require-hashes
    """
    TEST_IMAGES = _TEST_IMAGES.copy()

    for key in TEST_IMAGES.keys():
        item = TEST_IMAGES[key]

        grabkw = {
            'appname': 'kwimage/demodata',
        }
        # item['sha512'] = 'not correct'

        # Wait until ubelt 9.1 is released to change hasher due to
        # issue in ub.grabdata
        # hasher_priority = ['sha512', 'sha1']
        hasher_priority = ['sha256']

        REQUIRE_EXISTING_HASH = ub.argflag('--require-hashes')
        if REQUIRE_EXISTING_HASH:
            for hasher in hasher_priority:
                if hasher in item:
                    grabkw.update({
                        'hash_prefix': item[hasher],
                        'hasher': hasher,
                    })
                    break

        if 'fname' in item:
            grabkw['fname'] = item['fname']

        item.pop('sha512', None)
        fpath = ub.grabdata(item['url'], **grabkw)
        if 'hasher' not in item:
            hasher = hasher_priority[0]
            prefix = ub.hash_file(fpath, hasher=hasher)
            item[hasher] = prefix[0:64]

        print('_TEST_IMAGES = ' + ub.repr2(TEST_IMAGES, nl=2))


def grab_test_image(key='astro', space='rgb', dsize=None,
                    interpolation='lanczos'):
    """
    Ensures that the test image exists (this might use the network), reads it
    and returns the the image pixels.

    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky
            airport - SkySat image of Beijing Capital International Airport on 18 February 2018
            See ``kwimage.grab_test_image.keys`` for a full list.

        space (str):
            which colorspace to return in. Defaults to 'rgb'

        dsize (Tuple[int, int]):
            if specified resizes image to this size

    Returns:
        ndarray: the requested image

    CommandLine:
        xdoctest -m kwimage.im_demodata grab_test_image

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import kwimage
        >>> for key in kwimage.grab_test_image.keys():
        >>>     print('attempt to grab key = {!r}'.format(key))
        >>>     kwimage.grab_test_image(key)
        >>>     print('grabbed key = {!r}'.format(key))
        >>> kwimage.grab_test_image('astro', dsize=(255, 255)).shape
        (255, 255, 3)
    """
    import kwimage
    # from kwimage import im_cv2
    if key == 'checkerboard':
        image = checkerboard()
    else:
        fpath = grab_test_image_fpath(key)
        image = kwimage.imread(fpath)
    if dsize:
        image = kwimage.imresize(image, dsize=dsize,
                                 interpolation=interpolation)
    return image


def grab_test_image_fpath(key='astro', dsize=None, overviews=None):
    """
    Ensures that the test image exists (this might use the network) and returns
    the cached filepath to the requested image.

    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky

        dsize (None | Tuple[int, int]):
            if specified, we will return a variant of the data with the
            specific dsize

        overviews (None | int):
            if specified, will return a variant of the data with overviews

    Returns:
        str: path to the requested image

    CommandLine:
        python -c "import kwimage; print(kwimage.grab_test_image_fpath('airport'))"

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import kwimage
        >>> for key in kwimage.grab_test_image.keys():
        ...     print('attempt to grab key = {!r}'.format(key))
        ...     kwimage.grab_test_image_fpath(key)
        ...     print('grabbed grab key = {!r}'.format(key))

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import kwimage
        >>> key = ub.peek(kwimage.grab_test_image.keys())
        >>> # specifying a dsize will construct a new image
        >>> fpath1 = kwimage.grab_test_image_fpath(key)
        >>> fpath2 = kwimage.grab_test_image_fpath(key, dsize=(32, 16))
        >>> print('fpath1 = {}'.format(ub.repr2(fpath1, nl=1)))
        >>> print('fpath2 = {}'.format(ub.repr2(fpath2, nl=1)))
        >>> assert fpath1 != fpath2
        >>> imdata2 = kwimage.imread(fpath2)
        >>> assert imdata2.shape[0:2] == (16, 32)
    """
    try:
        item = _TEST_IMAGES[key]
    except KeyError:
        valid_keys = sorted(_TEST_IMAGES.keys())
        raise KeyError(
            'Unknown key={!r}. Valid keys are {!r}'.format(
                key, valid_keys))
    if not isinstance(item, dict):
        item = {'url': item}

    grabkw = {
        'appname': 'kwimage/demodata',
    }
    hasher_priority = ['sha256']
    for hasher in hasher_priority:
        if hasher in item:
            grabkw.update({
                'hash_prefix': item[hasher],
                'hasher': hasher,
            })
            break
    if 'fname' in item:
        grabkw['fname'] = item['fname']

    fpath = ub.grabdata(item['url'], **grabkw)

    augment_params = {
        'dsize': dsize,
        'overviews': overviews,
    }
    for k, v in list(augment_params.items()):
        if v is None:
            augment_params.pop(k)

    if augment_params:
        import os
        stem_suffix = '_' + ub.repr2(augment_params, compact=True)

        ext = None
        if 'overviews' in augment_params:
            ext = '.tif'

        fpath_aug = ub.Path(ub.augpath(fpath, suffix=stem_suffix, ext=ext))

        # stamp = ub.CacheStamp.sidecar_for(fpath_aug, depends=[dsize])
        stamp = ub.CacheStamp(fpath_aug.name + '.stamp', dpath=fpath_aug.parent,
                              depends=augment_params, ext='.json')
        if stamp.expired():
            import kwimage

            imdata = kwimage.imread(fpath)

            if 'dsize' in augment_params:
                imdata = kwimage.imresize(
                    imdata, dsize=augment_params['dsize'])

            writekw = {}
            if 'overviews' in augment_params:
                writekw['overviews'] = augment_params['overviews']
                writekw['backend'] = 'gdal'

            kwimage.imwrite(fpath_aug, imdata, **writekw)
            stamp.renew()
        fpath = os.fspath(fpath_aug)

    return fpath

grab_test_image.keys = lambda: _TEST_IMAGES.keys()
grab_test_image_fpath.keys = lambda: _TEST_IMAGES.keys()


def checkerboard(num_squares='auto', square_shape='auto', dsize=(512, 512)):
    """
    Creates a checkerboard image

    Args:
        num_squares (int | str):
            Number of squares in a row. If 'auto' defaults to 8

        square_shape (int | Tuple[int, int] | str):
            If 'auto', chosen based on `num_squares`. Otherwise this is
            the height, width of each square in pixels.

        dsize (Tuple[int, int]): width and height

    References:
        https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Example:
        >>> from kwimage.im_demodata import *  # NOQA
        >>> img = checkerboard()
        >>> print(checkerboard(dsize=(16, 16)).shape)
        >>> print(checkerboard(num_squares=4, dsize=(16, 16)).shape)
        >>> print(checkerboard(square_shape=3, dsize=(23, 17)).shape)
        >>> print(checkerboard(square_shape=3, dsize=(1451, 1163)).shape)
        >>> print(checkerboard(square_shape=3, dsize=(1202, 956)).shape)
    """
    import numpy as np
    if num_squares == 'auto' and square_shape == 'auto':
        num_squares = 8

    want_w, want_h = dsize

    if square_shape != 'auto':
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]
        h, w = square_shape
        gen_h, gen_w = _next_multiple_of(want_h, h * 2), _next_multiple_of(want_w, w * 2)
    else:
        gen_h, gen_w = _next_multiple_of(want_h, 4), _next_multiple_of(want_w, 4)

    if num_squares == 'auto':
        assert square_shape != 'auto'
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]
        h, w = square_shape
        num_w = gen_w // w
        num_h = gen_h // h
        num_squares = num_h, num_w
    elif square_shape == 'auto':
        assert num_squares != 'auto'
        if not ub.iterable(num_squares):
            num_squares = [num_squares, num_squares]
        num_h, num_w = num_squares
        w = gen_w // num_w
        h = gen_h // num_h
        square_shape = (h, w)
    else:
        if not ub.iterable(num_squares):
            num_squares = [num_squares, num_squares]
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]

    num_h, num_w = num_squares

    num_pairs_w = int(num_w // 2)
    num_pairs_h = int(num_h // 2)
    # img_size = 512
    base = np.array([[1, 0] * num_pairs_w, [0, 1] * num_pairs_w] * num_pairs_h)
    expansion = np.ones((h, w))
    img = np.kron(base, expansion)[0:want_h, 0:want_w]
    return img


def _next_power_of_two(x):
    """
    References:
        https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return 2 ** (x - 1).bit_length()


def _next_multiple_of_two(x):
    """
    References:
        https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return x + (x % 2)


def _next_multiple_of(x, m):
    """
    References:
        https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return (x // m) * m + m
