"""
Keep a manifest of demo images used for testing
"""
import ubelt as ub


_TEST_IMAGES = {
    'airport': {
        'fname': 'airport.jpg',
        'sha256': 'bff5f9212d5c77dd47f2b80e5dc1b4409fa7813b08fc39b504294497b3483ffc',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
        'ipfs_cids': [
            'bafkreif76x4sclk4o7oup4vybzo4dncat6tycoyi7q43kbbjisl3gsb77q',
        ],
    },
    'amazon': {
        'fname': 'amazon.jpg',
        'sha256': 'ef352b60f2577692ab3e9da19d09a49fa9da9937f892afc48094988a17c32dc3',
        'url': 'https://data.kitware.com/api/v1/file/611e9f4b2fa25629b9dc0ca2/download',
        'ipfs_cids': [
            'bafybeia3telu2s742xco3ap5huh4tk45cikwuxczwhrd6gwc3rcuat7odq',
        ],
    },
    'astro': {
        'fname': 'astro.png',
        'sha256': '9f2b4671e868fd51451f03809a694006425eee64ad472f7065da04079be60c53',
        'url': 'https://i.imgur.com/KXhKM72.png',
        'note': 'An image of Eileen Collins.',
        'ipfs_cids': [
            'bafybeif2w42xgi6vkfuuwmn3c6apyetl56fukkj6wnfgzcbsrpocciuv3i',
        ],
    },
    'carl': {
        'fname': 'carl.jpg',
        'sha256': '94c2a5acfe53cf7d0f6c75577bd947bf559a4a99d1a200e0fca537602a966da2',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/b/be/Carl_Sagan_Planetary_Society.JPG',
        'mirrors': [
            'https://i.imgur.com/YnrLyry.jpg',
        ],
        'note': 'An image of Carl Sagan.',
        'ipfs_cids': [
            'bafkreieuyks2z7stz56q63dvk555sr57kwnevgoruiaob7ffg5qcvftnui',
        ],
    },
    'lowcontrast': {
        'fname': 'lowcontrast.jpg',
        'sha256': '532572a245d2b401583488ccf9f033e5960dc9f3f56b8ca6b933a8986ec9e95e',
        'url': 'https://i.imgur.com/dyC68Bi.jpg',
        'ipfs_cids': [
            'bafkreictevzkeroswqavqneizt47am7fsyg4t47vnogknojtvcmg5spjly',
        ],
    },
    'paraview': {
        'fname': 'paraview.png',
        'sha256': '859423aefce1037b2b6959b74d7b137a4104acd6db95a9247abb26c2d0aa93b8',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
        'ipfs_cids': [
            'bafkreiefsqr257hban5sw2kzw5gxwe32ieckzvw3swusi6v3e3bnbkutxa',
        ],
    },
    'parrot': {
        'fname': 'parrot.png',
        'sha256': 'fadd4cdddc46e43185999421dcb1ae9d3ba6d13b5b6d0acc05268fc7246f3e59',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
        'ipfs_cids': [
            'bafkreih23vgn3xcg4qyylgmueholdlu5hotnco23nufmybjgr7dsi3z6le',
        ],
    },
    'stars': {
        'fname': 'stars.png',
        'sha256': '36391b4d36b4b5e2597c53f9465951910542fbec82f5a0213715759d1de9714f',
        'url': 'https://i.imgur.com/kCi7C1r.png',
        'ipfs_cids': [
            'bafkreibwhenu2nvuwxrfs7ct7fdfsumravbpx3ec6wqccnyvowor32lrj4',
        ],
    },
    'pm5644': {
        'fname': 'Philips_Pattern_pm5644.png',
        'sha256': 'eba500341492649d4fa4e83b5200abbffa6673de5de4c20ed669dedeb00d3941',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Philips_Pattern_pm5644.png',
        'note': 'A test pattern good for checking aliasing effects',
        'ipfs_cids': [
            'bafkreihluuadifesmsou7jhihnjabk577jthhxs54tba5vtj33pladjzie',
        ],
    },
    'tsukuba_l': {
        'fname': 'tsukuba_l.png',
        'sha256': 'e29144841f4e2200e88eb6ad928cfa3ee0c55ccac0a28532c9293c4a5e0b284d',
        'url': 'https://i.imgur.com/DhIKgGx.png',
        'mirrors': [
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imL.png',
        ],
        'ipfs_cids': [
            'bafkreihcsfciih2oeiaordvwvwjiz6r64dcvzswaukctfsjjhrff4cziju',
        ],
    },
    'tsukuba_r': {
        'fname': 'tsukuba_r.png',
        'sha256': 'fb4e8b1561c177a9aba23693bd576d0e06f5778b8d44e1c1cc5c5dd35d5fd1d4',
        'url': 'https://i.imgur.com/38RST9H.png',
        'mirrors': [
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imR.png',
        ],
        'ipfs_cids': [
            'bafkreih3j2frkyobo6u2xirwso6vo3ioa32xpc4nitq4dtc4lxjv2x6r2q',
        ],
    },
}


def _update_hashes():
    """
    for dev use to update hashes of the demo images

    CommandLine:
        xdoctest -m kwimage.im_demodata _update_hashes
        xdoctest -m kwimage.im_demodata _update_hashes --require-hashes
    """
    TEST_IMAGES = _TEST_IMAGES.copy()

    ENSURE_IPFS = ub.argflag('--ensure-ipfs')
    REQUIRE_EXISTING_HASH = ub.argflag('--require-hashes')

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

        if ENSURE_IPFS:
            ipfs_cids = item.get('ipfs_cids', [])
            if not ipfs_cids:
                info = ub.cmd('ipfs add {} --cid-version=1'.format(fpath), verbose=3)
                cid = info['out'].split(' ')[1]
                ipfs_cids.append(cid)
                item['ipfs_cids'] = ipfs_cids

    print('_TEST_IMAGES = ' + ub.repr2(TEST_IMAGES, nl=3, sort=0))

    if ENSURE_IPFS:
        args = ' '.join(list(ub.flatten([item.get('ipfs_cids') for item in TEST_IMAGES.values()])))
        command = 'ipfs pin add ' + args
        print('To pin on another machine:')
        print(command)


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
        >>> key_to_image = {}
        >>> for key in kwimage.grab_test_image.keys():
        >>>     print('attempt to grab key = {!r}'.format(key))
        >>>     # specifying dsize will returned a resized variant
        >>>     imdata = kwimage.grab_test_image(key, dsize=(256, None))
        >>>     key_to_image[key] = imdata
        >>>     print('grabbed key = {!r}'.format(key))
        >>> # xdoc: +REQUIRES(--show)
        >>> # xdoc: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> to_stack = [kwimage.draw_header_text(
        >>>     imdata, text=key, color='kw_blue')
        >>>     for key, imdata in key_to_image.items()]
        >>> stacked = kwimage.stack_images_grid(to_stack, bg_value='kw_darkgray')
        >>> stacked = kwimage.draw_header_text(stacked, 'kwimage.grab_test_image', fit=True, color='kitware_green')
        >>> kwplot.imshow(stacked)
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


def checkerboard(num_squares='auto', square_shape='auto', dsize=(512, 512),
                 dtype=float, on_value=1, off_value=0):
    """
    Creates a checkerboard image

    Args:
        num_squares (int | str):
            Number of squares in a row. If 'auto' defaults to 8

        square_shape (int | Tuple[int, int] | str):
            If 'auto', chosen based on `num_squares`. Otherwise this is
            the height, width of each square in pixels.

        dsize (Tuple[int, int]): width and height

        dtype (type): return data type

        on_value (Number | int):
            The value of one checker. Defaults to 1.

        off_value (Number | int):
            The value off the other checker. Defaults to 0.

    References:
        https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> img = kwimage.checkerboard()
        >>> print(kwimage.checkerboard(dsize=(16, 16)).shape)
        >>> print(kwimage.checkerboard(num_squares=4, dsize=(16, 16)).shape)
        >>> print(kwimage.checkerboard(square_shape=3, dsize=(23, 17)).shape)
        >>> print(kwimage.checkerboard(square_shape=3, dsize=(1451, 1163)).shape)
        >>> print(kwimage.checkerboard(square_shape=3, dsize=(1202, 956)).shape)
        >>> print(kwimage.checkerboard(dsize=(4, 4), on_value=(255, 0, 0), off_value=(0, 0, 1), dtype=np.uint8))

    Example:
        >>> import kwimage
        >>> img = kwimage.checkerboard(
        >>>     dsize=(64, 64), on_value='kw_green', off_value='kw_blue')
        >>> # xdoc: +REQUIRES(--show)
        >>> # xdoc: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(kwimage.checkerboard))
    """
    import numpy as np
    import kwimage
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

    if isinstance(on_value, str):
        on_value = kwimage.Color(on_value).forimage(dtype)

    if isinstance(off_value, str):
        off_value = kwimage.Color(off_value).forimage(dtype)

    num_pairs_w = int(num_w // 2)
    num_pairs_h = int(num_h // 2)
    # img_size = 512
    base = np.array([
        [on_value, off_value] * num_pairs_w,
        [off_value, on_value] * num_pairs_w
    ] * num_pairs_h, dtype=dtype)

    if len(base.shape) == 3:
        base = base.transpose([2, 0, 1])
    expansion = np.ones((h, w), dtype=dtype)
    img = np.kron(base, expansion)[0:want_h, 0:want_w]
    if len(base.shape) == 3:
        img = img.transpose([1, 2, 0])
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
