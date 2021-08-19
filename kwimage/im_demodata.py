# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import cv2


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

        space (str, default='rgb'):
            which colorspace to return in

        dsize (Tuple[int, int], default=None):
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
    from kwimage import im_cv2
    if key == 'checkerboard':
        image = checkerboard()
    else:
        fpath = grab_test_image_fpath(key)
        bgr = cv2.imread(fpath)
        image = im_cv2.convert_colorspace(bgr, 'bgr', dst_space=space,
                                          implicit=True)
    if dsize:
        interpolation = im_cv2._coerce_interpolation(interpolation,
                                                     cv2.INTER_LANCZOS4)
        image = cv2.resize(image, dsize, interpolation=interpolation)
    return image


def grab_test_image_fpath(key='astro'):
    """
    Ensures that the test image exists (this might use the network) and returns
    the cached filepath to the requested image.

    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky

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
    return fpath

grab_test_image.keys = lambda: _TEST_IMAGES.keys()
grab_test_image_fpath.keys = lambda: _TEST_IMAGES.keys()


def checkerboard(num_squares=8, dsize=(512, 512)):
    """
    Creates a checkerboard image

    Args:
        num_squares (int): number of squares in a row
        dsize (Tuple[int, int]): width and height

    References:
        https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy

    Example:
        >>> from kwimage.im_demodata import *  # NOQA
        >>> img = checkerboard()
    """
    import numpy as np
    num_squares = 8
    num_pairs = num_squares // 2
    # img_size = 512
    w = dsize[0] // num_squares
    h = dsize[1] // num_squares
    base = np.array([[1, 0] * num_pairs, [0, 1] * num_pairs] * num_pairs)
    expansion = np.ones((h, w))
    img = np.kron(base, expansion)
    return img
