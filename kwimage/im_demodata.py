# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import cv2
from . import im_cv2


_TEST_IMAGES = {
    'airport': {
        'fname': 'airport.jpg',
        'sha1': '52f15b9cccf2cc95a82ccacd96f1f15dc76a8544',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
    },
    'amazon': {
        'fname': 'amazon.jpg',
        'sha1': '50a475dd4b294eb9413971a20648b3329cd7ef4d',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b6/Fires_and_Deforestation_on_the_Amazon_Frontier%2C_Rondonia%2C_Brazil_-_August_12%2C_2007.jpg',
    },
    'astro': {
        'fname': 'astro.png',
        'sha1': '160b6e5989d2788c0296eac45b33e90fe612da23',
        'url': 'https://i.imgur.com/KXhKM72.png',
    },
    'carl': {
        'fname': 'carl.jpg',
        'sha1': 'f498fa6f6b24b4fa79322612144fedd5fad85bc3',
        'url': 'https://i.imgur.com/flTHWFD.png',  # imgur thinks this is a PNG for some reason
    },
    'paraview': {
        'fname': 'paraview.png',
        'sha1': 'd3c6240ccb4748e9bd5de07f0aa3f86724edeee7',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
    },
    'parrot': {
        'fname': 'parrot.png',
        'sha1': '6f97b8f9095031aa26152aaa16cbd4e7e7ea16d9',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
    },
    'stars': {
        'fname': 'stars.png',
        'sha1': 'bbf162d14537948e12169ccc26ca1b4e74f6a67e',
        'url': 'https://i.imgur.com/kCi7C1r.png',
    },
    'tsukuba_l': {
        'fname': 'tsukuba_l.png',
        'sha1': '9208dce1d8c6521e24a9105f90e361a0b355db69',
        'url': 'https://i.imgur.com/DhIKgGx.png',
    },
    'tsukuba_r': {
        'fname': 'tsukuba_r.png',
        'sha1': '10f9d2d832610253a3702d40f191e72e1af8b28b',
        'url': 'https://i.imgur.com/38RST9H.png',
    },
    'lowcontrast': {
        'fname': 'lowcontrast.jpg',
        'sha1': 'ade84f4aa22f07f58cd530882d4ecd92e0609b81',
        'url': 'https://i.imgur.com/dyC68Bi.jpg',
    }
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
        hasher_priority = ['sha1']

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

        space (str, default='rgb'):
            which colorspace to return in

        dsize (Tuple[int, int], default=None):
            if specified resizes image to this size

    Returns:
        ndarray: the requested image

    CommandLine:
        xdoctest -m kwimage.im_demodata grab_test_image

    Example:
        >>> for key in grab_test_image.keys():
        ...     grab_test_image(key)
        >>> grab_test_image('astro', dsize=(255, 255)).shape
        (255, 255, 3)
    """
    fpath = grab_test_image_fpath(key)
    bgr = cv2.imread(fpath)
    if dsize:
        interpolation = im_cv2._coerce_interpolation(interpolation,
                                                     cv2.INTER_LANCZOS4)
        bgr = cv2.resize(bgr, dsize, interpolation=interpolation)
    image = im_cv2.convert_colorspace(bgr, 'bgr', dst_space=space,
                                      implicit=True)
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
        >>> for key in grab_test_image.keys():
        ...     grab_test_image_fpath(key)
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
    hasher_priority = ['sha1']
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
