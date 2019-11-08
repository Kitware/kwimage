# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import cv2
from . import im_cv2


_TEST_IMAGES = {
    'astro': {
        'url': 'https://i.imgur.com/KXhKM72.png',
        'sha1': '160b6e5989d2788c0296eac45b33e90fe612da23',
    },
    'carl': {
        'url': 'https://i.imgur.com/flTHWFD.png',
        'sha1': 'f498fa6f6b24b4fa79322612144fedd5fad85bc3',
    },
    'stars': {
        'url': 'https://i.imgur.com/kCi7C1r.png',
        'sha1': 'bbf162d14537948e12169ccc26ca1b4e74f6a67e',
    },
    'paraview': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
        'sha1': 'd3c6240ccb4748e9bd5de07f0aa3f86724edeee7',
    },
    'airport': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
        'sha1': '52f15b9cccf2cc95a82ccacd96f1f15dc76a8544',
    },
    'parrot': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
        'sha1': '6f97b8f9095031aa26152aaa16cbd4e7e7ea16d9',
    }
}


def grab_test_image(key='astro', space='rgb', dsize=None,
                    interpolation='lanczos'):
    """
    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky
            airport - SkySat image of Beijing Capital International Airport on 18 February 2018

        space (str): which colorspace to return in (defaults to RGB)

    Example:
        >>> for key in grab_test_image.keys():
        ...     grab_test_image(key)
        >>> grab_test_image('astro', dsize=(255, 255)).shape
        (255, 255, 3)
    """
    fpath = grab_test_image_fpath(key)
    bgr = cv2.imread(fpath)
    if dsize:
        interpolation = im_cv2._rectify_interpolation(interpolation,
                                                      cv2.INTER_LANCZOS4)
        bgr = cv2.resize(bgr, dsize, interpolation=interpolation)
    image = im_cv2.convert_colorspace(bgr, 'bgr', dst_space=space,
                                      implicit=True)
    return image


def grab_test_image_fpath(key='astro'):
    """
    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky

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

    if 'sha1' in item:
        fpath = ub.grabdata(item['url'], hash_prefix=item['sha1'],
                            appname='kwimage/demodata', hasher='sha1')
    else:
        fpath = ub.grabdata(item['url'], appname='kwimage')
    return fpath

grab_test_image.keys = lambda: _TEST_IMAGES.keys()
grab_test_image_fpath.keys = lambda: _TEST_IMAGES.keys()
