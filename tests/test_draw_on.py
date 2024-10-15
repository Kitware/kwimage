import kwimage
import ubelt as ub
import numpy as np


def _random_drawables():
    from kwimage._backend_info import _have_cv2
    if not _have_cv2():
        import pytest
        pytest.skip('requires cv2')
    drawables = {
        'boxes': kwimage.Boxes.random(),
        'points': kwimage.Points.random(),
        'polys': kwimage.Polygon.random(),
        'mpolys': kwimage.MultiPolygon.random(),
        'dets': kwimage.Polygon.random(),
    }
    return drawables


def _random_extreme_images():
    images = {}
    for w in [0, 1, 2]:
        for h in [0, 1, 2]:
            for nchans in [None, 1, 3]:
                key = f'{w}x{h}x{nchans}'
                if nchans is None:
                    shape = (h, w)
                else:
                    shape = (h, w, nchans)
                data = np.random.rand(*shape)
                images[key] = data
    return images


def test_draw_on_extreme_sizes():
    """
    Test that draw_on doesn't fail with zero sized images or small images in
    general.
    """
    from kwimage._backend_info import _have_cv2
    if not _have_cv2():
        import pytest
        pytest.skip('requires cv2')
    drawables = _random_drawables()
    images = _random_extreme_images()

    errors = []
    for draw_key, drawable in drawables.items():
        for im_key, image in images.items():

            try:
                result = drawable.draw_on(image)
                assert result.shape[0:2] == image.shape[0:2]
            except Exception as ex:
                errors.append((draw_key, im_key, ex))
                raise

    if errors:
        print('errors = {}'.format(ub.urepr(errors, nl=1)))
        raise AssertionError('errors = {}'.format(ub.urepr(errors, nl=1)))
