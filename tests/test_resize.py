import kwimage
import kwarray
import numpy as np
import ubelt as ub


def image_variations(image_basis):
    """
    Helper to make several variations of image inputs for opencv with different
    dtypes etc..
    """
    rng = kwarray.ensure_rng(0)

    if image_basis is None:
        image_basis = {
            'dims': [(32, 32), (37, 41), (53, 31)],
            'channels': [None, 1, 3, 4, 20, 1024],
            'dtype': ['uint8', 'int64', 'float32', 'float64'],
        }

    # TODO: how to specify conditionals?
    # conditionals = {
    #     np.uint8
    # }

    for imgkw in list(ub.named_product(image_basis)):
        if imgkw['channels'] is None:
            shape = imgkw['dims']
        else:
            shape = imgkw['dims'] + (imgkw['channels'],)
        dtype = np.dtype(imgkw['dtype'])
        img = rng.rand(*shape)
        if dtype.kind in {'i', 'u'}:
            img = img * 255
        img = img.astype(dtype)
        yield imgkw, img


def test_imresize_multi_channel():
    """
    Test that imresize works with multiple channels in various configurations
    """

    resize_kw_basis = {
        'dsize': [
            # (10, 10),
            (60, 60)
        ],
        'interpolation': [
            'auto',
            # 'area',
            # 'linear',
            # 'cubic',
            'nearest'
        ],
        'antialias': [
            True,
            # False,
        ]
    }

    image_basis = {
        'dims': [
            # (32, 32),
            # (37, 41),
            (53, 31)
        ],
        'channels': [
            None,
            1,
            3,
            4,
            20,
            1024,
        ],
        'dtype': [
            'uint8',
            'float32',
            'float64'
        ],
    }

    resize_kw_list = list(ub.named_product(resize_kw_basis))

    failures = []
    success = []
    import timerit
    ti = timerit.Timerit(1, bestof=1, verbose=0)

    for imgkw, img in image_variations(image_basis):
        for resize_kw in resize_kw_list:
            params = dict(resize_kw=resize_kw, imgkw=imgkw)
            try:
                label = ub.repr2(params, nl=0, nobr=True, si=1, sv=1, kvsep='=', itemsep='')
                for timer in ti.reset(label):
                    with timer:
                        kwimage.imresize(img, **resize_kw)
            except Exception:
                failures.append(label)
                print('FAILED = {!r}'.format(label))
                raise
            else:
                success.append(label)

    print('n_pass = {}'.format(len(success)))
    print('n_fail = {}'.format(len(failures)))
    print('failures = {}'.format(ub.repr2(failures, nl=1)))
