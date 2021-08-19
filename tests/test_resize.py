import kwimage
import itertools as it
import numpy as np
import kwarray
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

    for imgkw in list(basis_product(image_basis)):
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


def basis_product(basis):
    """
    Args:
        basis (Dict[str, List[T]]): list of values for each axes

    Yields:
        Dict[str, T] - points in the grid

    TODO:
        - [ ] Where does this live?

        - [ ] What is a better name?

            * labeled_product
            * grid_product

    Example:
        >>> basis = {
        >>>     'a': [1, 2, 3],
        >>>     'b': [4, 5],
        >>>     'c': [6],
        >>> }
        >>> list(basis_product(basis))
        [{'a': 1, 'b': 4, 'c': 6},
         {'a': 1, 'b': 5, 'c': 6},
         {'a': 2, 'b': 4, 'c': 6},
         {'a': 2, 'b': 5, 'c': 6},
         {'a': 3, 'b': 4, 'c': 6},
         {'a': 3, 'b': 5, 'c': 6}]
    """
    keys = list(basis.keys())
    for vals in it.product(*basis.values()):
        yield dict(zip(keys, vals))


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
            (32, 32),
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

    resize_kw_list = list(basis_product(resize_kw_basis))

    failures = []
    success = []
    ti = ub.Timerit(1, bestof=1, verbose=1)

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

    # print('ti.rankings = {}'.format(
    #     ub.repr2(ti.rankings, nl=2, align=':', precision=6)))

    # np.split(img, img.shape[-1], -1)
    # if 0:
    #     # numpy seems faster
    #     img = np.random.rand(64, 64, 3)
    #     import timerit
    #     ti = timerit.Timerit(100, bestof=10, verbose=2)
    #     for timer in ti.reset('cv2'):
    #         with timer:
    #             parts = cv2.split(img)

    #     for timer in ti.reset('np'):
    #         with timer:
    #             parts = np.split(img, img.shape[-1], -1)
    #     parts = cv2.split(img)
