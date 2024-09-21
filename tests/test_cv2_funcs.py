import pytest


def test_cv2_func_dtypes():
    import kwimage
    from functools import partial
    import numpy as np
    try:
        import cv2  # NOQA
    except ImportError:
        pytest.skip('requires cv2')

    # TODO: make this work with more dtypes
    funcs = {
        'resize': partial(kwimage.imresize, scale=2),
        'morphology': partial(kwimage.morphology, mode='erode'),
        'warp_affine': partial(kwimage.warp_affine, transform=kwimage.Affine.random()),
        'gaussian_blur': partial(kwimage.gaussian_blur),
        'imcrop': partial(kwimage.imcrop, dsize=(10, 10)),
    }

    dtypes = {
        'uint8': np.uint8,
        'int8': np.int8,
        'uint16': np.uint16,
        'int16': np.int16,
        'uint32': np.uint32,
        'int32': np.int32,
        'uint64': np.uint64,
        'int64': np.int64,
    }

    rows = []

    for dtk, dtype in dtypes.items():
        img = (np.random.rand(32, 32, 1) * 10).astype(dtype)
        for k, func in funcs.items():
            try:
                new_img = func(img)
                assert new_img.dtype == img.dtype, 'bad array shape'
            except Exception as ex:
                status = str(ex)
                success = False
            else:
                status = 'pass'
                success = True
            rows.append({
                'func': k,
                'dtype': dtk,
                'success': success,
                'status': status,
            })

    must_work = ['uint8', 'int16', 'uint16']
    for row in rows:
        if row['dtype'] in must_work:
            assert row['success']

    if 0:
        import pandas as pd
        import rich
        df = pd.DataFrame(rows)
        rich.print(df)
        rich.print(df.pivot(['func'], ['dtype'], ['success']).to_string())
