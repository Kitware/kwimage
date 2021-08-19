

def imread2_bench():
    """
    Another imread benchmark
    """
    import itertools as it

    def basis_product(basis):
        """
        Generates the Cartesian product of the ``basis.values()``, where each
        generated item labeled by ``basis.keys()``.
        """
        keys = list(basis.keys())
        for vals in it.product(*basis.values()):
            kw = dict(zip(keys, vals))
            yield kw

    data_basis = {
        'dsize': [
            (32, 32),
            (128, 128),
            (512, 512),
            (800, 800),
            (1200, 1200)
        ],
        'channels': [
            3,
            10,
            64,
            128,
        ],
        'dtype': [
            'uint8',
            'float32'
        ],
    }
    data_profiles = list(basis_product(data_basis))

    method_profiles = [
        # {'backend': 'skimage'},
        # {'backend': 'gdal'},
        {'backend': 'cv2', 'space': None},
    ]

    import numpy as np
    import ubelt as ub
    from os.path import join
    import kwimage

    dpath = ub.ensure_app_cache_dir('kwimage/bench/imread2')

    import timerit
    read_ti = timerit.Timerit(2, bestof=1, verbose=2)

    measures = []
    for data_profile in data_profiles:
        data_profile_key = ub.repr2(data_profile, nobr=1, itemsep='', keysep='=', nl=0, sv=1, sk=1, kvsep='=', sort=0)
        width, height = data_profile['dsize']
        channels = data_profile['channels']
        shape = (height, width, channels)

        data = (np.random.rand(*shape) * 255).astype(data_profile['dtype'])
        fpath = join(dpath, data_profile_key + '.tiff')
        kwimage.imwrite(fpath, data, backend='gdal')

        for method_profile in method_profiles:

            if method_profile['backend'] == 'cv2' and channels > 4:
                continue

            method_profile_key = ub.repr2(method_profile, nobr=1, itemsep='', keysep='=', nl=0, sv=1, sk=1, kvsep='=', sort=0)
            for timer in read_ti.reset(f'{data_profile_key} {method_profile_key}'):
                with timer:
                    kwimage.imread(fpath, **method_profile)

            time_result = {
                'mean': read_ti.mean(),
                'std': read_ti.std(),
                'min': read_ti.min(),
            }
            row = ub.dict_union(data_profile, method_profile, time_result)
            measures.append(row)

    import pandas as pd
    for row in measures:
        width, height = row['dsize']
        channels = row['channels']
        gridsize = width * height * channels
        if row['dtype'] == 'float32':
            row['bits'] =  gridsize * 32
        elif row['dtype'] == 'uint8':
            row['bits'] = gridsize * 8
        else:
            raise Exception

    df = pd.DataFrame(measures).sort_values('mean')
    print(df)

    import kwplot
    sns = kwplot.autosns()

    ax = sns.lineplot(data=df, x='bits', y='min', hue='backend')
    ax.set_xscale('log')
    ax.set_yscale('log')
