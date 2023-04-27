

def imread2_bench():
    """
    Another imread benchmark
    """
    import ubelt as ub

    data_basis = {
        'dsize': [
            (32, 32),
            (128, 128),
            (512, 512),
            # (800, 800),
            # (1200, 1200)
        ],
        'channels': [
            3,
            13,
            64,
            # 128,
        ],
        'dtype': [
            'uint8',
            'float32'
        ],
    }
    data_profiles = list(ub.named_product(data_basis))

    imread_profiles = [
        {'backend': 'skimage'},
        {'backend': 'gdal'},
        # {'backend': 'cv2', 'space': None},
    ]

    imwrite_profiles = [
        {'backend': 'skimage'},
        {'backend': 'gdal'},
        # {'backend': 'cv2', 'space': None},
    ]

    import numpy as np
    import ubelt as ub
    from os.path import join
    import kwimage

    dpath = ub.Path.appdir('kwimage/bench/imread2').ensuredir()

    import timerit
    read_ti = timerit.Timerit(1, bestof=1, verbose=2)

    measures = []
    for data_profile in data_profiles:
        data_profile_key = ub.urepr(data_profile, compact=1, sort=0)
        width, height = data_profile['dsize']
        channels = data_profile['channels']
        shape = (height, width, channels)

        data = (np.random.rand(*shape) * 255).astype(data_profile['dtype'])
        fpath = join(dpath, data_profile_key + '.tiff')

        for imwrite_profile in imwrite_profiles:

            imwrite_profile_key = ub.urepr(imwrite_profile, compact=1, sort=0)
            kwimage.imwrite(fpath, data, **imwrite_profile)

            for imread_profile in imread_profiles:

                if imread_profile['backend'] == 'cv2' and channels > 4:
                    continue

                imread_profile_key = ub.urepr(imread_profile, compact=1, sort=0)

                io_profile_key = 'imwrite({})->imread({})'.format(
                    imwrite_profile_key, imread_profile_key)

                for timer in read_ti.reset(f'{data_profile_key} {io_profile_key}'):
                    with timer:
                        kwimage.imread(fpath, **imread_profile)

                time_result = {
                    'mean': read_ti.mean(),
                    'std': read_ti.std(),
                    'min': read_ti.min(),
                }
                row = ub.dict_union(data_profile, time_result)
                row['imwrite_profile_key'] = imwrite_profile_key
                row['imread_profile_key'] = imread_profile_key
                row['data_profile_key'] = data_profile_key
                row['io_profile_key'] = io_profile_key
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
    plt = kwplot.autoplt()

    df['io_profile_key'] = (
        'imwrite(' + df['imwrite_profile_key'] + ')' +
        '->' +
        'imread(' + df['imread_profile_key'] + ')'
    )
    ax = sns.lineplot(data=df, x='bits', y='min', hue='io_profile_key')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_imread2.py
    """
    imread2_bench()
