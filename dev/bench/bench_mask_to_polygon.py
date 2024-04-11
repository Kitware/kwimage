

def bench_mask_to_polygon():
    import ubelt as ub
    import pandas as pd
    import timerit

    import kwimage
    from kwimage.structs.mask import MaskFormat  # NOQA
    # mask = kwimage.Mask.random(shape=(512, 512), rng=0)

    # Mask with one small polygon away from the border
    poly = kwimage.Polygon.random().scale(32)

    # test_formats = [MaskFormat.C_MASK]
    test_formats = MaskFormat.cannonical
    # mask = input_formats['c_mask']
    # import xdev
    # xdev.profile_now(mask.to_multi_polygon)()

    method_lut = locals()  # can populate this some other way

    ti = timerit.Timerit(100, bestof=10, verbose=2)

    basis = {
        'format': test_formats,
        'pixels_are': ['points', 'areas'],
        'dim': [0, 128, 256, 512, 640]
    }
    xlabel = 'dim'
    kw_labels = ['pixels_are']
    group_labels = {
        'size': ['pixels_are'],
    }
    group_labels['hue'] = list(
        (ub.oset(basis) - {xlabel}) - set.union(*map(set, group_labels.values())))
    grid_iter = list(ub.named_product(basis))

    # For each variation of your experiment, create a row.
    rows = []
    for params in grid_iter:
        group_keys = {}
        for gname, labels in group_labels.items():
            group_keys[gname + '_key'] = ub.urepr(
                ub.dict_isect(params, labels), compact=1, si=1)
        key = ub.urepr(params, compact=1, si=1)
        kwargs = ub.dict_isect(params.copy(),  kw_labels)

        dim = int(params['dim'])
        ox = int(params['dim'] * 0.75)
        oy = int(params['dim'] * 0.65)

        mask = poly.translate(ox, oy).to_mask(dims=(dim, dim))
        mask = mask.toformat(params['format'])

        for timer in ti.reset(key):
            with timer:
                mask.to_multi_polygon()
        row = {
            'mean': ti.mean(),
            'min': ti.min(),
            'key': key,
            **group_keys,
            **params,
        }
        rows.append(row)

    if 0:
        import xdev
        xdev.profile_now(mask.to_multi_polygon)()

    # The rows define a long-form pandas data array.
    # Data in long-form makes it very easy to use seaborn.
    data = pd.DataFrame(rows)
    data = data.sort_values('min')
    print(data)

    plot = True
    if plot:
        # import seaborn as sns
        # kwplot autosns works well for IPython and script execution.
        # not sure about notebooks.
        import kwplot
        sns = kwplot.autosns()

        # Your variables may change
        ax = kwplot.figure(fnum=1, doclf=True).gca()
        plotkw = {}
        for gname, labels in group_labels.items():
            if labels:
                plotkw[gname] = gname + '_key'

        sns.lineplot(
            data=data, x=xlabel, y='min', marker='o', ax=ax, **plotkw
        )
        ax.set_title('Benchmark')
        ax.set_xlabel('A better x-variable description')
        ax.set_ylabel('A better y-variable description')
