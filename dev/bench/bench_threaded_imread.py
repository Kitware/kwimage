import ubelt as ub
import pandas as pd
import timerit
import kwcoco
import kwimage


def parallel_read_images(fpaths, backend, mode, workers):
    pool = ub.JobPool(mode=mode, max_workers=workers)
    for fpath in fpaths:
        pool.submit(kwimage.imread, fpath, backend=backend)

    imdata_objs = []
    for job in pool.as_completed():
        imdata = job.result()
        imdata_objs.append(imdata)


def bench_threaded_imread():

    dset = kwcoco.CocoDataset.demo('vidshapes256')
    fpaths = []
    for coco_img in dset.images().coco_images:
        fpath = coco_img.primary_image_filepath()
        fpaths.append(fpath)
        ub.JobPool()

    def method2(x):
        ret = [i for i in range(x)]
        return ret

    method_lut = locals()  # can populate this some other way

    ti = timerit.Timerit(100, bestof=10, verbose=2)

    basis = {
        # 'backend': ['gdal', 'cv2', 'skimage'],
        'backend': ['gdal'],
        'workers': [0, 4, 8],
        'mode': ['serial', 'thread', 'process'],
        # 'param_name': [param values],
    }
    xlabel = 'workers'
    hue_labels = list(ub.oset(basis) - {xlabel})
    grid_iter = list(ub.named_product(basis))
    for params in grid_iter:
        if params['mode'] == 'serial':
            params['workers'] = 0
        if params['workers'] == 0:
            params['mode'] = 'serial'

    grid_iter = list(ub.unique(grid_iter, key=ub.hash_data))

    # For each variation of your experiment, create a row.
    rows = []
    for params in grid_iter:
        hue_key = ub.urepr(ub.dict_isect(params, hue_labels), compact=1, si=1)
        key = ub.urepr(params, compact=1, si=1)
        kwargs = params.copy()
        # Timerit will run some user-specified number of loops.
        # and compute time stats with similar methodology to timeit
        for timer in ti.reset(key):
            with timer:
                parallel_read_images(fpaths=fpaths, **kwargs)
        row = {
            'mean': ti.mean(),
            'min': ti.min(),
            'key': key,
            'hue_key': hue_key,
            **params,
        }
        rows.append(row)

    # The rows define a long-form pandas data array.
    # Data in long-form makes it very easy to use seaborn.
    data = pd.DataFrame(rows)
    print(data)

    plot = True
    if plot:
        # import seaborn as sns
        # kwplot autosns works well for IPython and script execution.
        # not sure about notebooks.
        import kwplot
        sns = kwplot.autosns()
        plt = kwplot.autoplt()

        # Your variables may change
        ax = kwplot.figure(fnum=1, doclf=True).gca()
        sns.lineplot(data=data, x=xlabel, y='min', hue='hue_key', marker='o', ax=ax)
        ax.set_title('Benchmark')
        ax.set_xlabel('A better x-variable description')
        ax.set_ylabel('A better y-variable description')
        plt.show()

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_threaded_imread.py
    """
    bench_threaded_imread()
