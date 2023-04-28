import torch
import numpy as np
import kwimage
import copy
import ubelt as ub
import itertools as it
# from kwimage.algo._nms_backend.torch_nms import torch_nms


def ensure_numpy_indices(keep):
    import kwarray
    keep = kwarray.ArrayAPI.numpy(keep)
    # if torch.is_tensor(keep):
    #     keep = keep.data.cpu().numpy()
    # else:
    #     keep = keep
    keep = np.array(sorted(np.array(keep).ravel()))
    return keep


def benchamrk_det_nms():
    """
    Benchmarks different implementations of non-max-supression on the CPU, GPU,
    and using cython / numpy / torch.

    CommandLine:
        xdoctest -m ~/code/kwimage/dev/bench_nms.py benchamrk_det_nms --show

    SeeAlso:
        PJR Darknet NonMax supression
        https://github.com/pjreddie/darknet/blob/master/src/box.c

        Lightnet NMS
        https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/data/transform/_postprocess.py#L116
    """

    # N = 200
    # bestof = 50
    N = 1
    bestof = 1

    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]

    # max number of boxes yolo will spit out at a time
    max_boxes = 19 * 19 * 5

    xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, max_boxes]
    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500]

    # Demo values
    xdata = [0, 1, 2, 3, 10, 100, 200, 300, 500]

    if ub.argflag('--small'):
        xdata = [10, 100, 500, 1000, 1500, 2000, 5000, 10000]

    if ub.argflag('--medium'):
        xdata = [1000, 5000, 10000, 20000, 50000, ]

    if ub.argflag('--large'):
        xdata = [1000, 5000, 10000, 20000, 50000, 100000, ]

    if ub.argflag('--extra-large'):
        xdata = [1000, 2000, 10000, 20000, 40000, 100000, 200000, ]

    title_parts = []

    SMALL_BOXES = ub.argflag('--small-boxes')
    if SMALL_BOXES:
        title_parts.append('small boxes')
    else:
        title_parts.append('large boxes')

    # NOTE: for large images we may have up to 21,850,753 detections!

    thresh = float(ub.argval('--thresh', default=0.4))
    title_parts.append('thresh={:.2f}'.format(thresh))

    from kwimage.algo.algo_nms import available_nms_impls
    valid_impls = available_nms_impls()
    print('valid_impls = {!r}'.format(valid_impls))

    basis = {
        'type': [
            'ndarray',
            'tensor',
            'tensor0'
        ],
        # 'daq': [True, False],
        # 'daq': [False],
        # 'device': [None],
        # 'impl': valid_impls,
        'impl': valid_impls + ['auto'],
    }

    if ub.argflag('--daq'):
        basis['daq'] = [True, False]

    # if torch.cuda.is_available():
    #     basis['device'].append(0)

    combos = [ub.dzip(basis.keys(), vals)
              for vals in it.product(*basis.values())]

    def is_valid_combo(combo):
        # if combo['impl'] in {'py', 'cython_cpu'} and combo['device'] is not None:
        #     return False
        # if combo['type'] == 'ndarray' and combo['impl'] == 'cython_gpu':
        #     if combo['device'] is None:
        #         return False
        # if combo['type'] == 'ndarray' and combo['impl'] != 'cython_gpu':
        #     if combo['device'] is not None:
        #         return False

        # if combo['type'].endswith('0'):
        #     if combo['impl'] in {'numpy', 'cython_gpu', 'cython_cpu'}:
        #         return False

        # if combo['type'] == 'ndarray':
        #     if combo['impl'] in {'torch'}:
        #         return False

        REMOVE_SLOW = True
        if REMOVE_SLOW:
            known_bad = [
                {'impl': 'torch', 'type': 'tensor'},
                {'impl': 'numpy', 'type': 'tensor'},
                # {'impl': 'cython_gpu', 'type': 'tensor'},
                {'impl': 'cython_cpu', 'type': 'tensor'},

                # {'impl': 'torch', 'type': 'tensor0'},
                {'impl': 'numpy', 'type': 'tensor0'},
                # {'impl': 'cython_gpu', 'type': 'tensor0'},
                # {'impl': 'cython_cpu', 'type': 'tensor0'},

                {'impl': 'torchvision', 'type': 'ndarray'},
            ]
            for known in known_bad:
                if all(combo[key] == val for key, val in known.items()):
                    return False

        return True
    combos = list(filter(is_valid_combo, combos))

    times = ub.ddict(list)
    for num in xdata:

        if num > 10000:
            N = 1
            bestof = 1
        if num > 1000:
            N = 3
            bestof = 1
        if num > 100:
            N = 10
            bestof = 3
        elif num > 10:
            N = 100
            bestof = 10
        else:
            N = 1000
            bestof = 10
        print('\n\n---- number of boxes = {} ----\n'.format(num))

        outputs = {}

        ti = ub.Timerit(N, bestof=bestof, verbose=1)

        # Build random test boxes and scores
        np_dets1 = kwimage.Detections.random(num // 2, scale=1000.0, rng=0)
        np_dets1.data['boxes'] = np_dets1.boxes.to_xywh()

        if SMALL_BOXES:
            max_dim = 100
            np_dets1.boxes.data[..., 2] = np.minimum(np_dets1.boxes.width, max_dim).ravel()
            np_dets1.boxes.data[..., 3] = np.minimum(np_dets1.boxes.height, max_dim).ravel()

        np_dets2 = copy.deepcopy(np_dets1)
        np_dets2.boxes.translate(10, inplace=True)
        # add boxes that will definately be removed
        np_dets = kwimage.Detections.concatenate([np_dets1, np_dets2])

        # make all scores unique to ensure comparability
        np_dets.scores[:] = np.linspace(0, 1, np_dets.num_boxes())

        np_dets.data['scores'] = np_dets.scores.astype(np.float32)
        np_dets.boxes.data = np_dets.boxes.data.astype(np.float32)

        typed_data = {}
        # ----------------------------------

        import netharn as nh
        for combo in combos:
            print('combo = {}'.format(ub.urepr(combo, nl=0)))

            label = nh.util.make_idstr(combo)
            mode = combo.copy()

            # if mode['impl'] == 'cython_gpu':
            #     mode['device_id'] = mode['device']

            mode_type = mode.pop('type')

            if mode_type in typed_data:
                dets = typed_data[mode_type]
            else:
                if mode_type == 'ndarray':
                    dets = np_dets.numpy()
                elif mode_type == 'tensor':
                    dets = np_dets.tensor(None)
                elif mode_type == 'tensor0':
                    dets = np_dets.tensor(0)
                else:
                    raise KeyError
                typed_data[mode_type] = dets

            for timer in ti.reset(label):
                with timer:
                    keep = dets.non_max_supression(thresh=thresh, **mode)
                    torch.cuda.synchronize()
            times[ti.label].append(ti.min())
            outputs[ti.label] = ensure_numpy_indices(keep)

        # ----------------------------------

        # Check that all kept boxes do not have more than `threshold` ious
        if 0:
            for key, keep_idxs in outputs.items():
                kept = np_dets.take(keep_idxs).boxes
                ious = kept.ious(kept)
                max_iou = (np.tril(ious) - np.eye(len(ious))).max()
                if max_iou > thresh:
                    print('{} produced a bad result with max_iou={}'.format(key, max_iou))

        # Check result consistency:
        print('\nResult stats:')
        for key in sorted(outputs.keys()):
            print('    * {:<20}: num={}'.format(key, len(outputs[key])))

        print('\nResult overlaps (method1, method2: jaccard):')
        datas = []
        for k1, k2 in it.combinations(sorted(outputs.keys()), 2):
            idxs1 = set(outputs[k1])
            idxs2 = set(outputs[k2])
            jaccard = len(idxs1 & idxs2) / max(len(idxs1 | idxs2), 1)
            datas.append((k1, k2, jaccard))

        datas = sorted(datas, key=lambda x: -x[2])
        for k1, k2, jaccard in datas:
            print('    * {:<20}, {:<20}: {:0.4f}'.format(k1, k2, jaccard))

    if True:
        ydata = {key: 1.0 / np.array(vals) for key, vals in times.items()}
        ylabel = 'Hz'
        reverse = True
        yscale = 'symlog'
    else:
        ydata = {key: np.array(vals) for key, vals in times.items()}
        ylabel = 'seconds'
        reverse = False
        yscale = 'linear'
    scores = {
        key: vals[-1]
        for key, vals in ydata.items()
    }
    ydata = ub.dict_subset(ydata, ub.argsort(scores, reverse=reverse))

    ###
    times_of_interest = [0, 10, 100, 200, 1000]
    times_of_interest = xdata

    lines = []
    record = lines.append
    record('### times_of_interest = {!r}'.format(times_of_interest))
    for x in times_of_interest:

        if times_of_interest[-1] == x:
            record('else:')
        elif times_of_interest[0] == x:
            record('if num <= {}:'.format(x))
        else:
            record('elif num <= {}:'.format(x))

        if x in xdata:
            pos =  xdata.index(x)
            score_wrt_x = {}
            for key, vals in ydata.items():
                score_wrt_x[key] = vals[pos]

            typekeys = ['tensor0', 'tensor', 'ndarray']
            type_groups = dict([
                (b, ub.group_items(score_wrt_x, lambda y: y.endswith(b))[True])
                for b in typekeys
            ])
            # print('\n=========')
            # print('x = {!r}'.format(x))
            record('    if code not in {!r}:'.format(set(typekeys)))
            record('        raise KeyError(code)')
            for typekey, group in type_groups.items():
                # print('-------')
                record('    if code == {!r}:'.format(typekey))
                # print('typekey = {!r}'.format(typekey))
                # print('group = {!r}'.format(group))
                group_x = ub.dict_isect(score_wrt_x, group)
                valid_keys = ub.argsort(group_x, reverse=True)
                valid_x = ub.dict_subset(group_x, valid_keys)
                # parts = [','.split(k) for k in valid_keys]
                ordered_impls = []
                ordered_impls2 = ub.odict()
                for k in valid_keys:
                    vals = valid_x[k]
                    p = k.split(',')
                    d = dict(i.split('=') for i in p)
                    ordered_impls2[d['impl']] = vals
                    ordered_impls.append(d['impl'])

                ordered_impls = list(ub.oset(ordered_impls) - {'auto'})
                ordered_impls2.pop('auto')
                record('        # {}'.format(ub.urepr(ordered_impls2, precision=1, nl=0, explicit=True)))
                record('        preference = {}'.format(ub.urepr(ordered_impls, nl=0)))
    record('### end times of interest ')
    print(ub.indent('\n'.join(lines), ' ' * 8))
    ###

    markers = {
        key: 'o' if 'auto' in key else ''
        for key, score in scores.items()
    }

    if ub.argflag('--daq'):
        markers = {
            key: '+' if 'daq=True' in key else ''
            for key, score in scores.items()
        }

    labels = {
        key: '{:.2f} {} - {}'.format(score, ylabel[0:3], key)
        for key, score in scores.items()
    }

    title = 'NSM-impl speed: ' + ', '.join(title_parts)

    import kwplot
    kwplot.autompl()
    kwplot.multi_plot(
        xdata, ydata, xlabel='num boxes', ylabel=ylabel, label=labels,
        yscale=yscale,
        title=title,
        marker=markers,
        # xscale='symlog',
    )

    kwplot.show_if_requested()


if __name__ == '__main__':
    """

    CommandLine:
        xdoctest -m kwimage.algo.algo_nms available_nms_impls

        python ~/code/kwimage/dev/bench_nms.py  --show --small-boxes --thresh=0.1
        python ~/code/kwimage/dev/bench_nms.py  --show --small-boxes --thresh=0.8
        python ~/code/kwimage/dev/bench_nms.py  --show --small-boxes --thresh=1.0
        python ~/code/kwimage/dev/bench_nms.py  --show

        python ~/code/kwimage/dev/bench_nms.py  --show --small
        python ~/code/kwimage/dev/bench_nms.py  --show --small --small-boxes
        python ~/code/kwimage/dev/bench_nms.py  --show --medium
        python ~/code/kwimage/dev/bench_nms.py  --show --large
        python ~/code/kwimage/dev/bench_nms.py  --show --extra-large --small-boxes
    """
    benchamrk_det_nms()
