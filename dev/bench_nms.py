import torch
import numpy as np
import kwimage
import copy
import ubelt as ub
import itertools as it
# from kwimage.algo._nms_backend.torch_nms import torch_nms


def ensure_numpy_indices(keep):
    if torch.is_tensor(keep):
        keep = keep.data.cpu().numpy()
    else:
        keep = keep
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

    ydata = ub.ddict(list)
    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]

    # max number of boxes yolo will spit out at a time
    max_boxes = 19 * 19 * 5

    xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, max_boxes]
    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    # xdata = [10, 100, 500, 1000, 1500, 2000]
    xdata = [1000, 5000, 10000, 20000, 50000, 100000, 1000000, 10000000 ]

    # NOTE: for large images we may have up to 21,850,753 detections!

    thresh = 0.01

    from kwimage.algo.algo_nms import available_nms_impls
    valid_impls = available_nms_impls()
    print('valid_impls = {!r}'.format(valid_impls))

    measure_gpu = True and torch.cuda.is_available()
    measure_cpu = True
    measure_daq = True
    measure_auto = False

    measure_cython_gpu = False
    measure_cython_cpu = False
    measure_torch = False
    measure_torch_cpu = False

    gpu = torch.device('cuda', 0)

    for num in xdata:
        print('\n\n---- number of boxes = {} ----\n'.format(num))

        outputs = {}

        ti = ub.Timerit(N, bestof=bestof, verbose=1)

        # Build random test boxes and scores
        np_dets1 = kwimage.Detections.random(num // 2, scale=1000.0, rng=0)
        np_dets1.data['boxes'] = np_dets1.boxes.to_xywh()

        SMALL_BOXES = True
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

        # ----------------------------------

        if measure_daq:
            if 'cpu' in valid_impls:
                for timer in ti.reset('daq_cython(cpu)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='cpu')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

            if 'gpu' in valid_impls:
                for timer in ti.reset('daq_cython(gpu)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='gpu')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

            if 'py' in valid_impls:
                for timer in ti.reset('daq_cython(py)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='py')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

            if measure_auto:
                for timer in ti.reset('daq_cython(auto)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='auto')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

        if measure_cpu:
            if 'torch' in valid_impls and measure_torch_cpu and measure_torch:
                cpu_dets = np_dets.tensor(None)
                for timer in ti.reset('torch(cpu)'):
                    with timer:
                        keep = cpu_dets.non_max_supression(thresh=thresh, impl='torch')
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

        if measure_gpu:
            # Move boxes to the GPU
            if 'torch' in valid_impls and measure_torch:
                gpu_dets = np_dets.tensor(gpu)
                for timer in ti.reset('torch(gpu)'):
                    with timer:
                        keep = gpu_dets.non_max_supression(thresh=thresh, impl='torch')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

            if 'gpu' in valid_impls and measure_cython_gpu:
                for timer in ti.reset('cython(gpu)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, impl='gpu')
                        torch.cuda.synchronize()
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

        if True:
            if 'cpu' in valid_impls and measure_cython_cpu:
                for timer in ti.reset('cython(cpu)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, impl='cpu')
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

            if 'py' in valid_impls:
                for timer in ti.reset('numpy(cpu)'):
                    with timer:
                        keep = np_dets.non_max_supression(thresh=thresh, impl='py')
                ydata[ti.label].append(ti.min())
                outputs[ti.label] = ensure_numpy_indices(keep)

        # ----------------------------------

        # Check that all kept boxes do not have more than `threshold` ious
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
            jaccard = len(idxs1 & idxs2) / len(idxs1 | idxs2)
            datas.append((k1, k2, jaccard))

        datas = sorted(datas, key=lambda x: -x[2])
        for k1, k2, jaccard in datas:
            print('    * {:<20}, {:<20}: {:0.4f}'.format(k1, k2, jaccard))

    import kwplot
    kwplot.autompl()

    ydata = ub.dict_subset(ydata, ub.argsort(ub.map_vals(lambda x: x[-1], ydata)))
    kwplot.multi_plot(
        xdata, ydata, xlabel='num boxes', ylabel='seconds',
        # yscale='symlog', xscale='symlog',
    )
    kwplot.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_nms.py  --show
    """
    benchamrk_det_nms()
