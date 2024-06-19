import kwimage
import torch
import ubelt as ub
import numpy as np
import itertools as it
from functools import partial


def bench_bbox_iou_method():
    """
    On my system the torch impl was fastest (when the data was on the GPU).
    """
    from kwimage.structs.boxes import _box_ious_torch, _box_ious_py, _bbox_ious_c

    ydata = ub.ddict(list)
    xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, 4000]
    bias = 0

    if _bbox_ious_c is None:
        print('CYTHON IMPLEMENATION IS NOT AVAILABLE')

    for num in xdata:
        results = {}

        # Setup Timer
        N = max(20, int(1000 / num))
        ti = ub.Timerit(N, bestof=10)

        # Setup input dat
        boxes1 = kwimage.Boxes.random(num, scale=10.0, rng=0, format='ltrb')
        boxes2 = kwimage.Boxes.random(num + 1, scale=10.0, rng=1, format='ltrb')

        ltrb1 = boxes1.tensor().data
        ltrb2 = boxes2.tensor().data
        for timer in ti.reset('iou-torch-cpu'):
            with timer:
                out = _box_ious_torch(ltrb1, ltrb2, bias)
        results[ti.label] = out.data.cpu().numpy()
        ydata[ti.label].append(ti.mean())

        gpu = torch.device(0)
        ltrb1 = boxes1.tensor().data.to(gpu)
        ltrb2 = boxes2.tensor().data.to(gpu)
        for timer in ti.reset('iou-torch-gpu'):
            with timer:
                out = _box_ious_torch(ltrb1, ltrb2, bias)
                torch.cuda.synchronize()
        results[ti.label] = out.data.cpu().numpy()
        ydata[ti.label].append(ti.mean())

        ltrb1 = boxes1.numpy().data
        ltrb2 = boxes2.numpy().data
        for timer in ti.reset('iou-numpy'):
            with timer:
                out = _box_ious_py(ltrb1, ltrb2, bias)
        results[ti.label] = out
        ydata[ti.label].append(ti.mean())

        if _bbox_ious_c:
            ltrb1 = boxes1.numpy().data.astype(np.float32)
            ltrb2 = boxes2.numpy().data.astype(np.float32)
            for timer in ti.reset('iou-cython'):
                with timer:
                    out = _bbox_ious_c(ltrb1, ltrb2, bias)
            results[ti.label] = out
            ydata[ti.label].append(ti.mean())

        eq = partial(np.allclose, atol=1e-07)
        passed = ub.allsame(results.values(), eq)
        if passed:
            print('All methods produced the same answer for num={}'.format(num))
        else:
            for k1, k2 in it.combinations(results.keys(), 2):
                v1 = results[k1]
                v2 = results[k2]
                if eq(v1, v2):
                    print('pass: {} == {}'.format(k1, k2))
                else:
                    diff = np.abs(v1 - v2)
                    print('FAIL: {} != {}: diff(max={}, mean={}, sum={})'.format(
                        k1, k2, diff.max(), diff.mean(), diff.sum()
                    ))

            raise AssertionError('different methods report different results')

        print('num = {!r}'.format(num))
        print('ti.measures = {}'.format(ub.urepr(
            ub.map_vals(ub.sorted_vals, ti.measures), align=':',
            nl=2, precision=6)))

    import kwplot
    kwplot.autoplt()
    kwplot.multi_plot(xdata, ydata, xlabel='num boxes', ylabel='seconds')
    kwplot.show_if_requested()
    # plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_bbox.py --show
    """
    bench_bbox_iou_method()
