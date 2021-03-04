# -*- coding: utf-8 -*-
"""
Generic Non-Maximum Suppression API with efficient backend implementations
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import ubelt as ub
import warnings
import kwarray

try:
    import torch
except Exception:
    torch = None


def daq_spatial_nms(ltrb, scores, diameter, thresh, max_depth=6,
                    stop_size=2048, recsize=2048, impl='auto', device_id=None):
    """
    Divide and conquor speedup non-max-supression algorithm for when bboxes
    have a known max size

    Args:
        ltrb (ndarray): boxes in (tlx, tly, brx, bry) format

        scores (ndarray): scores of each box

        diameter (int or Tuple[int, int]): Distance from split point to
            consider rectification. If specified as an integer, then number
            is used for both height and width. If specified as a tuple, then
            dims are assumed to be in [height, width] format.

        thresh (float): iou threshold. Boxes are removed if they overlap
            greater than this threshold. 0 is the most strict, resulting in the
            fewest boxes, and 1 is the most permissive resulting in the most.

        max_depth (int): maximum number of times we can divide and conquor

        stop_size (int): number of boxes that triggers full NMS computation

        recsize (int): number of boxes that triggers full NMS recombination

        impl (str): algorithm to use

    LookInfo:
        # Didn't read yet but it seems similar
        http://www.cyberneum.de/fileadmin/user_upload/files/publications/CVPR2010-Lampert_[0].pdf

        https://www.researchgate.net/publication/220929789_Efficient_Non-Maximum_Suppression

        # This seems very similar
        https://projet.liris.cnrs.fr/m2disco/pub/Congres/2006-ICPR/DATA/C03_0406.PDF

    Example:
        >>> import kwimage
        >>> # Make a bunch of boxes with the same width and height
        >>> #boxes = kwimage.Boxes.random(230397, scale=1000, format='cxywh')
        >>> boxes = kwimage.Boxes.random(237, scale=1000, format='cxywh')
        >>> boxes.data.T[2] = 10
        >>> boxes.data.T[3] = 10
        >>> #
        >>> ltrb = boxes.to_ltrb().data.astype(np.float32)
        >>> scores = np.arange(0, len(ltrb)).astype(np.float32)
        >>> #
        >>> n_megabytes = (ltrb.size * ltrb.dtype.itemsize) / (2 ** 20)
        >>> print('n_megabytes = {!r}'.format(n_megabytes))
        >>> #
        >>> thresh = iou_thresh = 0.01
        >>> impl = 'auto'
        >>> max_depth = 20
        >>> diameter = 10
        >>> stop_size = 2000
        >>> recsize = 500
        >>> #
        >>> import ubelt as ub
        >>> #
        >>> with ub.Timer(label='daq'):
        >>>     keep1 = daq_spatial_nms(ltrb, scores,
        >>>         diameter=diameter, thresh=thresh, max_depth=max_depth,
        >>>         stop_size=stop_size, recsize=recsize, impl=impl)
        >>> #
        >>> with ub.Timer(label='full'):
        >>>     keep2 = non_max_supression(ltrb, scores,
        >>>         thresh=thresh, impl=impl)
        >>> #
        >>> # Due to the greedy nature of the algorithm, there will be slight
        >>> # differences in results, but they will be mostly similar.
        >>> similarity = len(set(keep1) & set(keep2)) / len(set(keep1) | set(keep2))
        >>> print('similarity = {!r}'.format(similarity))
    """
    def _rectify(ltrb, both_keep, needs_rectify):
        if len(needs_rectify) == 0:
            keep = sorted(both_keep)
        else:
            nr_arr = np.array(sorted(needs_rectify))
            nr = needs_rectify
            bk = set(both_keep)
            rectified_keep = non_max_supression(
                ltrb[nr_arr], scores[nr_arr], thresh=thresh,
                impl=impl, device_id=device_id)
            rk = set(nr_arr[rectified_keep])
            keep = sorted((bk - nr) | rk)
        return keep

    def _recurse(ltrb, scores, dim, depth, diameter_wh):
        """
        Args:
            dim (int): flips between 0 and 1
            depth (int): recursion depth
        """
        # print('recurse')
        n_boxes = len(ltrb)
        if depth >= max_depth or n_boxes < stop_size:
            # print('n_boxes = {!r}'.format(n_boxes))
            # print('depth = {!r}'.format(depth))
            # print('stop')
            keep = non_max_supression(ltrb, scores, thresh=thresh, impl=impl)
            both_keep = sorted(keep)
            needs_rectify = set()
        else:
            # Break up the NMS into two subproblems.
            middle = np.median(ltrb.T[dim])
            left_flags = ltrb.T[dim] < middle
            right_flags = ~left_flags

            left_idxs = np.where(left_flags)[0]
            right_idxs = np.where(right_flags)[0]

            left_scores = scores[left_idxs]
            left_ltrb = ltrb[left_idxs]

            right_scores = scores[right_idxs]
            right_ltrb = ltrb[right_idxs]

            next_depth = depth + 1
            next_dim = 1 - dim

            # Solve each subproblem
            left_keep_, lrec_ = _recurse(
                left_ltrb, left_scores, depth=next_depth, dim=next_dim,
                diameter_wh=diameter_wh)

            right_keep_, rrec_ = _recurse(
                right_ltrb, right_scores, depth=next_depth, dim=next_dim,
                diameter_wh=diameter_wh)

            # Recombine the results (note that because we have a diameter_wh,
            # we have to check less results)
            rrec = set(right_idxs[sorted(rrec_)])
            lrec = set(left_idxs[sorted(lrec_)])

            left_keep = left_idxs[left_keep_]
            right_keep = right_idxs[right_keep_]

            both_keep = np.hstack([left_keep, right_keep])
            both_keep.sort()

            dist_to_middle = np.abs(ltrb[both_keep].T[dim] - middle)

            # Find all surviving boxes that are close to the midpoint.  We will
            # need to recheck these because they may overlap, but they also may
            # have been split into different subproblems.
            rectify_flags = dist_to_middle < diameter_wh[dim]

            needs_rectify = set(both_keep[rectify_flags])
            needs_rectify.update(rrec)
            needs_rectify.update(lrec)

            nrec = len(needs_rectify)
            # print('nrec = {!r}'.format(nrec))
            if nrec > recsize:
                both_keep = _rectify(ltrb, both_keep, needs_rectify)
                needs_rectify = set()
        return both_keep, needs_rectify

    if not ub.iterable(diameter):
        diameter_wh = [diameter, diameter]
    else:
        diameter_wh = diameter[::-1]

    depth = 0
    dim = 0
    both_keep, needs_rectify = _recurse(ltrb, scores, dim=dim, depth=depth,
                                        diameter_wh=diameter_wh)
    keep = _rectify(ltrb, both_keep, needs_rectify)
    return keep


_impls = None


class _NMS_Impls():
    # TODO: could make this prettier
    def __init__(self):
        self._funcs = None

    def _lazy_init(self):
        _funcs = {}

        TRUTHY_ENVIRONS = {'true', 'on', 'yes', '1'}
        DISABLE_C_EXTENSIONS = os.environ.get(
            'KWIMAGE_DISABLE_C_EXTENSIONS', '').lower() in TRUTHY_ENVIRONS
        DISABLE_TORCHVISION_NMS = os.environ.get(
            'KWIMAGE_DISABLE_TORCHVISION_NMS', '').lower() in TRUTHY_ENVIRONS

        # These are pure python and should always be available
        from kwimage.algo._nms_backend import py_nms
        from kwimage.algo._nms_backend import torch_nms
        _funcs['numpy'] = py_nms.py_nms

        if torch is not None:
            _funcs['torch'] = torch_nms.torch_nms

            if not DISABLE_TORCHVISION_NMS:
                # The torchvision _C libraray may cause segfaults, which is
                # why we have an option to disable even trying it
                try:
                    # TODO: torchvision impl might be the best, need to test
                    from torchvision import _C as C  # NOQA
                    import torchvision
                    _funcs['torchvision'] = torchvision.ops.nms
                except (ImportError, UnicodeDecodeError) as ex:
                    warnings.warn(
                        'optional torchvision C nms is not available: {}'.format(
                            str(ex)))

        try:
            if not DISABLE_C_EXTENSIONS:
                from kwimage.algo._nms_backend import cpu_nms
                _funcs['cython_cpu'] = cpu_nms.cpu_nms
        except Exception as ex:
            warnings.warn(
                'optional cpu_nms is not available: {}'.format(str(ex)))
        try:
            if not DISABLE_C_EXTENSIONS:
                if torch is not None and torch.cuda.is_available():
                    from kwimage.algo._nms_backend import gpu_nms
                    _funcs['cython_gpu'] = gpu_nms.gpu_nms
                    # NOTE: GPU is not the fastests on all systems.
                    # See the benchmarks for more info.
                    # ~/code/kwimage/dev/bench_nms.py
        except Exception as ex:
            warnings.warn
            ('optional gpu_nms is not available: {}'.format(str(ex)))
        self._funcs = _funcs
        self._valid = frozenset(_impls._funcs.keys())


_impls = _NMS_Impls()


def available_nms_impls():
    """
    List available values for the `impl` kwarg of `non_max_supression`

    CommandLine:
        xdoctest -m kwimage.algo.algo_nms available_nms_impls

    Example:
        >>> impls = available_nms_impls()
        >>> assert 'numpy' in impls
        >>> print('impls = {!r}'.format(impls))
    """
    if not _impls._funcs:
        _impls._lazy_init()
    return list(_impls._funcs.keys())


# @ub.memoize
def _heuristic_auto_nms_impl(code, num, valid=None):
    """
    Defined with help from ``~/code/kwimage/dev/bench_nms.py``

    Args:
        code (str): text that indicates which type of data you have
            tensor0 is a tensor on a cuda device, tensor is on the cpu, and
            numpy is a ndarray.

        num (int): number of boxes you have to supress.

        valid (List[str]): the list of valid implementations, an error will be
            raised if heuristic preferences do not intersect with this list.

    Ignore:
        _impls._funcs
        valid_pref = ub.oset(preference) & set(_impls._funcs.keys())
        python ~/code/kwimage/dev/bench_nms.py --show --small-boxes --thresh=0.6
    """
    if code not in {'tensor0', 'tensor', 'ndarray'}:
        raise KeyError(code)

    if num <= 10:
        if code == 'tensor0':
            # dict(cython_cpu=4118.4, torchvision=3042.5, cython_gpu=2244.4, torch=841.9)
            preference = ['cython_cpu', 'torchvision', 'cython_gpu', 'torch']
        if code == 'tensor':
            # dict(torchvision=5857.1, cython_gpu=3058.1)
            preference = ['torchvision', 'cython_gpu', 'torch', 'numpy']
        if code == 'ndarray':
            # dict(cython_cpu=12226.1, numpy=7759.1, cython_gpu=3679.0, torch=1786.2)
            preference = ['cython_cpu', 'numpy', 'cython_gpu', 'torch']
    elif num <= 100:
        if code == 'tensor0':
            # dict(cython_cpu=4160.7, torchvision=3089.9, cython_gpu=2261.8, torch=846.8)
            preference = ['cython_cpu', 'torchvision', 'cython_gpu', 'torch', 'numpy']
        if code == 'tensor':
            # dict(torchvision=5875.3, cython_gpu=3076.9)
            preference = ['torchvision', 'cython_gpu', 'torch', 'numpy']
        if code == 'ndarray':
            # dict(cython_cpu=12256.7, cython_gpu=3702.9, numpy=2311.3, torch=1738.0)
            preference = ['cython_cpu', 'cython_gpu', 'numpy', 'torch']
    elif num <= 200:
        if code == 'tensor0':
            # dict(cython_cpu=3460.8, torchvision=2912.9, cython_gpu=2125.2, torch=782.4)
            preference = ['cython_cpu', 'torchvision', 'cython_gpu', 'torch']
        if code == 'tensor':
            # dict(torchvision=3394.6, cython_gpu=2641.2)
            preference = ['torchvision', 'cython_gpu', 'torch', 'numpy']
        if code == 'ndarray':
            # dict(cython_cpu=8220.6, cython_gpu=3114.5, torch=1240.7, numpy=309.5)
            preference = ['cython_cpu', 'cython_gpu', 'torch', 'numpy']
    elif num <= 300:
        if code == 'tensor0':
            # dict(torchvision=2647.1, cython_cpu=2264.9, cython_gpu=1915.5, torch=672.0)
            preference = ['torchvision', 'cython_cpu', 'cython_gpu', 'torch']
        if code == 'tensor':
            # dict(cython_gpu=2496.9, torchvision=1781.1)
            preference = ['cython_gpu', 'torchvision', 'torch', 'numpy']
        if code == 'ndarray':
            # dict(cython_cpu=4085.6, cython_gpu=2944.4, torch=799.8, numpy=173.0)
            preference = ['cython_cpu', 'cython_gpu', 'torch', 'numpy']
    else:
        if code == 'tensor0':
            # dict(torchvision=2585.5, cython_gpu=1868.7, cython_cpu=1650.6, torch=623.1)
            preference = ['torchvision', 'cython_gpu', 'cython_cpu', 'torch']
        if code == 'tensor':
            # dict(cython_gpu=2463.1, torchvision=1126.2)
            preference = ['cython_gpu', 'torchvision', 'torch', 'numpy']
        if code == 'ndarray':
            # dict(cython_gpu=2880.2, cython_cpu=2432.5, torch=511.9, numpy=114.0)
            preference = ['cython_gpu', 'cython_cpu', 'torch', 'numpy']

    if valid:
        valid_pref = ub.oset(preference) & valid
    else:
        valid_pref = preference

    if not valid_pref:
        raise Exception(
            'no valid nms algo: code={}, num={}, valid={}, preference={}, valid_pref={}'.format(
                code, num, valid, preference, valid_pref))

    impl = valid_pref[0]
    return impl


def non_max_supression(ltrb, scores, thresh, bias=0.0, classes=None,
                       impl='auto', device_id=None):
    """
    Non-Maximum Suppression - remove redundant bounding boxes

    Args:
        ltrb (ndarray[float32]): Nx4 boxes in ltrb format

        scores (ndarray[float32]): score for each bbox

        thresh (float): iou threshold.
            Boxes are removed if they overlap greater than this threshold
            (i.e. Boxes are removed if iou > threshold).
            Thresh = 0 is the most strict, resulting in the fewest boxes, and 1
            is the most permissive resulting in the most.

        bias (float): bias for iou computation either 0 or 1

        classes (ndarray[int64] or None): integer classes.
            If specified NMS is done on a perclass basis.

        impl (str): implementation can be "auto", "python", "cython_cpu",
            "gpu", "torch", or "torchvision".

        device_id (int): used if impl is gpu, device id to work on. If not
            specified `torch.cuda.current_device()` is used.

    Notes:
        Using impl='cython_gpu' may result in an CUDA memory error that is not exposed
        to the python processes. In other words your program will hard crash if
        impl='cython_gpu', and you feed it too many bounding boxes. Ideally this will
        be fixed in the future.

    References:
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx <- TODO

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/algo/algo_nms.py non_max_supression

    Example:
        >>> from kwimage.algo.algo_nms import *
        >>> from kwimage.algo.algo_nms import _impls
        >>> ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .5, .9, .1])
        >>> keep = non_max_supression(ltrb, scores, thresh=0.5, impl='numpy')
        >>> print('keep = {!r}'.format(keep))
        >>> assert keep == [2, 1, 3]
        >>> thresh = 0.0
        >>> non_max_supression(ltrb, scores, thresh, impl='numpy')
        >>> if 'numpy' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='numpy')
        >>>     assert list(keep) == [2, 1]
        >>> if 'cython_cpu' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='cython_cpu')
        >>>     assert list(keep) == [2, 1]
        >>> if 'cython_gpu' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='cython_gpu')
        >>>     assert list(keep) == [2, 1]
        >>> if 'torch' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='torch')
        >>>     assert set(keep.tolist()) == {2, 1}
        >>> if 'torchvision' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='torchvision')  # note torchvision has no bias
        >>>     assert list(keep) == [2]
        >>> thresh = 1.0
        >>> if 'numpy' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='numpy')
        >>>     assert list(keep) == [2, 1, 3, 0]
        >>> if 'cython_cpu' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='cython_cpu')
        >>>     assert list(keep) == [2, 1, 3, 0]
        >>> if 'cython_gpu' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='cython_gpu')
        >>>     assert list(keep) == [2, 1, 3, 0]
        >>> if 'torch' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='torch')
        >>>     assert set(keep.tolist()) == {2, 1, 3, 0}
        >>> if 'torchvision' in available_nms_impls():
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl='torchvision')  # note torchvision has no bias
        >>>     assert set(kwarray.ArrayAPI.tolist(keep)) == {2, 1, 3, 0}

    Example:
        >>> import ubelt as ub
        >>> ltrb = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> scores = np.linspace(0, 1, len(ltrb))
        >>> thresh = .2
        >>> solutions = {}
        >>> if not _impls._funcs:
        >>>     _impls._lazy_init()
        >>> for impl in _impls._funcs:
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl=impl)
        >>>     solutions[impl] = sorted(keep)
        >>> assert 'numpy' in solutions
        >>> print('solutions = {}'.format(ub.repr2(solutions, nl=1)))
        >>> assert ub.allsame(solutions.values())

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/algo/algo_nms.py non_max_supression

    Example:
        >>> import ubelt as ub
        >>> # Check that zero-area boxes are ok
        >>> ltrb = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [0, 0, 0, 0],
        >>>     [10, 10, 10, 10],
        >>> ], dtype=np.float32)
        >>> scores = np.array([1, 2, 3], dtype=np.float32)
        >>> thresh = .2
        >>> solutions = {}
        >>> if not _impls._funcs:
        >>>     _impls._lazy_init()
        >>> for impl in _impls._funcs:
        >>>     keep = non_max_supression(ltrb, scores, thresh, impl=impl)
        >>>     solutions[impl] = sorted(keep)
        >>> assert 'numpy' in solutions
        >>> print('solutions = {}'.format(ub.repr2(solutions, nl=1)))
        >>> assert ub.allsame(solutions.values())
    """

    if impl == 'cpu':
        import warnings
        warnings.warn(
            'impl="cpu" is deprecated use impl="cython_cpu" instead',
            DeprecationWarning)
        impl = 'cython_impl'
    elif impl == 'gpu':
        import warnings
        warnings.warn(
            'impl="gpu" is deprecated use impl="cython_gpu" instead',
            DeprecationWarning)
        impl = 'cython_gpu'
    elif impl == 'py':
        import warnings
        warnings.warn(
            'impl="py" is deprecated use impl="numpy" instead',
            DeprecationWarning)
        impl = 'numpy'

    if not _impls._funcs:
        _impls._lazy_init()

    if ltrb.shape[0] == 0:
        return []

    if impl == 'auto':
        is_tensor = torch is not None and torch.is_tensor(ltrb)
        num = len(ltrb)
        if is_tensor:
            if ltrb.device.type == 'cuda':
                code = 'tensor0'
            else:
                code = 'tensor'
        else:
            code = 'ndarray'
        valid = _impls._valid
        impl = _heuristic_auto_nms_impl(code, num, valid)
        # print('impl._valid = {!r}'.format(_impls._valid))
        # print('impl = {!r}'.format(impl))

    elif ub.iterable(impl):
        # if impl is iterable, it is a preference order
        found = False
        for item in impl:
            if item in _impls._funcs:
                impl = item
                found = True
                break
        if not found:
            raise KeyError('Unknown impls={}'.format(impl))

    if classes is not None:
        keep = []
        for idxs in ub.group_items(range(len(classes)), classes).values():
            # cls_ltrb = ltrb.take(idxs, axis=0)
            # cls_scores = scores.take(idxs, axis=0)
            cls_ltrb = ltrb[idxs]
            cls_scores = scores[idxs]
            cls_keep = non_max_supression(cls_ltrb, cls_scores, thresh=thresh,
                                          bias=bias, impl=impl)
            keep.extend(list(ub.take(idxs, cls_keep)))
        return keep
    else:

        if impl == 'numpy':
            api = kwarray.ArrayAPI.coerce(ltrb)
            ltrb = api.numpy(ltrb)
            scores = api.numpy(scores)
            func = _impls._funcs['numpy']
            keep = func(ltrb, scores, thresh, bias=float(bias))
        elif impl == 'torch' or impl == 'torchvision':
            api = kwarray.ArrayAPI.coerce(ltrb)
            ltrb = api.tensor(ltrb).float()
            scores = api.tensor(scores).float()
            # Default output of torch impl is a mask
            if impl == 'torchvision':
                # if bias != 1:
                #     warnings.warn('torchvision only supports bias==1')
                func = _impls._funcs['torchvision']
                # Torchvision returns indices
                keep = func(ltrb, scores, iou_threshold=thresh)
            else:
                func = _impls._funcs['torch']
                flags = func(ltrb, scores, thresh=thresh, bias=float(bias))
                keep = torch.nonzero(flags).view(-1)

            # Ensure than input type is the same as output type
            keep = api.numpy(keep)
        else:
            # TODO: it would be nice to be able to pass torch tensors here
            nms = _impls._funcs[impl]
            ltrb = kwarray.ArrayAPI.numpy(ltrb)
            scores = kwarray.ArrayAPI.numpy(scores)
            ltrb = ltrb.astype(np.float32)
            scores = scores.astype(np.float32)
            if impl == 'cython_gpu':
                # TODO: if the data is already on a torch GPU can we just
                # use it?
                # HACK: we should parameterize which device is used
                if device_id is None:
                    device_id = torch.cuda.current_device()
                keep = nms(ltrb, scores, float(thresh), bias=float(bias),
                           device_id=device_id)
            elif impl == 'cython_cpu':
                keep = nms(ltrb, scores, float(thresh), bias=float(bias))
            else:
                raise KeyError(impl)
        return keep


# TODO: add soft nms bindings


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.algo.algo_nms
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
