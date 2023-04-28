"""

Determine if opencv / numpy release the GIL, if they do then there is
some advantage to using TheadPoolExecutor

"""
import numpy as np
import cv2
import kwimage
from concurrent import futures
from ndsampler.utils import util_futures
import timerit
import ubelt as ub


def numpy_work():
    data = np.random.rand(1000, 1)
    mat = data @ data.T
    u, s, vt = np.linalg.svd(mat)


def opencv_io_work(fpath):
    cv2.imread(fpath)


def opencv_cpu_io_work(fpath):
    data = cv2.imread(fpath)
    cv2.GaussianBlur(data, (3, 3), 2.0)


def main():
    modes = ['serial', 'thread', 'process']
    max_workers = 8
    njobs = 100

    ti = timerit.Timerit(6, bestof=2, verbose=3, unit='ms')
    for mode in modes:
        for timer in ti.reset('time numpy_work ' + mode):
            executor = util_futures.Executor(mode, max_workers=max_workers)
            with executor:
                with timer:
                    fs = [executor.submit(numpy_work) for i in range(njobs)]
                    for f in futures.as_completed(fs):
                        f.result()
    print('ti.measures = {}'.format(ub.urepr(ti.measures, nl=2, precision=4)))

    ti = timerit.Timerit(10, bestof=3, verbose=3, unit='ms')
    fpath = kwimage.grab_test_image_fpath()
    for mode in modes:
        for timer in ti.reset('time opencv_io_work ' + mode):
            executor = util_futures.Executor(mode, max_workers=max_workers)
            with executor:
                with timer:
                    fs = [executor.submit(opencv_io_work, fpath) for i in range(njobs)]
                    for f in futures.as_completed(fs):
                        f.result()
    print('ti.measures = {}'.format(ub.urepr(ti.measures, nl=2, precision=4)))

    ti = timerit.Timerit(10, bestof=3, verbose=3, unit='ms')
    fpath = kwimage.grab_test_image_fpath()
    for mode in modes:
        for timer in ti.reset('time opencv_io_work ' + mode):
            executor = util_futures.Executor(mode, max_workers=max_workers)
            with executor:
                with timer:
                    fs = [executor.submit(opencv_cpu_io_work, fpath) for i in range(njobs)]
                    for f in futures.as_completed(fs):
                        f.result()
    print('ti.measures = {}'.format(ub.urepr(ti.measures, nl=2, precision=4)))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_futures.py
    """
    main()
