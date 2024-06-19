def bench_argmax():
    """
    Ignore:
        >>> from kwimage.util.util_numpy import *
        >>> arr = np.random.rand(1000)
        >>> num = 3
        >>> axis = None
        >>> import ubelt as ub
        >>> ub.Timerit(100, bestof=10, label='n=1').call(argmaxima, arr, num=1)
        >>> ub.Timerit(100, bestof=10, label='n=2').call(argmaxima, arr, num=2)
        >>> ub.Timerit(100, bestof=10, label='n=25').call(argmaxima, arr, num=25, ordered=True)
        >>> ub.Timerit(100, bestof=10, label='n=25').call(argmaxima, arr, num=25, ordered=False)
        >>> ub.Timerit(100, bestof=10, label='n=25').call(argmaxima, arr, num=990, ordered=False)
        >>> ub.Timerit(100, bestof=10, label='n=25').call(argmaxima, arr, num=1000)

    Ignore:
        >>> ub.Timerit(100, label='argmax').call(np.argmax, arr)
        >>> ub.Timerit(100, label='argmin').call(np.argmin, arr)
        >>> ub.Timerit(100, label='argsort').call(np.argsort, arr)
        >>> ub.Timerit(100, label='argpartition').call(np.argpartition, arr, 1)
        >>> ub.Timerit(100, label='argpartition').call(np.argpartition, arr, 5)
        >>> ub.Timerit(100, label='argpartition').call(np.argpartition, arr, 10)
        >>> ub.Timerit(100, label='argpartition').call(np.argpartition, arr, len(arr) - 5)
        >>> ub.Timerit(100, label='argpartition').call(np.argpartition, arr, len(arr) // 2)
        >>> ub.Timerit(100).call(np.argpartition, arr, 5).print()

    Benchmark:
        >>> # Demonstrates the speedup of using argmaxima
        >>> arr = np.random.rand(128, 128).astype(np.float32)
        >>> num = 10
        >>> import ubelt as ub
        >>> ti = ub.Timerit(1000, bestof=100, verbose=1, unit='us')
        >>> for timer in ti.reset(label='argmaxima-' + str(num)):
        >>>     with timer:
        >>>         idxs1 = argmaxima(arr, num, ordered=False)
        >>> for timer in ti.reset(label='argsort-' + str(num)):
        >>>     with timer:
        >>>         idxs2 = arr.argsort(axis=None)[-num:]
        Timed best=80.561 µs, mean=83.626 ± 3.6 µs for argmaxima-10
        Timed best=1118.433 µs, mean=1191.260 ± 46.9 µs for argsort-10
    """
