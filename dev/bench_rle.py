def benchmark_select_rle_conversions():
    """
    Check what is the fastest way to encode an RLE
    """
    import kwimage
    import ubelt as ub
    c_mask = kwimage.Mask.random(shape=(256, 256))
    f_mask = c_mask.to_fortran_mask(copy=True)

    img = c_mask.data

    ti = ub.Timerit(1000, bestof=50, verbose=1)

    for timer in ti.reset('img -> encode_run_length(non-binary)'):
        with timer:
            kwimage.encode_run_length(img, binary=False)

    for timer in ti.reset('img -> encode_run_length(binary)'):
        with timer:
            kwimage.encode_run_length(img, binary=True)

    for timer in ti.reset('c_mask -> to_array_rle'):
        with timer:
            c_mask.to_array_rle()

    for timer in ti.reset('c_mask -> to_bytes_rle'):
        with timer:
            c_mask.to_bytes_rle()

    for timer in ti.reset('f_mask -> to_array_rle'):
        with timer:
            f_mask.to_array_rle()

    for timer in ti.reset('f_mask -> to_bytes_rle'):
        with timer:
            f_mask.to_bytes_rle()


def benchmark_all_mask_conversions():
    import kwimage
    import ubelt as ub

    base_mask = kwimage.Mask.random(shape=(256, 256))
    ti = ub.Timerit(1000, bestof=50, verbose=1, unit='us')

    from kwimage.structs.mask import MaskFormat  # NOQA
    for format1 in MaskFormat.cannonical:
        print('--- {} ---'.format(format1))
        mask1 = base_mask.toformat(format1)
        for format2 in MaskFormat.cannonical:
            for timer in ti.reset('{} -> {}'.format(format1, format2)):
                with timer:
                    mask1.toformat(format2)


def _test_array_from_bytes():
    import kwimage
    base_mask = kwimage.Mask.random(shape=(256, 256), rng=0)
    array_mask = base_mask.to_array_rle()
    bytes_mask = base_mask.to_bytes_rle()

    print(len(array_mask.data['counts']))
    print(len(bytes_mask.data['counts']))

    s = bytes_mask.data['counts']
    w, h = bytes_mask.data['size']

    def _unpack(s):
        import numpy as np
        # verbatim inefficient impl
        cnts = np.empty(len(s), dtype=np.int64)
        p = 0
        m = 0
        while p < len(s) and s[p]:
            x = 0
            k = 0
            more = 1
            while more != 0:
                c = s[p] - 48
                x |= (c & 0x1f) << 5 * k
                more = c & 0x20
                p += 1
                k += 1
                if more == 0 and (c & 0x10):
                    x |= (-1 << 5 * k)
            if m > 2:
                x += np.int64(cnts[m - 2])
            cnts[m] = x
            m += 1

        cnts = cnts[:m]
        return cnts

    print(_unpack(s))
    print(array_mask.data['counts'])
