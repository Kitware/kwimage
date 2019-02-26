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
