import itertools as it
import numpy as np
import ubelt as ub


def test_rle_translate():
    """
    Ensure that RLE translation works on a variety of data
    """
    from kwimage.structs.mask import Mask
    from kwimage.structs.mask import MaskFormat  # NOQA

    SMALL = True

    if SMALL:
        # Only test a restricted number of examples to make things faster for
        # automated tests.
        N, M = 2, 3
        offset_choices = [
            (1, 1), (0, 0), (-1, -1),
        ]
        shape_choices = [
            (4, 4), (N, M), (2, 2),
        ]
    else:
        N, M = 3, 5
        offset_choices = [
            (1, 2), (0, 0), (-1, 0), (0, -1), (-1, -2),
        ]
        shape_choices = [
            (N + 2, M + 2),
            (N, M),
            (N - 2, M - 2),
            (N - 1, M),
            (N, M - 1),
        ]

    # Ensure this works all possible NxM binary images
    pixel_choices = [[0, 1]] * (N * M)
    total = np.prod(list(map(len, pixel_choices)))

    iter_ = it.product(*pixel_choices)
    iter_ = ub.ProgIter(iter_, total=total, desc='testing RLE translate')
    for choice in iter_:
        mask = np.array(choice, dtype=np.uint8).reshape(N, M)
        for offset in offset_choices:
            for shape in shape_choices:
                self1 = Mask.from_mask(mask, offset, shape, 'naive')
                self2 = Mask.from_mask(mask, offset, shape, 'faster')

                m1 = self1.to_c_mask().data
                m2 = self2.to_c_mask().data

                assert np.all(m1 == m2)

                # if False:
                #     import kwimage
                #     encoded = kwimage.encode_run_length(mask, binary=True, order='F')
                #     encoded['size'] = encoded['shape']
                #     self = Mask(encoded, MaskFormat.ARRAY_RLE)
