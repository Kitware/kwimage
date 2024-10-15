"""
Helpers that may be required across different backends.
"""


def _coerce_warp_dsize_inputs(dsize, input_dsize, transform, require_warped_info=False):
    """
    Given a warp operation, we will often need to preallocate size for the
    destination canvas. This may be specified by the user, but it is helpful to
    be able to precalculate it based on the transform and a specified
    heuristic.

    Ignore:
        import kwimage
        from kwimage._common import *  # NOQA
        from kwimage._common import _coerce_warp_dsize_inputs
        input_dsize = (128, 128)
        dsize = (128, 128)
        transform = kwimage.Affine.coerce({'scale': 0.2})

        basis = {
            'input_dsize': [
                (128, 128),
            ],
            'dsize': [
                (128, 128),
                'auto',
                'content',
            ],
            'transform': list(map(kwimage.Affine.coerce, [
                {'scale': 0.5},
                {'scale': 2.0},
                {'scale': 2.0, 'offset': -100},
            ])),
        }

        import rich
        for kwargs in ub.named_product(**basis):
            info = _coerce_warp_dsize_inputs(**kwargs)
            row = {'params': kwargs, 'result': info}
            rich.print(f'row = {ub.urepr(row, nl=2)}')
            ...
    """
    import numpy as np
    import kwimage

    w, h = input_dsize
    dsize

    # FIXME: we may need to modify this based on the origin convention for
    # correctness.

    if isinstance(dsize, str) or require_warped_info:
        # calculate dimensions needed for auto/max/try_large_warp
        input_box = kwimage.Boxes(np.array([[0, 0, w, h]]), 'xywh')
        warped_box = input_box.warp(transform)
        max_dsize = tuple(map(int, warped_box.to_xywh().quantize().data[0, 2:4]))
        new_origin = warped_box.to_ltrb().data[0, 0:2]
        if 0:
            # import rich
            # print('---------')
            # rich.print(f'box={box}')
            # rich.print(transform)
            # rich.print(f'warped_box={warped_box}')
            # TODO: should we enable this?  This seems to break if there is an
            # axis or orientation flip because the Boxes was designed with
            # slices in mind, so we need to maintain orientation information to
            # handle this correctly.
            warped_box._ensure_nonnegative_extent(inplace=True)
            # rich.print(f'warped_box={warped_box}')
            warped_box = warped_box.to_ltrb()
            # rich.print(f'warped_box={warped_box}')
            warped_box = warped_box.to_xywh().quantize()
            # rich.print(f'warped_box={warped_box}')
            max_dsize = tuple(map(int, warped_box.data[0, 2:4]))
            # print('warped_box = {}'.format(ub.urepr(warped_box, nl=1)))
            # print('max_dsize = {}'.format(ub.urepr(max_dsize, nl=1)))
    else:
        max_dsize = None
        new_origin = None

    transform_ = transform

    if dsize is None:
        # If unspecified, leave the canvas size unchanged
        dsize = (w, h)
    elif isinstance(dsize, str):
        # Handle special "auto-compute" dsize keys
        if dsize in {'positive', 'auto'}:
            quantized_warped_box = warped_box.to_ltrb().quantize()
            dsize = tuple(map(int, quantized_warped_box.data[0, 2:4]))
            if 0:
                affine_params = None
                # rich.print('affine_params = {}'.format(ub.urepr(affine_params, nl=1)))
                if affine_params is None:
                    affine_params = transform_.decompose()
                sx, sy = affine_params['scale']
                new_w, new_h = dsize
                if sx < 0:
                    new_w += 1
                if sy < 0:
                    new_h += 1
                dsize = (new_w, new_h)
        elif dsize in {'content', 'max'}:
            dsize = max_dsize
            transform_ = kwimage.Affine.translate(-new_origin) @ transform
            new_origin = np.array([0, 0])
        else:
            raise KeyError('Unknown dsize={}'.format(dsize))

    dsize_inputs = {
        'max_dsize': max_dsize,
        'new_origin': new_origin,
        'transform': transform_,
        'dsize': dsize,
    }
    return dsize_inputs
