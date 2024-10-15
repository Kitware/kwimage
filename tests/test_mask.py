

def test_mask_with_bool_data():
    """
    Ensure that `to_multi_polygon` doesn't break when the mask is a boolean
    type. We mainly just run these to ensure there is no crash.
    """
    from kwimage._backend_info import _have_cv2
    if not _have_cv2():
        import pytest
        pytest.skip('requires cv2')
    import kwimage
    import ubelt as ub
    import numpy as np
    import kwarray
    rng = kwarray.ensure_rng(42134)
    mask_data = rng.rand(32, 32) > 0.5
    mask = kwimage.Mask(mask_data, 'c_mask')

    multi_poly = mask.to_multi_polygon()
    hull = mask.get_convex_hull()
    assert len(multi_poly) > 0
    assert isinstance(hull, np.ndarray)

    if ub.modname_to_modpath('kwimage_ext'):
        try:
            assert mask.iou(mask) > 0.999999
        except NotImplementedError:
            # kwimage_ext is likely not built correctly for this platform in
            # this case.
            ...

    coco_mask = mask.to_coco()
    assert isinstance(coco_mask, dict)
    assert mask.shape == (32, 32)
    assert mask.area > 0

    mask.translate(3)
    if ub.modname_to_modpath('torch'):
        mask.warp(kwimage.Affine.eye(), output_dims='same')
        mask.scale(2.3)
    mask.get_patch()
