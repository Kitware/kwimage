
def test_bbox_isect_failure_case():
    import ubelt as ub
    import numpy as np
    import kwimage
    small_box = kwimage.Boxes(np.array([[119.86356797952962, -0.8880716867719497, 119.86816351547256, -0.892667222714885]]), 'ltrb')
    big_box = kwimage.Boxes(np.array([[119.8631329409319, -0.8610904994264246, 119.92872610984104, -0.9266836683730109]]), 'ltrb')

    print('big_box.height = {!r}'.format(big_box.height))
    print('small_box.height = {!r}'.format(small_box.height))

    # Simple fix to negative height boxes
    small_box = small_box.scale((1, -1))
    big_box = big_box.scale((1, -1))

    isect_box = big_box.intersection(small_box)
    print('isect_box       = {!r}'.format(isect_box))

    isect_area = big_box.isect_area(small_box)
    print('isect_area      = {!r}'.format(isect_area))

    sh_box1 = small_box.to_shapely()[0]
    sh_box2 = big_box.to_shapely()[0]
    sh_isect_box = sh_box1.intersection(sh_box2)
    sh_isect_area = sh_isect_box.area
    print('sh_isect_area   = {!r}'.format(sh_isect_area))

    recon_isect_box = kwimage.Boxes.from_shapely(sh_isect_box)
    print('recon_isect_box = {}'.format(ub.repr2(recon_isect_box, nl=1)))
