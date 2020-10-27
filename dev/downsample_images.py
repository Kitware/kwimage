
def resize_all():
    in_dpath = '/media/joncrall/raid/Pictures/Bezoar'
    out_dpath = '/media/joncrall/raid/Pictures/BezoarSmall'

    IMAGE_EXTENSIONS = (
        '.bmp', '.pgm', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
        '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif', '.r0',
        '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
    )
    import os
    from os.path import join, relpath
    import kwimage
    recursive = False

    max_dim = 2000

    tasks = []
    for root, ds, fs in os.walk(in_dpath):
        if not recursive:
            ds[:] = []

        valid_fs = [f for f in fs if f.lower().endswith(IMAGE_EXTENSIONS)]
        valid_fpaths = [join(root, fname) for fname in valid_fs]

        for in_fpath in valid_fpaths:
            out_fpath = join(out_dpath, relpath(in_fpath, in_dpath))
            tasks.append((in_fpath, out_fpath))

    import ubelt as ub
    for in_fpath, out_fpath in ub.ProgIter(tasks):
        in_imdata = kwimage.imread(in_fpath)
        out_imdata = kwimage.imresize(in_imdata, max_dim=max_dim)
        kwimage.imwrite(out_fpath, out_imdata)
