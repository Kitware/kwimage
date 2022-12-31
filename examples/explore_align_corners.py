"""
Explores the align corners behavior of various kwimage functions and their
underlying cv2 / torch implementations.

TODO:

Handle different settings of align corners in both imresize and warp_affine.
Behaviors should follow:


    +---------------+-----------------------+
    | align_corners | pixels interpretation |
    +---------------+-----------------------+
    | True          | points in a grid      |
    +---------------+-----------------------+
    | False         | areas of 1x1 squares  |
    +---------------+-----------------------+

References:
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://medium.com/@elagwoog/you-might-have-misundertood-the-meaning-of-align-corners-c681d0e38300
    https://user-images.githubusercontent.com/9757500/58150486-c5315900-7c34-11e9-9466-24f2bd431fa4.png


SeeAlso:
    Notes in warp_tensor

"""
import numpy as np
import kwimage


def main():
    D1 = 2
    D2 = 12

    raw_scale = D2 / D1
    new_dsize = (D2, 1)

    results = {}

    img = np.arange(D1)[None, :].astype(np.float32)
    img_imresize = kwimage.imresize(img, dsize=new_dsize, interpolation='linear')
    results['imresize'] = img_imresize

    # Shift and scale (align_corners=True)
    T1 = kwimage.Affine.translate((+0.5, 0))
    T2 = kwimage.Affine.translate((-0.5, 0))
    S = kwimage.Affine.scale((raw_scale, 1))
    S2 = T2 @ S @ T1
    img_shiftscale = kwimage.warp_affine(img, S2, dsize=new_dsize, border_mode='reflect')
    results['aff_shiftscale'] = img_shiftscale

    import torch  # NOQA
    input = torch.from_numpy(img)[None, :, : , None]
    results['torch_ac0'] = torch.nn.functional.interpolate(input, size=new_dsize, mode='bilinear', align_corners=False)[0, :, :, 0].numpy()

    # Pure scaling (align_corners=True)
    S = kwimage.Affine.scale((raw_scale, 1))
    img_rawscale = kwimage.warp_affine(img, S, dsize=new_dsize)
    results['aff_scale'] = img_rawscale
    results['torch_ac1'] = torch.nn.functional.interpolate(input, size=new_dsize, mode='bilinear', align_corners=True)[0, :, :, 0].numpy()

    import pandas as pd
    import ubelt as ub
    import rich
    df = pd.DataFrame(ub.udict(results).map_values(lambda x: x.ravel()))
    rich.print(df.to_string())


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/examples/explore_align_corners.py
    """
    main()
