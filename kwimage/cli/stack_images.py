#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class StackImagesConfig(scfg.DataConfig):
    input_fpaths = scfg.Value(None, nargs='+', position=1, help='input')
    axis = scfg.Value('grid', help='stack axis')
    pad = scfg.Value(None)
    out = scfg.Value(None, help='output path. If unspecified uses a hash')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(module:scriptconfig)
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    import kwimage
    config = StackImagesConfig.legacy(cmdline=cmdline, data=kwargs)
    print('config = ' + ub.urepr(dict(config), nl=1))
    fpaths = config['input_fpaths']
    images = [kwimage.imread(p) for p in ub.ProgIter(fpaths, desc='read images')]

    if config['axis'] == 'grid':
        canvas = kwimage.stack_images_grid(images, pad=config['pad'])
    else:
        canvas = kwimage.stack_images(images, axis=config['axis'], pad=config['pad'])

    out_fpath = config['out']
    if out_fpath is None:
        out_fpath = 'stack_' + ub.hash_data(fpaths)[0:16] + '.png'
    print(f'write to: {out_fpath}')
    kwimage.imwrite(out_fpath, canvas)

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwimage/kwimage/cli/stack_images.py
        python -m stack_images
    """
    main()
