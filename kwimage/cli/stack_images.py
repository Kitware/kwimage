#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class StackImagesCLI(scfg.DataConfig):
    """
    Stacks multiple images on disk into a single stacked image.
    """
    __command__ = 'stack_images'

    input_fpaths = scfg.Value(None, nargs='+', position=1, type=str, help=ub.paragraph(
        '''
        A list of input file paths, directories, or glob patterns. If a directory
        is specified, all files with a known image extension are included.
        This functionality requires `kwutil` to resolve glob patterns and directory inputs.
        '''))
    axis = scfg.Value('grid', help=ub.paragraph(
        '''
        The axis to stack over. Use `0` for vertical stacking, `1` for horizontal stacking,
        or `grid` to arrange images in a grid pattern. Default is `grid`.
        '''))
    pad = scfg.Value(None, help=ub.paragraph(
        '''
        The amount of padding (in pixels) to add between stacked images.
        If `None`, no padding is applied.
        '''
    ))
    out = scfg.Value(None, help=ub.paragraph(
        '''
        Path to save the output stacked image. If unspecified, a uses a
        hash-based filename (derived from the input image paths).
        '''
    ))

    @classmethod
    def main(StackImagesCLI, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(module:scriptconfig)
            >>> cmdline = 0
            >>> kwargs = dict(
            >>> )
            >>> main(cmdline=cmdline, **kwargs)
        """
        config = StackImagesCLI.cli(cmdline=cmdline, data=kwargs)
        import kwimage
        print('config = ' + ub.urepr(dict(config), nl=1))
        fpaths = config['input_fpaths']

        try:
            import kwutil
        except ImportError:
            import warnings
            warnings.warn('kwutil is not available; glob patterns and directory input may be limited.')
        else:
            # If available use kwutil to allow for a better
            from kwimage import im_io
            fpaths = kwutil.util_path.coerce_patterned_paths(fpaths, expected_extension=im_io.IMAGE_EXTENSIONS)

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


__cli__ = StackImagesCLI

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/kwimage/cli/stack_images.py
        python -m stack_images
    """
    __cli__.main()
