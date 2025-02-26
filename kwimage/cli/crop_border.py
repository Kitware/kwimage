#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CropBorderCLI(scfg.DataConfig):
    """
    Crop uniform borders from an image.

    This script reads an input image, detects and removes uniform-colored (e.g.
    all white) borders, and then saves the result to the specified destination.
    If no destination is provided, the original image is overwritten.
    """
    __command__ = 'crop_border'
    src = scfg.Value(None, position=1, help="Path to the input image.")
    dst = scfg.Value(None, position=2, help=ub.paragraph(
        '''
        Path to save the cropped image. Defaults to overwriting the input file
        if not specified
        '''))

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwimage.cli.crop_border import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = CropBorderCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        import kwimage
        from rich.markup import escape
        from kwimage.im_core import crop_border_by_color
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        src_fpath = ub.Path(config.src)
        if config.dst is None:
            dst_fpath = src_fpath
        else:
            dst_fpath = ub.Path(config.dst)
        imdata = kwimage.imread(src_fpath)
        imdata = crop_border_by_color(imdata)
        kwimage.imwrite(dst_fpath, imdata)


__cli__ = CropBorderCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwimage/kwimage/cli/crop_border.py
        python -m kwimage.cli.crop_border
    """
    __cli__.main()
