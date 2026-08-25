#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Sequence

import kwconf
import ubelt as ub


class CropBorderCLI(kwconf.Config):
    """
    Crop uniform borders from an image.

    This script reads an input image, detects and removes uniform-colored (e.g.
    all white) borders, and then saves the result to the specified destination.
    If no destination is provided, the original image is overwritten.
    """

    __command__ = 'crop_border'
    src: str = kwconf.Value(
        required=True,
        position=1,
        help='Path to the input image.',
    )
    dst: str | None = kwconf.Value(
        None,
        position=2,
        help=ub.paragraph(
            """
        Path to save the cropped image. Defaults to overwriting the input file
        if not specified
        """
        ),
    )

    @classmethod
    def main(
        cls: type[CropBorderCLI],
        argv: bool | Sequence[str] | str = True,
        **kwargs: object,
    ) -> None:
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwimage.cli.crop_border import CropBorderCLI
            >>> CropBorderCLI.main(argv=False, src='input.png')
        """
        import rich
        from rich.markup import escape

        import kwimage
        from kwimage.im_core import crop_border_by_color

        config = cls()
        config.load(data=kwargs, argv=argv, strict=True)
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
