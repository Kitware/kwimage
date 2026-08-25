#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import kwconf
import ubelt as ub


class StackImagesCLI(kwconf.Config):
    """
    Stacks multiple images on disk into a single stacked image.
    """

    __command__ = 'stack_images'

    input_fpaths: list[str] = kwconf.Value(
        required=True,
        nargs='+',
        position=1,
        help=ub.paragraph(
            """
        A list of input file paths, directories, or glob patterns. If a directory
        is specified, all files with a known image extension are included.
        This functionality requires `kwutil` to resolve glob patterns and directory inputs.
        """
        ),
    )
    axis: Literal['grid'] | int = kwconf.Value(
        'grid',
        help=ub.paragraph(
            """
        The axis to stack over. Use `0` for vertical stacking, `1` for horizontal stacking,
        or `grid` to arrange images in a grid pattern. Default is `grid`.
        """
        ),
    )
    pad: int | None = kwconf.Value(
        None,
        help=ub.paragraph(
            """
        The amount of padding (in pixels) to add between stacked images.
        If `None`, no padding is applied.
        """
        ),
    )
    out: str | None = kwconf.Value(
        None,
        help=ub.paragraph(
            """
        Path to save the output stacked image. If unspecified, uses a
        hash-based filename derived from the input image paths.
        """
        ),
    )

    @classmethod
    def main(
        cls: type[StackImagesCLI],
        argv: bool | Sequence[str] | str = True,
        **kwargs: object,
    ) -> None:
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(module:kwconf)
            >>> argv = False
            >>> kwargs = {'input_fpaths': ['a.png', 'b.png']}
            >>> StackImagesCLI.main(argv=argv, **kwargs)
        """
        config = cls()
        config.load(data=kwargs, argv=argv, strict=True)
        import kwimage

        print('config = ' + ub.urepr(dict(config), nl=1))
        fpaths: Sequence[str | PathLike[str]] = config.input_fpaths

        try:
            import kwutil
        except ImportError:
            import warnings

            warnings.warn(
                'kwutil is not available; glob patterns and directory input may be limited.'
            )
        else:
            from kwimage import im_io

            fpaths = kwutil.util_path.coerce_patterned_paths(
                fpaths, expected_extension=im_io.IMAGE_EXTENSIONS
            )

        images = [
            kwimage.imread(p) for p in ub.ProgIter(fpaths, desc='read images')
        ]

        if config.axis == 'grid':
            canvas = kwimage.stack_images_grid(images, pad=config.pad)
        else:
            canvas = kwimage.stack_images(images, axis=config.axis, pad=config.pad)

        out_fpath = config.out
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
