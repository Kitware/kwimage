#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

from collections.abc import Sequence

import kwconf

from kwimage.cli.crop_border import CropBorderCLI
from kwimage.cli.stack_images import StackImagesCLI


class KwimageCLI(kwconf.ModalCLI):
    """The Kitware Image CLI."""

    stack_images = StackImagesCLI
    crop_border = CropBorderCLI


def main(cmdline: bool | Sequence[str] | None = True) -> int:
    import kwimage

    if cmdline is True:
        argv = None
    elif cmdline is False:
        argv = []
    else:
        argv = cmdline

    modal = KwimageCLI(version=kwimage.__version__)
    result: object = modal.run(argv=argv)
    if isinstance(result, int):
        return result
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
