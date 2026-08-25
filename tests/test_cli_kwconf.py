from __future__ import annotations

import pytest

import kwconf

from kwimage.cli.__main__ import KwimageCLI
from kwimage.cli.crop_border import CropBorderCLI
from kwimage.cli.stack_images import StackImagesCLI


def test_kwimage_cli_configs_use_kwconf():
    assert issubclass(StackImagesCLI, kwconf.Config)
    assert issubclass(CropBorderCLI, kwconf.Config)
    assert issubclass(KwimageCLI, kwconf.ModalCLI)


def test_stack_images_cli_typed_parsing():
    config = StackImagesCLI.cli(
        argv=['a.png', 'b.png', '--axis=1', '--pad=3', '--out=stacked.png']
    )
    assert config.input_fpaths == ['a.png', 'b.png']
    assert config.axis == 1
    assert config.pad == 3
    assert config.out == 'stacked.png'


def test_crop_border_cli_typed_parsing():
    config = CropBorderCLI.cli(argv=['input.png', 'output.png'])
    assert config.src == 'input.png'
    assert config.dst == 'output.png'


@pytest.mark.parametrize('cli_cls', [StackImagesCLI, CropBorderCLI])
def test_kwimage_cli_required_positionals(cli_cls):
    with pytest.raises(ValueError, match='Required variable'):
        cli_cls.cli(argv=[])


def test_kwimage_modal_contains_commands():
    parser = KwimageCLI(version='test').argparse()
    help_text = parser.format_help()
    assert 'stack_images' in help_text
    assert 'crop_border' in help_text
