"""
Manages the internal global state of kwimage.

Things like environment variables, are good candidates for living here.

This should be used sparingly.
"""
import os


def _boolean_environ(key):
    value = os.environ.get(key, '').lower()
    TRUTHY_ENVIRONS = {'true', 'on', 'yes', '1'}
    return value in TRUTHY_ENVIRONS


KWIMAGE_DISABLE_WARNINGS = _boolean_environ('KWIMAGE_DISABLE_WARNINGS')
KWIMAGE_DISABLE_TRANSFORM_WARNINGS = KWIMAGE_DISABLE_WARNINGS or _boolean_environ('KWIMAGE_DISABLE_TRANSFORM_WARNINGS')
KWIMAGE_DISABLE_IMPORT_WARNINGS = KWIMAGE_DISABLE_WARNINGS or _boolean_environ('KWIMAGE_DISABLE_IMPORT_WARNINGS')

KWIMAGE_DISABLE_TORCHVISION_NMS = _boolean_environ('KWIMAGE_DISABLE_TORCHVISION_NMS')
KWIMAGE_DISABLE_C_EXTENSIONS = _boolean_environ('KWIMAGE_DISABLE_C_EXTENSIONS')
