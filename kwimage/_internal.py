"""
Manages the internal global state of kwimage.

Things like environment variables, are good candidates for living here.

This should be used sparingly.
"""
import os
import ubelt as ub


def _boolean_environ(key, default=False):
    value = os.environ.get(key, '').lower()
    TRUTHY_ENVIRONS = {'true', 'on', 'yes', '1'}
    FALSY_ENVIRONS = {'true', 'on', 'yes', '1'}
    if value in TRUTHY_ENVIRONS:
        return True
    elif value in FALSY_ENVIRONS:
        return False
    else:
        return default


KWIMAGE_DISABLE_WARNINGS = _boolean_environ('KWIMAGE_DISABLE_WARNINGS')
KWIMAGE_DISABLE_TRANSFORM_WARNINGS = KWIMAGE_DISABLE_WARNINGS or _boolean_environ('KWIMAGE_DISABLE_TRANSFORM_WARNINGS')
KWIMAGE_DISABLE_IMPORT_WARNINGS = KWIMAGE_DISABLE_WARNINGS or _boolean_environ('KWIMAGE_DISABLE_IMPORT_WARNINGS')

KWIMAGE_DISABLE_TORCHVISION_NMS = _boolean_environ('KWIMAGE_DISABLE_TORCHVISION_NMS', default=ub.WIN32)
KWIMAGE_DISABLE_C_EXTENSIONS = _boolean_environ('KWIMAGE_DISABLE_C_EXTENSIONS')
