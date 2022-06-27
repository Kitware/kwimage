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


def schedule_deprecation3(modname, name='?', type='?', migration='',
                          deprecate=None, error=None, remove=None):  # nocover
    """
    Deprecation machinery to help provide users with a smoother transition.

    TODO:
        This can be a new utility to help with generic deprecations

    Example:
        from ubelt._util_deprecated import *  # NOQA
        schedule_deprecation3(
            'ubelt', 'myfunc', 'function', 'do something else',
            error='0.0.0',)

    """
    import ubelt as ub
    import sys
    import warnings
    try:  # nocover
        from packaging.version import parse as Version
    except ImportError:
        from distutils.version import LooseVersion as Version
    module = sys.modules[modname]
    current = Version(module.__version__)

    deprecate_str = ''
    remove_str = ''
    error_str = ''

    if deprecate is not None:
        deprecate = Version(deprecate)
        deprecate_str = ' in {}'.format(deprecate)

    if remove is not None:
        remove = Version(remove)
        remove_str = ' in {}'.format(remove)

    if error is not None:
        error = Version(error)
        error_str = ' in {}'.format(error)

    if deprecate is None or current >= deprecate:
        msg = ub.paragraph(
            '''
            The "{name}" {type} was deprecated{deprecate_str}, will cause
            an error{error_str} and will be removed{remove_str}. The current
            version is {current}. {migration}
            ''').format(**locals()).strip()
        if remove is not None and current >= remove:
            raise AssertionError('Forgot to remove deprecated: ' + msg)
        if error is not None and current >= error:
            raise DeprecationWarning(msg)
        else:
            warnings.warn(msg, DeprecationWarning)
