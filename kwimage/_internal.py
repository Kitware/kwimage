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


def schedule_deprecation3(modname, migration='', name='?', type='?',
                          deprecate=None, error=None, remove=None):  # nocover
    """
    Deprecation machinery to help provide users with a smoother transition.
    """
    import ubelt as ub
    import sys
    from distutils.version import LooseVersion
    import warnings
    module = sys.modules[modname]
    current = LooseVersion(module.__version__)
    deprecate = None if deprecate is None else LooseVersion(deprecate)
    remove = None if remove is None else LooseVersion(remove)
    error = None if error is None else LooseVersion(error)
    if deprecate is None or current >= deprecate:
        if migration is None:
            migration = ''
        msg = ub.paragraph(
            '''
            The "{name}" {type} was deprecated in {deprecate}, will cause
            an error in {error} and will be removed in {remove}. The current
            version is {current}. {migration}
            ''').format(**locals()).strip()
        if remove is not None and current >= remove:
            raise AssertionError('forgot to remove a deprecated function')
        if error is not None and current >= error:
            raise DeprecationWarning(msg)
        else:
            warnings.warn(msg, DeprecationWarning)
