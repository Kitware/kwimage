"""
Helpers to query information about available backends
"""

try:
    from functools import cache
except ImportError:
    from ubelt import memoize as cache


@cache
def _have_turbojpg():
    """
    pip install PyTurboJPEG

    """
    try:
        import turbojpeg  # NOQA

        turbojpeg.TurboJPEG()
    except Exception:
        return False
    else:
        return True


@cache
def _have_gdal():
    try:
        from osgeo import gdal  # NOQA
    except Exception:
        return False
    else:
        return True


@cache
def _have_cv2():
    try:
        import cv2  # NOQA
    except Exception:
        return False
    else:
        return True


@cache
def _default_backend():
    """
    Define the default backend for simple cases.
    In kwimage < 0.11.0, this was always cv2, but now cv2 is optional, so we
    will fallback to skimage (or is PIL a better option?)
    """
    if _have_cv2():
        return 'cv2'
    else:
        return 'skimage'


def _iter_exception_chain(ex):
    """
    Walk explicit / implicit exception chains.

    Useful when a package catches a useful low-level ImportError and then
    raises a misleading higher-level ModuleNotFoundError.
    """
    seen = set()
    while ex is not None and id(ex) not in seen:
        yield ex
        seen.add(id(ex))
        ex = ex.__cause__ or ex.__context__


def _find_static_tls_error(ex):
    for subex in _iter_exception_chain(ex):
        text = str(subex)
        if 'cannot allocate memory in static TLS block' in text:
            return subex
    return None


def _import_osgeo_component(component_name):
    """
    Import a GDAL / OSGeo component with support for both modern and legacy
    GDAL Python binding layouts.

    Tries, in order:

        1. osgeo.<component_name>
        2. <component_name>

    For example:

        osgeo.gdal -> gdal
        osgeo.osr  -> osr

    Gives a better diagnostic for static TLS import-order failures.
    """
    import sys
    import importlib

    def _raise_static_tls_error(tls_ex, import_name):
        suspects = [
            'torch',
            'tensorflow',
            'jax',
            'jaxlib',
            'cv2',
            'pyarrow',
            'sklearn',
            'xgboost',
            'lightgbm',
        ]
        loaded_suspects = [m for m in suspects if m in sys.modules]

        suspect_text = (
            ', '.join(loaded_suspects)
            if loaded_suspects else
            'none of the usual suspects detected in sys.modules'
        )

        raise ImportError(
            f'GDAL/OSGeo component {component_name!r} failed while importing '
            f'{import_name!r} because the dynamic loader could not allocate '
            'static TLS space. This is often import-order-sensitive with '
            'binary extension modules. Try importing GDAL / OSGeo before '
            'importing kwimage, torch, or other heavy binary packages.\n\n'
            f'Loaded suspect modules before OSGeo import failed: '
            f'{suspect_text}\n\n'
            f'Original loader error: {tls_ex}'
        ) from tls_ex

    import_names = [
        f'osgeo.{component_name}',
        component_name,  # legacy GDAL bindings, e.g. ``import gdal``
    ]

    errors = []

    for import_name in import_names:
        try:
            return importlib.import_module(import_name)
        except ImportError as ex:
            tls_ex = _find_static_tls_error(ex)
            if tls_ex is not None:
                _raise_static_tls_error(tls_ex, import_name)
            errors.append((import_name, ex))

    error_lines = [
        f'Could not import GDAL/OSGeo component {component_name!r}.',
        '',
        'Tried import paths:',
    ]

    for import_name, ex in errors:
        error_lines.append(f'    - {import_name!r}: {ex!r}')

    error_lines += [
        '',
        'The GDAL Python bindings are required for this operation. '
        'Install GDAL / osgeo for this Python environment.',
    ]

    raise ModuleNotFoundError('\n'.join(error_lines)) from errors[-1][1]


def import_gdal():
    """
    Import GDAL with support for both ``osgeo.gdal`` and legacy ``gdal``.
    """
    return _import_osgeo_component('gdal')


def import_osr():
    """
    Import OSR with support for both ``osgeo.osr`` and legacy ``osr``.
    """
    return _import_osgeo_component('osr')
