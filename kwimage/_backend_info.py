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
