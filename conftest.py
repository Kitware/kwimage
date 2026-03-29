# content of conftest.py
import pytest  # NOQA


def pytest_addoption(parser):
    # Allow --network to be passed in as an option on sys.argv
    parser.addoption('--network', action='store_true')


def pytest_sessionstart(session):
    """
    In 3.14, there is a TLS (thread local storage) issue due to the order in
    which libraries are loaded. I don't quite understand it. To work around it,
    import gdal first if we have it.

    Ideally we find a more robust solution here and understand the cause, and a
    precise minimal way to reproduce it.
    """
    # Importing any of these libraries doesn't seem to impact anything
    # import shapely
    # import cv2
    # import skimage
    # import numpy
    # import scipy
    # from PIL import Image

    if 1:
        try:
            import osgeo  # NOQA
            # from osgeo import osr
            # from osgeo import gdal  # NOQA
        except ImportError:
            ...
