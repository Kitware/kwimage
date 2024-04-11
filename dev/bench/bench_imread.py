"""
What is the best way to read a common tiff image?


Ignore:
        du -sh ~/data/sample_ptif.ptif
        du -sh ~/data/sample_cog.cog.tif
        gdal_translate ~/data/sample_ptif.ptif ~/data/sample_cog.cog.tif -co TILED=YES -co COMPRESS=LZW -co COPY_SRC_OVERVIEWS=YES
        gdal_translate ~/data/sample_ptif.ptif ~/data/sample_cog.cog.tif -co TILED=YES -co COMPRESS=JPEG -co COPY_SRC_OVERVIEWS=YES -co PHOTOMETRIC=YCBCR

        gdal_translate foo.png test.cog.tif -co TILED=YES -co COMPRESS=JPEG -co COPY_SRC_OVERVIEWS=YES -co PHOTOMETRIC=YCBCR
        gdal_translate foo.png test1.cog.tif -co TILED=YES -co COMPRESS=JPEG -co COPY_SRC_OVERVIEWS=YES
        gdal_translate foo.png test2.cog.tif -co TILED=YES -co COMPRESS=LZW -co COPY_SRC_OVERVIEWS=YES

        python -m ndsampler.validate_cog --verbose test.cog.tif
        python -m ndsampler.validate_cog --verbose foo.png
"""
import numpy as np


def _read_gdal_v1(fpath):
    from osgeo import gdal
    try:
        gdal_dset = gdal.Open(fpath)
        if gdal_dset.RasterCount == 1:
            band = gdal_dset.GetRasterBand(1)

            color_table = band.GetColorTable()
            if color_table is None:
                image = np.array(band.ReadAsArray())
            else:
                raise Exception
        elif gdal_dset.RasterCount == 3:
            bands = [gdal_dset.GetRasterBand(i) for i in [1, 2, 3]]
            channels = [np.array(band.ReadAsArray()) for band in bands]
            image = np.dstack(channels)
        else:
            raise NotImplementedError(
                'Can only read 1 or 3 channel NTF images. '
                'Got {}'.format(gdal_dset.RasterCount))
    except Exception:
        raise
    finally:
        gdal_dset = None
    return image


def _read_gdal_v2(fpath):
    import gdal
    try:
        gdal_dset = gdal.Open(fpath)
        if gdal_dset.RasterCount == 1:
            band = gdal_dset.GetRasterBand(1)

            color_table = band.GetColorTable()
            if color_table is None:
                image = np.array(band.ReadAsArray())
            else:
                raise Exception('cant handle color tables yet')
        elif gdal_dset.RasterCount == 3:
            _gdal_dtype_lut = {
                1: np.uint8,     2: np.uint16,
                3: np.int16,     4: np.uint32,      5: np.int32,
                6: np.float32,   7: np.float64,     8: np.complex_,
                9: np.complex_,  10: np.complex64,  11: np.complex128
            }
            bands = [gdal_dset.GetRasterBand(i) for i in [1, 2, 3]]
            gdal_type_code = bands[0].DataType
            dtype = _gdal_dtype_lut[gdal_type_code]
            shape = (gdal_dset.RasterYSize, gdal_dset.RasterXSize, gdal_dset.RasterCount)
            # Preallocate and populate image
            image = np.empty(shape, dtype=dtype)
            for i, band in enumerate(bands):
                image[:, :, i] = band.ReadAsArray()
        else:
            raise NotImplementedError(
                'Can only read 1 or 3 channel NTF images. '
                'Got {}'.format(gdal_dset.RasterCount))
    except Exception:
        raise
    finally:
        gdal_dset = None
    return image


def _read_rasterio(fpath):
    import rasterio
    dataset = rasterio.open(fpath)
    image = dataset.read().transpose((1, 2, 0))
    return image


def _read_pil(fpath):
    from PIL import Image
    pil_img = Image.open(fpath)
    image = np.asarray(pil_img)
    return image


def bench_imread():
    import ubelt as ub
    # fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')

    import kwimage
    fpath = kwimage.grab_test_image_fpath('airport')

    # A color-table geotiff
    # https://download.osgeo.org/geotiff/samples/
    # fpath = ub.grabdata('https://download.osgeo.org/geotiff/samples/usgs/c41078a1.tif')

    ti = ub.Timerit(100, bestof=5, verbose=2)

    results = {}

    # fpath = '/home/joncrall/data/sample_ptif.ptif'
    # fpath = '/home/joncrall/data/sample_cog.cog.tif'

    if 0:
        for timer in ti.reset('gdal-v1'):
            with timer:
                image = _read_gdal_v1(fpath)
        results[ti.label] = image.sum()

        for timer in ti.reset('gdal-v2'):
            with timer:
                image = _read_gdal_v2(fpath)
        results[ti.label] = image.sum()

        for timer in ti.reset('rasterio'):
            with timer:
                image = _read_rasterio(fpath)
        results[ti.label] = image.sum()

    import skimage.io
    """
    pip install tifffile
    pip install imagecodecs

    """
    for timer in ti.reset('skimage'):
        with timer:
            image = skimage.io.imread(fpath)
    results[ti.label] = image.sum()

    import kwimage
    for timer in ti.reset('kwimage'):
        with timer:
            image = kwimage.imread(fpath)
    results[ti.label] = image.sum()

    import cv2
    for timer in ti.reset('cv2'):
        with timer:
            image = cv2.imread(fpath)
    results[ti.label] = image.sum()

    for timer in ti.reset('pil'):
        with timer:
            image = _read_pil(fpath)
    results[ti.label] = image.sum()


# TODO: integrate in this test
# tiffile is a potential alternative, but it seems slow
# def _imwrite_tiffile():
#     """
#     Example:
#         >>> from kwimage.im_io import *  # NOQA
#         >>> import tempfile
#         >>> data = np.random.rand(800, 800, 3).astype(np.float32)
#         >>> tempfile = tempfile.NamedTemporaryFile(suffix='.tif')
#         >>> fpath = tempfile.name

#         >>> import tifffile
#         >>> tifffile.imsave(fpath, data, tile=(256, 256), metadata={'axes': 'YXC'})
#         >>> recon = tifffile.imread(fpath)
#         >>> assert np.all(recon == data)

#         >>> import kwimage
#         >>> kwimage.imwrite(fpath, data, space=None, backend='gdal')
#         >>> recon2 = kwimage.imread(fpath)
#         >>> assert np.all(recon2 == data)
#         >>> _ = ub.cmd('gdalinfo ' + fpath, verbose=1)


#         >>> # Benchmark
#         >>> import timerit
#         >>> ti = timerit.Timerit(10, bestof=3, verbose=2)
#         >>> # Run benchmark variants
#         >>> for timer in ti.reset('tiffile.imsave'):
#         >>>     with timer:
#         >>>         tifffile.imsave(fpath, data, tile=(256, 256), metadata={'axes': 'YXC'})
#         >>> for timer in ti.reset('tiffile.imread'):
#         >>>     with timer:
#         >>>         recon = tifffile.imread(fpath)
#         >>> for timer in ti.reset('kwimage.imwrite'):
#         >>>     with timer:
#         >>>         kwimage.imwrite(fpath, data, space=None, backend='gdal')
#         >>> for timer in ti.reset('kwimage.imread'):
#         >>>     with timer:
#         >>>         recon = kwimage.imread(fpath)
#         >>> print('ti.rankings = {}'.format(ub.urepr(ti.rankings['mean'], nl=2, align=':')))
#     """


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_imread.py
    """
    bench_imread()
