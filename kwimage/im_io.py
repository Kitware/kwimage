# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import warnings
import cv2
from os.path import exists
from . import im_cv2


def imread(fpath, space='rgb'):
    """
    reads image data in RGB format

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> from os.path import splitext  # NOQA
        >>> fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
        >>> fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> ext = splitext(fpath)[1]
        >>> img1 = imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=ext)
        >>> imwrite(tmp.name, img1)
        >>> img2 = imread(tmp.name)
        >>> assert np.all(img2 == img1)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> img1 = imread(ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff')
        >>> img1 = imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

        import plottool as pt
        pt.qtensure()
        pt.imshow(tif_im / 2 ** 16, pnum=(1, 2, 1), fnum=1)
        pt.imshow(png_im / 2 ** 16, pnum=(1, 2, 2), fnum=1)

    Ignore:
        from PIL import Image
        pil_img = Image.open(tif_fpath)
        assert int(Image.PILLOW_VERSION.split('.')[0]) > 4
    """
    try:
        if fpath.lower().endswith(('.ntf', '.nitf')):
            try:
                import gdal
            except ImportError:
                raise Exception('cannot read NITF images without gdal')
            try:
                gdal_dset = gdal.Open(fpath)
                if gdal_dset.RasterCount == 1:
                    band = gdal_dset.GetRasterBand(1)
                    image = np.array(band.ReadAsArray())
                elif gdal_dset.RasterCount == 3:
                    bands = [
                        gdal_dset.GetRasterBand(i)
                        for i in [1, 2, 3]
                    ]
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
        elif fpath.lower().endswith(('.tif', '.tiff')):
            import skimage.io
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # skimage reads in RGB, convert to BGR
                image = skimage.io.imread(fpath)
                im_cv2.convert_colorspace(image, 'rgb', dst_space=space,
                                          implicit=True)
        else:
            image = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                if exists(fpath):
                    raise IOError('OpenCV cannot read this image: "{}", '
                                  'but it exists'.format(fpath))
                else:
                    raise IOError('OpenCV cannot read this image: "{}", '
                                  'because it does not exist'.format(fpath))
            if space is not None:
                image = im_cv2.convert_colorspace(image, src_space='bgr',
                                                  dst_space=space,
                                                  implicit=True)
        return image
    except Exception as ex:
        print('Error reading fpath = {!r}'.format(fpath))
        raise


def imwrite(fpath, image, space='rgb'):
    """
    writes image data in BGR format
    """
    if fpath.endswith(('.tif', '.tiff')):
        import skimage.io
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage writes in RGB, convert from BGR
            image = im_cv2.convert_colorspace(image, space, dst_space='rgb',
                                              implicit=True)
            return skimage.io.imsave(fpath, image)
    else:
        # OpenCV writes in bgr
        image = im_cv2.convert_colorspace(image, space, dst_space='bgr',
                                          implicit=True)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = im_cv2.convert_colorspace(image, 'bgra', dst_space='bgr',
                                              implicit=False)
        try:
            return cv2.imwrite(fpath, image)
        except cv2.error as ex:
            if 'could not find a writer for the specified extension' in str(ex):
                raise ValueError(
                    'Image fpath {!r} does not have a known image extension '
                    '(e.g. png/jpg)'.format(fpath))
            else:
                raise
