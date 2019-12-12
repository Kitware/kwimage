# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import warnings  # NOQA
import cv2
from os.path import exists
from . import im_cv2
from . import im_core


def imread(fpath, space='auto', backend='auto'):
    """
    Reads image data in a specified format using some backend implementation.

    Args:
        fpath (str): path to the file to be read

        space (str, default='auto'): the desired colorspace of the image. Can
            by any colorspace accepted by `convert_colorspace`, or it can be
            'auto', in which case the colorspace of the image is unmodified
            (except in the case where a color image is read by opencv, in which
            case we convert BGR to RGB by default). If None, then no
            modification is made to whaveter backend is used to read the image.

        backend (str, default='auto'): which backend reader to use. By default
            the file extension is used to determine this, but it can be
            manually overridden. Valid backends are gdal, skimage, and cv2.

    Note:
        if space is something non-standard like HSV or LAB, then the file must
        be a normal 8-bit color image, otherwise an error will occur.

    Raises:
        IOError - If the image cannot be read
        ImportError - If trying to read a nitf without gdal
        NotImplementedError - if trying to read a corner-case image

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from kwimage.im_io import *  # NOQA
        >>> import tempfile
        >>> from os.path import splitext  # NOQA
        >>> # Test a non-standard image, which encodes a depth map
        >>> fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> img1 = imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=splitext(fpath)[1])
        >>> imwrite(tmp.name, img1)
        >>> img2 = imread(tmp.name)
        >>> assert np.all(img2 == img1)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img1, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(img2, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> img1 = imread(ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(png_im, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(tif_im, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff', fname='pepper.tif'),
        >>> img1 = imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(png_im / 2 ** 16, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(tif_im / 2 ** 16, pnum=(1, 2, 2), fnum=1)
    """
    if backend == 'auto':
        # Determine the backend reader using the file extension
        _fpath_lower = fpath.lower()
        # Note: rset dataset (https://trac.osgeo.org/gdal/ticket/3457) support is hacked
        GDAL_EXTENSIONS = (
            '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif',
            '.r0', '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
        )
        if _fpath_lower.endswith(GDAL_EXTENSIONS):
            backend = 'gdal'
        elif _fpath_lower.endswith(('.tif', '.tiff')):
            if _have_gdal():
                backend = 'gdal'
            else:
                backend = 'skimage'
        else:
            backend = 'cv2'

    try:
        if backend == 'gdal':
            image, src_space, auto_dst_space = _imread_gdal(fpath)
        elif backend == 'skimage':
            image, src_space, auto_dst_space = _imread_skimage(fpath)
        elif backend == 'cv2':
            image, src_space, auto_dst_space = _imread_cv2(fpath)
        else:
            raise KeyError('Unknown imread backend={!r}'.format(backend))

        if space == 'auto':
            dst_space = auto_dst_space
        else:
            dst_space = space

        if dst_space is not None:
            if src_space is None:
                raise ValueError((
                    'Cannot convert to destination colorspace ({}) because'
                    ' the source colorspace could not be determined. Use '
                    ' space=None to return the raw data.'
                ).format(dst_space))

            image = im_cv2.convert_colorspace(image, src_space=src_space,
                                              dst_space=dst_space,
                                              implicit=False)

        return image
    except Exception as ex:
        print('ex = {!r}'.format(ex))
        print('Error reading fpath = {!r}'.format(fpath))
        raise


def _imread_skimage(fpath):
    import skimage.io
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    # skimage reads color in RGB by default
    image = skimage.io.imread(fpath)
    n_channels = im_core.num_channels(image)
    if n_channels == 3:
        src_space = 'rgb'
    elif n_channels == 4:
        src_space = 'rgba'
    elif n_channels == 1:
        src_space = 'gray'
    else:
        raise NotImplementedError('unknown number of channels')
    auto_dst_space = src_space
    return image, src_space, auto_dst_space


def _imread_cv2(fpath):
    # opencv reads color in BGR by default
    image = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)
    if image is None:
        if exists(fpath):
            # TODO: this could be a permissions error. We could test for that.
            # and print a better error message in that case.
            raise IOError('OpenCV cannot read this image: "{}", '
                          'but it exists'.format(fpath))
        else:
            raise IOError('OpenCV cannot read this image: "{}", '
                          'because it does not exist'.format(fpath))

    n_channels = im_core.num_channels(image)
    if n_channels == 3:
        src_space = 'bgr'
        auto_dst_space = 'rgb'
    elif n_channels == 4:
        src_space = 'bgra'
        auto_dst_space = 'rgba'
    elif n_channels == 1:
        src_space = 'gray'
        auto_dst_space = 'gray'
    else:
        raise NotImplementedError('unknown number of channels')
    return image, src_space, auto_dst_space


def _imread_gdal(fpath):
    """ gdal imread backend """
    import gdal
    try:
        gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        if gdal_dset is None:
            raise IOError('GDAL cannot read: {!r}'.format(fpath))

        num_channels = gdal_dset.RasterCount

        _gdal_dtype_lut = {
            1: np.uint8,     2: np.uint16,
            3: np.int16,     4: np.uint32,      5: np.int32,
            6: np.float32,   7: np.float64,     8: np.complex_,
            9: np.complex_,  10: np.complex64,  11: np.complex128
        }

        if num_channels == 1:
            band = gdal_dset.GetRasterBand(1)

            color_table = band.GetColorTable()
            if color_table is None:
                buf = band.ReadAsArray()
                if buf is None:
                    # Sometimes this works if you try again. I don't know why.
                    # It spits out annoying messages, not sure how to supress.
                    buf = band.ReadAsArray()
                    if buf is None:
                        raise IOError('GDal was unable to read this band')
                image = np.array(buf)

            else:
                # The buffer is an index into the color table
                buf = band.ReadAsArray()

                gdal_dtype = color_table.GetPaletteInterpretation()
                dtype = _gdal_dtype_lut[gdal_dtype]

                num_colors = color_table.GetCount()
                if num_colors <= 0:
                    raise AssertionError('invalid color table')
                idx_to_color = []
                for idx in range(num_colors):
                    color = color_table.GetColorEntry(idx)
                    idx_to_color.append(color)

                # The color table specifies the real number of channels
                num_channels = len(color)

                idx_to_color = np.array(idx_to_color, dtype=dtype)
                image = idx_to_color[buf]

        else:
            bands = [gdal_dset.GetRasterBand(i)
                     for i in range(1, num_channels + 1)]
            gdal_type_code = bands[0].DataType
            dtype = _gdal_dtype_lut[gdal_type_code]
            shape = (gdal_dset.RasterYSize, gdal_dset.RasterXSize,
                     gdal_dset.RasterCount)
            # Preallocate and populate image
            image = np.empty(shape, dtype=dtype)
            for i, band in enumerate(bands):
                image[:, :, i] = band.ReadAsArray()

        # note this isn't a safe assumption, but it is an OK default
        # hueristic
        if num_channels == 1:
            src_space = 'gray'
        elif num_channels == 3:
            src_space = 'rgb'
        elif num_channels == 4:
            src_space = 'rgba'
        else:
            # We have no hint of the source color space in this instance
            src_space = None

    except Exception:
        raise
    finally:
        gdal_dset = None
    auto_dst_space = src_space
    return image, src_space, auto_dst_space


def imwrite(fpath, image, space='auto'):
    """
    Writes image data to disk.

    Args:
        fpath (PathLike): location to save the imaeg
        image (ndarray): image data
        space (str): the colorspace of the image to save. Can by any colorspace
            accepted by `convert_colorspace`, or it can be 'auto', in which
            case we assume the input image is either RGB, RGBA or grayscale.
            If None, then absolutely no color modification is made and
            whatever backend is used writes the image as-is.

    Notes:
        The image may be modified to preserve its colorspace depending on which
        backend is used to write the image.

    Raises:
        Exception : if the image cannot be written

    Doctest:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # This should be moved to a unit test
        >>> import tempfile
        >>> test_image_paths = [
        >>>    ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff', fname='pepper.tif'),
        >>>    ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'),
        >>>    #ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif'),
        >>>    ub.grabdata('https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png', fname='parrot.png')
        >>> ]
        >>> for fpath in test_image_paths:
        >>>     for space in ['auto', 'rgb', 'bgr', 'gray', 'rgba']:
        >>>         img1 = imread(fpath, space=space)
        >>>         print('Test im-io consistency of fpath = {!r} in {} space, shape={}'.format(fpath, space, img1.shape))
        >>>         # Write the image in TIF and PNG format
        >>>         tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>>         imwrite(tmp_tif.name, img1, space=space)
        >>>         imwrite(tmp_png.name, img1, space=space)
        >>>         tif_im = imread(tmp_tif.name, space=space)
        >>>         png_im = imread(tmp_png.name, space=space)
        >>>         assert np.all(tif_im == png_im), 'im-read/write inconsistency'
        >>>         if space == 'gray':
        >>>             assert tif_im.ndim == 2
        >>>             assert png_im.ndim == 2
        >>>         elif space in ['rgb', 'bgr']:
        >>>             assert tif_im.shape[2] == 3
        >>>             assert png_im.shape[2] == 3
        >>>         elif space in ['rgba', 'bgra']:
        >>>             assert tif_im.shape[2] == 4
        >>>             assert png_im.shape[2] == 4
    """
    if space is not None:
        n_channels = im_core.num_channels(image)

    if space == 'auto':
        if n_channels == 3:
            auto_src_space = 'rgb'
        elif n_channels == 4:
            auto_src_space = 'rgba'
        elif n_channels == 1:
            auto_src_space = 'gray'
        else:
            # TODO: allow writing of arbitrary num channels using gdal
            raise NotImplementedError('unknown number of channels')
        src_space = auto_src_space
    else:
        src_space = space

    if fpath.endswith(('.tif', '.tiff')):
        # TODO: allow writing of tiff images using gdal (allow COGS)

        if space is not None:
            # skimage writes images in RGB(A)/ grayscale
            if n_channels == 3:
                dst_space = 'rgb'
            elif n_channels == 4:
                dst_space = 'rgba'
            elif n_channels == 1:
                dst_space = 'gray'
            else:
                raise AssertionError('impossible state')
            image = im_cv2.convert_colorspace(
                image, src_space=src_space, dst_space=dst_space,
                implicit=False)

        import skimage.io
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        return skimage.io.imsave(fpath, image)
    else:
        if space is not None:
            # OpenCV writes images in BGR(A)/ grayscale
            if n_channels == 3:
                dst_space = 'bgr'
            elif n_channels == 4:
                dst_space = 'bgra'
            elif n_channels == 1:
                dst_space = 'gray'
            else:
                raise AssertionError('impossible state')
            image = im_cv2.convert_colorspace(
                image, src_space=src_space, dst_space=dst_space,
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


def _have_gdal():
    try:
        import gdal  # NOQA
    except ImportError:
        return False
    else:
        return True
