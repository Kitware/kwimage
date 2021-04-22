# -*- coding: utf-8 -*-
"""
This module provides functions ``imread`` and ``imwrite`` which are wrappers
around concrete readers/writers provided by other libraries. This allows us to
support a wider array of formats than any of individual backends.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import warnings  # NOQA
import cv2
from os.path import exists
import ubelt as ub
from . import im_cv2
from . import im_core


# Common image extensions
JPG_EXTENSIONS = (
    '.jpg', '.jpeg'
)

# These should be supported by opencv / PIL
_WELL_KNOWN_EXTENSIONS = (
    JPG_EXTENSIONS +
    ('.bmp', '.pgm', '.png',)
)


# Extensions that usually will require GDAL
GDAL_EXTENSIONS = (
    '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif',
    '.r0', '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
    '.jp2', '.vrt',
)

IMAGE_EXTENSIONS = (
    _WELL_KNOWN_EXTENSIONS +
    ('.tif', '.tiff',) +
    GDAL_EXTENSIONS
)


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
            modification is made to whatever backend is used to read the image.

        backend (str, default='auto'): which backend reader to use. By default
            the file extension is used to determine this, but it can be
            manually overridden. Valid backends are gdal, skimage, and cv2.

    Returns:
        ndarray: the image data in the specified color space.

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
        >>> kwplot.imshow(img1, pnum=(1, 2, 1), fnum=1, norm=True)
        >>> kwplot.imshow(img2, pnum=(1, 2, 2), fnum=1, norm=True)

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
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff', fname='pepper.tif')
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

    Benchmark:
        >>> from kwimage.im_io import *  # NOQA
        >>> import timerit
        >>> import kwimage
        >>> import tempfile
        >>> #
        >>> dsize = (1920, 1080)
        >>> img1 = kwimage.grab_test_image('amazon', dsize=dsize)
        >>> ti = timerit.Timerit(10, bestof=3, verbose=1, unit='us')
        >>> formats = {}
        >>> dpath = ub.ensure_app_cache_dir('cache')
        >>> space = 'auto'
        >>> formats['png'] = kwimage.imwrite(join(dpath, '.png'), img1, space=space, backend='cv2')
        >>> formats['jpg'] = kwimage.imwrite(join(dpath, '.jpg'), img1, space=space, backend='cv2')
        >>> formats['tif_raw'] = kwimage.imwrite(join(dpath, '.raw.tif'), img1, space=space, backend='gdal', compress='RAW')
        >>> formats['tif_deflate'] = kwimage.imwrite(join(dpath, '.deflate.tif'), img1, space=space, backend='gdal', compress='DEFLATE')
        >>> formats['tif_lzw'] = kwimage.imwrite(join(dpath, '.lzw.tif'), img1, space=space, backend='gdal', compress='LZW')
        >>> grid = [
        >>>     ('cv2', 'png'),
        >>>     ('cv2', 'jpg'),
        >>>     ('gdal', 'jpg'),
        >>>     ('turbojpeg', 'jpg'),
        >>>     ('gdal', 'tif_raw'),
        >>>     ('gdal', 'tif_lzw'),
        >>>     ('gdal', 'tif_deflate'),
        >>>     ('skimage', 'tif_raw'),
        >>> ]
        >>> backend, filefmt = 'cv2', 'png'
        >>> for backend, filefmt in grid:
        >>>     for timer in ti.reset(f'imread-{filefmt}-{backend}'):
        >>>         with timer:
        >>>             kwimage.imread(formats[filefmt], space=space, backend=backend)
        >>> # Test all formats in auto mode
        >>> for filefmt in formats.keys():
        >>>     for timer in ti.reset(f'kwimage.imread-{filefmt}-auto'):
        >>>         with timer:
        >>>             kwimage.imread(formats[filefmt], space=space, backend='auto')
        >>> ti.measures = ub.map_vals(ub.sorted_vals, ti.measures)
        >>> import netharn as nh
        >>> print('ti.measures = {}'.format(nh.util.align(ub.repr2(ti.measures['min'], nl=2), ':')))
        Timed best=42891.504 µs, mean=44008.439 ± 1409.2 µs for imread-png-cv2
        Timed best=33146.808 µs, mean=34185.172 ± 656.3 µs for imread-jpg-cv2
        Timed best=40120.306 µs, mean=41220.927 ± 1010.9 µs for imread-jpg-gdal
        Timed best=30798.162 µs, mean=31573.070 ± 737.0 µs for imread-jpg-turbojpeg
        Timed best=6223.170 µs, mean=6370.462 ± 150.7 µs for imread-tif_raw-gdal
        Timed best=42459.404 µs, mean=46519.940 ± 5664.9 µs for imread-tif_lzw-gdal
        Timed best=36271.175 µs, mean=37301.108 ± 861.1 µs for imread-tif_deflate-gdal
        Timed best=5239.503 µs, mean=6566.574 ± 1086.2 µs for imread-tif_raw-skimage
        ti.measures = {
            'imread-tif_raw-skimage' : 0.0052395030070329085,
            'imread-tif_raw-gdal'    : 0.006223169999429956,
            'imread-jpg-turbojpeg'   : 0.030798161998973228,
            'imread-jpg-cv2'         : 0.03314680799667258,
            'imread-tif_deflate-gdal': 0.03627117499127053,
            'imread-jpg-gdal'        : 0.040120305988239124,
            'imread-tif_lzw-gdal'    : 0.042459404008695856,
            'imread-png-cv2'         : 0.042891503995633684,
        }


        >>> print('ti.measures = {}'.format(nh.util.align(ub.repr2(ti.measures['mean'], nl=2), ':')))
    """
    if backend == 'auto':
        # TODO: memoize the extensions?
        # Determine the backend reader using the file extension
        _fpath_lower = fpath.lower()
        # Note: rset dataset (https://trac.osgeo.org/gdal/ticket/3457) support is hacked
        if _fpath_lower.endswith(GDAL_EXTENSIONS):
            backend = 'gdal'
        elif _fpath_lower.endswith(('.tif', '.tiff')):
            if _have_gdal():
                backend = 'gdal'
            else:
                backend = 'skimage'
        elif _fpath_lower.endswith(JPG_EXTENSIONS):
            if _have_turbojpg():
                backend = 'turbojpeg'
            else:
                backend = 'cv2'
        else:
            backend = 'cv2'

    try:
        if backend == 'gdal':
            image, src_space, auto_dst_space = _imread_gdal(fpath)
        elif backend == 'cv2':
            image, src_space, auto_dst_space = _imread_cv2(fpath)
        elif backend == 'turbojpeg':
            image, src_space, auto_dst_space = _imread_turbojpeg(fpath)
        elif backend == 'skimage':
            image, src_space, auto_dst_space = _imread_skimage(fpath)
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


def _imread_turbojpeg(fpath):
    """
    See: https://www.learnopencv.com/efficient-image-loading/

    References:
        https://pypi.org/project/PyTurboJPEG/

    Bash:
        pip install PyTurboJPEG
        sudo apt install libturbojpeg -y

    Ignore:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # xdoctest: +REQUIRES(turbojpeg)
        >>> import kwimage
        >>> rgb_fpath = kwimage.grab_test_image_fpath('amazon')
        >>> assert rgb_fpath.endswith('.jpg')
        >>> #
        >>> rgb = kwimage.imread(rgb_fpath)
        >>> gray = kwimage.convert_colorspace(rgb, 'rgb', 'gray')
        >>> gray_fpath = rgb_fpath + '.gray.jpg'
        >>> kwimage.imwrite(gray_fpath, gray)
        >>> #
        >>> fpath = gray_fpath
        >>> #
        >>> from kwimage.im_io import _imread_turbojpeg, _imread_skimage, _imread_cv2
        >>> import timerit
        >>> ti = timerit.Timerit(50, bestof=10, verbose=2)
        >>> #
        >>> for timer in ti.reset('turbojpeg'):
        >>>     with timer:
        >>>         im_turbo = _imread_turbojpeg(fpath)
        >>> #
        >>> for timer in ti.reset('cv2'):
        >>>     with timer:
        >>>         im_cv2 = _imread_cv2(fpath)
    """
    import turbojpeg
    jpeg = turbojpeg.TurboJPEG()
    with open(fpath, 'rb') as file:
        data = file.read()
        (width, height, jpeg_subsample, jpeg_colorspace) = jpeg.decode_header(data)
        # print('width = {!r}'.format(width))
        # print('height = {!r}'.format(height))
        # print('jpeg_subsample = {!r}'.format(jpeg_subsample))
        # print('jpeg_colorspace = {!r}'.format(jpeg_colorspace))
        if jpeg_colorspace == turbojpeg.TJCS_GRAY:
            pixel_format = turbojpeg.TJPF_GRAY
            src_space = 'gray'
            auto_dst_space = 'gray'
        else:
            pixel_format = turbojpeg.TJPF_RGB
            src_space = 'rgb'
            auto_dst_space = 'rgb'
        image = jpeg.decode(data, pixel_format=pixel_format)
    return image, src_space, auto_dst_space


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
    try:
        from osgeo import gdal
    except ImportError:
        import gdal
    try:
        gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        if gdal_dset is None:
            raise IOError('GDAL cannot read: {!r}'.format(fpath))

        num_channels = gdal_dset.RasterCount

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
                dtype = _gdal_to_numpy_dtype(gdal_dtype)

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
            gdal_dtype = bands[0].DataType
            dtype = _gdal_to_numpy_dtype(gdal_dtype)
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


def imwrite(fpath, image, space='auto', backend='auto', **kwargs):
    """
    Writes image data to disk.

    Args:
        fpath (PathLike): location to save the image

        image (ndarray): image data

        space (str): the colorspace of the image to save. Can by any colorspace
            accepted by `convert_colorspace`, or it can be 'auto', in which
            case we assume the input image is either RGB, RGBA or grayscale.
            If None, then absolutely no color modification is made and
            whatever backend is used writes the image as-is.

        backend (str, default='auto'): which backend writer to use. By default
            the file extension is used to determine this. Valid backends are
            gdal, skimage, and cv2.

        **kwargs : args passed to the backend writer

    Returns:
        str: path to the written file

    Notes:
        The image may be modified to preserve its colorspace depending on which
        backend is used to write the image.

        When saving as a jpeg or png, the image must be encoded with the uint8
        data type. When saving as a tiff, any data type is allowed.

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
        >>>         imwrite(tmp_tif.name, img1, space=space, backend='skimage')
        >>>         imwrite(tmp_png.name, img1, space=space)
        >>>         tif_im = imread(tmp_tif.name, space=space)
        >>>         png_im = imread(tmp_png.name, space=space)
        >>>         assert np.all(tif_im == png_im), 'im-read/write inconsistency'
        >>>         if _have_gdal:
        >>>             tmp_tif2 = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>             imwrite(tmp_tif2.name, img1, space=space, backend='gdal')
        >>>             tif_im2 = imread(tmp_tif2.name, space=space)
        >>>             assert np.all(tif_im == tif_im2), 'im-read/write inconsistency'
        >>>         if space == 'gray':
        >>>             assert tif_im.ndim == 2
        >>>             assert png_im.ndim == 2
        >>>         elif space in ['rgb', 'bgr']:
        >>>             assert tif_im.shape[2] == 3
        >>>             assert png_im.shape[2] == 3
        >>>         elif space in ['rgba', 'bgra']:
        >>>             assert tif_im.shape[2] == 4
        >>>             assert png_im.shape[2] == 4

    Benchmark:
        >>> import timerit
        >>> import os
        >>> import kwimage
        >>> import tempfile
        >>> #
        >>> img1 = kwimage.grab_test_image('astro', dsize=(1920, 1080))
        >>> space = 'auto'
        >>> #
        >>> file_sizes = {}
        >>> #
        >>> ti = timerit.Timerit(10, bestof=3, verbose=2)
        >>> #
        >>> for timer in ti.reset('imwrite-skimage-tif'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='skimage')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-cv2-png'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.png')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='cv2')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-cv2-jpg'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.jpg')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='cv2')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-gdal-raw'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='gdal', compress='RAW')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-gdal-lzw'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='gdal', compress='LZW')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-gdal-zstd'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='gdal', compress='ZSTD')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-gdal-deflate'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='gdal', compress='DEFLATE')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> for timer in ti.reset('imwrite-gdal-jpeg'):
        >>>     with timer:
        >>>         tmp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         kwimage.imwrite(tmp.name, img1, space=space, backend='gdal', compress='JPEG')
        >>>     file_sizes[ti.label] = os.stat(tmp.name).st_size
        >>> #
        >>> file_sizes = ub.sorted_vals(file_sizes)
        >>> import xdev
        >>> file_sizes_human = ub.map_vals(lambda x: xdev.byte_str(x, 'MB'), file_sizes)
        >>> print('ti.rankings = {}'.format(ub.repr2(ti.rankings, nl=2)))
        >>> print('file_sizes = {}'.format(ub.repr2(file_sizes_human, nl=1)))
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

    if backend == 'auto':
        if fpath.endswith(('.tif', '.tiff')):
            if _have_gdal():
                backend = 'gdal'
            else:
                backend = 'skimage'
        elif fpath.endswith(GDAL_EXTENSIONS):
            if _have_gdal():
                backend = 'gdal'
        else:
            backend = 'cv2'

    if space is not None:
        if backend == 'cv2':
            # OpenCV writes images in BGR(A)/ grayscale
            if n_channels == 3:
                dst_space = 'bgr'
            elif n_channels == 4:
                dst_space = 'bgra'
            elif n_channels == 1:
                dst_space = 'gray'
            else:
                raise AssertionError('impossible state')
        else:
            # most writers like skimage and gdal write images in RGB(A)/ grayscale
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

    if backend == 'cv2':
        try:
            cv2.imwrite(fpath, image, **kwargs)
        except cv2.error as ex:
            if 'could not find a writer for the specified extension' in str(ex):
                raise ValueError(
                    'Image fpath {!r} does not have a known image extension '
                    '(e.g. png/jpg)'.format(fpath))
            else:
                raise
    elif backend == 'skimage':
        import skimage.io
        skimage.io.imsave(fpath, image, **kwargs)
    elif backend == 'gdal':
        _imwrite_cloud_optimized_geotiff(fpath, image, **kwargs)
    elif backend == 'turbojpeg':
        raise NotImplementedError
    else:
        raise KeyError('Unknown imwrite backend={!r}'.format(backend))

    return fpath


def load_image_shape(fpath):
    """
    Determine the height/width/channels of an image without reading the entire
    file.

    Args:
        fpath (str): path to an image

    Returns:
        Tuple - shape of the dataset.
            Recall this library uses the convention that "shape" is refers to
            height,width,channels and "size" is width,height ordering.

    Benchmark:
        >>> # For large files, PIL is much faster
        >>> import gdal
        >>> from PIL import Image
        >>> #
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> #
        >>> ti = ub.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('gdal'):
        >>>     with timer:
        >>>         gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        >>>         width = gdal_dset.RasterXSize
        >>>         height = gdal_dset.RasterYSize
        >>>         gdal_dset = None
        >>> #
        >>> for timer in ti.reset('PIL'):
        >>>     with timer:
        >>>         pil_img = Image.open(fpath)
        >>>         width, height = pil_img.size
        >>>         pil_img.close()
        Timed gdal for: 100 loops, best of 10
            time per loop: best=62.967 µs, mean=63.991 ± 0.8 µs
        Timed PIL for: 100 loops, best of 10
            time per loop: best=46.640 µs, mean=47.314 ± 0.4 µs
    """
    from PIL import Image
    try:
        pil_img = Image.open(fpath)
        width, height = pil_img.size
        num_channels = len(pil_img.getbands())
        pil_img.close()
    except Exception as pil_ex:
        if not _have_gdal():
            raise
        try:
            import gdal
            gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
            if gdal_dset is None:
                raise Exception
            width = gdal_dset.RasterXSize
            height = gdal_dset.RasterYSize
            num_channels = gdal_dset.RasterCount
            gdal_dset = None
        except (ImportError, Exception):
            raise pil_ex
    shape = (height, width, num_channels)
    return shape


def __inspect_optional_overhead():
    r"""
        Benchmark:
            >>> from kwimage.im_io import _have_gdal, _have_turbojpg  # NOQA
            >>> def dis_instructions(func):
            >>>     import dis
            >>>     import io
            >>>     buf = io.StringIO()
            >>>     dis.disassemble(func.__code__, file=buf)
            >>>     _, text = buf.seek(0), buf.read()
            >>>     return text
            >>> func = _have_turbojpg
            >>> func = _have_gdal
            >>> memo = ub.memoize(func)
            >>> print(func_dis := dis_instructions(func))
            >>> print(memo_dis := dis_instructions(memo))
            >>> n = max(func_dis.count('\n'), memo_dis.count('\n'))
            >>> sep = ' | \n' * n
            >>> prefix = '| \n' * n
            >>> print('\n\n')
            >>> print(ub.hzcat([prefix, x, sep, y]))

        Benchmark:
            >>> from kwimage.im_io import _have_gdal, _have_turbojpg  # NOQA
            >>> funcs = []
            >>> funcs += [_have_turbojpg]
            >>> funcs += [_have_gdal]
            >>> for func in funcs:
            >>>     memo = ub.memoize(func)
            >>>     print('func = {!r}'.format(func))
            >>>     print('memo = {!r}'.format(memo))
            >>>     import timerit
            >>>     ti = timerit.Timerit(100, bestof=10, verbose=1, unit='us')
            >>>     ti.reset('call func').call(func).report()
            >>>     ti.reset('call memo').call(memo).report()
    """
    raise NotImplementedError


@ub.memoize
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


def _have_gdal():
    try:
        import gdal  # NOQA
    except Exception:
        return False
    else:
        return True


def _imwrite_cloud_optimized_geotiff(fpath, data, compress='auto',
                                     blocksize=256, overviews=None,
                                     overview_resample='NEAREST', options=[]):
    """
    Writes data as a cloud-optimized geotiff using gdal

    Args:
        fpath (PathLike): file path to save the COG to.

        data (ndarray[ndim=3]): Raw HWC image data to save. Dimensions should
            be height, width, channels.

        compress (bool, default='auto'): Can be JPEG (lossy) or LZW (lossless),
            or DEFLATE (lossless). Can also be 'auto', which will try to
            heuristically choose a sensible choice.

        blocksize (int, default=256): size of tiled blocks

        overviews (None | int | list, default=None):
            if specified as a list, then uses exactly those overviews. If
            specified as an integer a list is created using powers of two.

        overview_resample (str, default='NEAREST'): resampling method for
            overview pyramid. Valid choices are: 'NEAREST', 'AVERAGE',
            'BILINEAR', 'CUBIC', 'CUBICSPLINE', 'LANCZOS'.

        options (List[str]): other gdal options

    Returns:
        str: the file path where the data was written

    References:
        https://geoexamples.com/other/2019/02/08/cog-tutorial.html#create-a-cog-using-gdal-python
        http://osgeo-org.1560.x6.nabble.com/gdal-dev-Creating-Cloud-Optimized-GeoTIFFs-td5320101.html
        https://gdal.org/drivers/raster/cog.html
        https://github.com/harshurampur/Geotiff-conversion
        https://github.com/sshuair/cogeotiff
        https://github.com/cogeotiff/rio-cogeo
        https://gis.stackexchange.com/questions/1104/should-gdal-be-set-to-produce-geotiff-files-with-compression-which-algorithm-sh

    Notes:
        Need to fix `CXXABI_1.3.11 not found` with conda gdal sometimes

        CLI to reproduce:
            python -c "import kwimage; kwimage.imread(kwimage.grab_test_image_fpath(), backend='gdal')"

        This error seems to be fixed by using 2.3.3 instead of 3.x gdal, not
        sure why, should look into that.

    CommandLine:
        xdoctest -m kwimage.im_io _imwrite_cloud_optimized_geotiff

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> from kwimage.im_io import *  # NOQA
        >>> from kwimage.im_io import _imwrite_cloud_optimized_geotiff
        >>> import tempfile
        >>> data = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.cog.tif')
        >>> fpath = tmp_tif.name
        >>> compress = 'JPEG'
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='JPEG')
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')

        >>> data = (np.random.rand(100, 100, 4) * 255).astype(np.uint8)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='JPEG')
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='DEFLATE')

        >>> data = (np.random.rand(100, 100, 5) * 255).astype(np.uint8)
        >>> _imwrite_cloud_optimized_geotiff(fpath, data, compress='LZW')

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, overviews=3)
        >>> from osgeo import gdal
        >>> ds = gdal.Open(fpath, gdal.GA_ReadOnly)
        >>> filename = ds.GetDescription()
        >>> main_band = ds.GetRasterBand(1)
        >>> assert main_band.GetOverviewCount() == 3

        >>> _imwrite_cloud_optimized_geotiff(fpath, data, overviews=[2, 4])

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> from kwimage.im_io import *  # NOQA
        >>> from kwimage.im_io import _imwrite_cloud_optimized_geotiff
        >>> import tempfile
        >>> import kwimage
        >>> # Test with uint16
        >>> shape = (100, 100, 1)
        >>> dtype = np.uint16
        >>> dinfo = np.iinfo(np.uint16)
        >>> data = kwimage.normalize(kwimage.gaussian_patch(shape))
        >>> data = ((data - dinfo.min) * (dinfo.max - dinfo.min)).astype(dtype)
        >>> import tempfile
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> fpath = tmp_tif.name
        >>> kwimage.imwrite(fpath, data)
        >>> loaded = kwimage.imread(fpath)
        >>> assert np.all(loaded.ravel() == data.ravel())
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.imshow(loaded / dinfo.max)
        >>> kwplot.show_if_requested()
    """
    from osgeo import gdal
    if len(data.shape) == 2:
        data = data[:, :, None]

    y_size, x_size, num_bands = data.shape

    data_set = None
    if compress == 'auto':
        compress = _gdal_auto_compress(data=data)

    # if compress not in ['JPEG', 'LZW', 'DEFLATE', 'RAW']:
    #     raise KeyError('unknown compress={}'.format(compress))
    # JPEG/LZW/PACKBITS/DEFLATE/CCITTRLE/CCITTFAX3/CCITTFAX4/LZMA/ZSTD/LERC/LERC_DEFLATE/LERC_ZSTD/WEBP/NONE

    if compress == 'JPEG' and num_bands >= 5:
        raise ValueError('Cannot use JPEG with more than 4 channels (got {})'.format(num_bands))

    eType = _numpy_to_gdal_dtype(data.dtype)
    if compress == 'JPEG':
        if eType not in [gdal.GDT_Byte, gdal.GDT_UInt16]:
            raise ValueError('JPEG compression must use 8 or 16 bit integers')

    # NEAREST/AVERAGE/BILINEAR/CUBIC/CUBICSPLINE/LANCZOS
    if overviews is None:
        overviewlist = []
    elif isinstance(overviews, int):
        overviewlist = (2 ** np.arange(1, overviews + 1)).tolist()
    else:
        overviewlist = overviews

    _options = [
        'TILED=YES',
        'BIGTIFF=YES',
        'BLOCKXSIZE={}'.format(blocksize),
        'BLOCKYSIZE={}'.format(blocksize),
    ]
    if compress != 'RAW':
        _options += ['COMPRESS={}'.format(compress)]
    if compress == 'JPEG' and num_bands == 3:
        # Using YCBCR speeds up jpeg compression by quite a bit
        _options += ['PHOTOMETRIC=YCBCR']

    if overviewlist:
        _options.append('COPY_SRC_OVERVIEWS=YES')

    _options += options

    _options = list(map(str, _options))  # python2.7 support

    # Create an in-memory dataset where we will prepare the COG data structure
    driver = gdal.GetDriverByName(str('MEM'))
    data_set = driver.Create(str(''), x_size, y_size, num_bands, eType=eType)
    for i in range(num_bands):
        band_data = np.ascontiguousarray(data[:, :, i])
        data_set.GetRasterBand(i + 1).WriteArray(band_data)

    if overviewlist:
        # Build the downsampled overviews (for fast zoom in / out)
        data_set.BuildOverviews(str(overview_resample), overviewlist)

    driver = None
    # Copy the in-memory dataset to an on-disk GeoTiff
    driver2 = gdal.GetDriverByName(str('GTiff'))
    data_set2 = driver2.CreateCopy(fpath, data_set, options=_options)
    data_set = None

    # OK, so setting things to None turns out to be important. Gah!
    data_set2.FlushCache()

    # Dereference everything
    data_set2 = None
    driver2 = None
    return fpath


def _numpy_to_gdal_dtype(numpy_dtype):
    """
    maps numpy dtypes to gdal dtypes
    """
    from osgeo import gdal
    if not hasattr(numpy_dtype, 'kind'):
        # convert to the dtype instance object
        numpy_dtype = numpy_dtype().dtype
    kindsize = (numpy_dtype.kind, numpy_dtype.itemsize)
    if kindsize == ('u', 1):
        eType = gdal.GDT_Byte
    elif kindsize == ('u', 2):
        eType = gdal.GDT_UInt16
    elif kindsize == ('u', 4):
        eType = gdal.GDT_UInt32
    elif kindsize == ('i', 2):
        eType = gdal.GDT_Int16
    elif kindsize == ('i', 4):
        eType = gdal.GDT_Int32
    elif kindsize == ('f', 4):
        eType = gdal.GDT_Float32
    elif kindsize == ('f', 8):
        eType = gdal.GDT_Float64
    elif kindsize == ('c', 8):
        eType = gdal.GDT_CFloat32
    elif kindsize == ('c', 16):
        eType = gdal.GDT_CFloat64
    else:
        raise TypeError('Unsupported GDAL dtype for {}'.format(kindsize))
    return eType


def _gdal_to_numpy_dtype(gdal_dtype):
    """
    maps gdal dtypes to numpy dtypes

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> numpy_types = [np.uint8, np.uint16, np.int16, np.uint32, np.int32,
        >>>                np.float32, np.float64, np.complex64,
        >>>                np.complex128]
        >>> for np_type in numpy_types:
        >>>     numpy_dtype1 = np_type().dtype
        >>>     gdal_dtype1 = _numpy_to_gdal_dtype(numpy_dtype1)
        >>>     numpy_dtype2 = _gdal_to_numpy_dtype(gdal_dtype1)
        >>>     gdal_dtype2 = _numpy_to_gdal_dtype(numpy_dtype2)
        >>>     assert gdal_dtype2 == gdal_dtype1
        >>>     assert _dtype_equality(numpy_dtype1, numpy_dtype2)
    """
    from osgeo import gdal
    _GDAL_DTYPE_LUT = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex_,
        gdal.GDT_CInt32: np.complex_,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128
    }
    return _GDAL_DTYPE_LUT[gdal_dtype]


def _gdal_auto_compress(src_fpath=None, data=None, data_set=None):
    """
    Heuristic for automatically choosing gdal compression type

    Args:
        src_fpath (str): path to source image if known
        data (ndarray): data pixels if known
        data_set (gdal.Dataset): gdal dataset if known

    Returns:
        str: gdal compression code

    References:
        https://kokoalberti.com/articles/geotiff-compression-optimization-guide/

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> assert _gdal_auto_compress(src_fpath='foo.jpg') == 'JPEG'
        >>> assert _gdal_auto_compress(src_fpath='foo.png') == 'LZW'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2)) == 'RAW'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 3).astype(np.uint8)) == 'RAW'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 4).astype(np.uint8)) == 'RAW'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 1).astype(np.uint8)) == 'RAW'
    """
    compress = None
    num_channels = None
    dtype = None

    if src_fpath is not None:
        # the filepath might hint at which compress method is best.
        ext = src_fpath[-5:].lower()
        if ext.endswith(('.jpg', '.jpeg')):
            compress = 'JPEG'
        elif ext.endswith(('.png', '.png')):
            compress = 'LZW'

    if compress is None:

        if data_set is not None:
            if dtype is None:
                main_band = data_set.GetRasterBand(1)
                dtype = _gdal_to_numpy_dtype(main_band.DataType)

            if num_channels is None:
                data_set.RasterCount == 3

        elif data is not None:
            if dtype is None:
                dtype = data.dtype

            if num_channels is None:
                if len(data.shape) == 3:
                    num_channels = data.shape[2]

    # if compress is None:
    #     if _dtype_equality(dtype, np.uint8) and num_channels == 3:
    #         compress = 'JPEG'

    if compress is None:
        # which backend is best in this case?
        compress = 'RAW'
    return compress


def _dtype_equality(dtype1, dtype2):
    """
    Check for numpy dtype equality

    References:
        https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype

    Example:
        dtype1 = np.empty(0, dtype=np.uint8).dtype
        dtype2 = np.uint8
        _dtype_equality(dtype1, dtype2)
    """
    dtype1_ = getattr(dtype1, 'type', dtype1)
    dtype2_ = getattr(dtype2, 'type', dtype2)
    return dtype1_ == dtype2_
