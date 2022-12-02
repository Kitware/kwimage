"""
This module provides functions ``imread`` and ``imwrite`` which are wrappers
around concrete readers/writers provided by other libraries. This allows us to
support a wider array of formats than any of individual backends.
"""
import os
import numpy as np
import warnings  # NOQA
import cv2
from os.path import exists, dirname
import ubelt as ub
from . import im_cv2
from . import im_core

__all__ = [
    'imread', 'imwrite', 'load_image_shape',
]


# Common image extensions
JPG_EXTENSIONS = (
    '.jpg', '.jpeg'
)

# These should be supported by opencv / PIL
_WELL_KNOWN_EXTENSIONS = (
    JPG_EXTENSIONS +
    ('.bmp', '.pgm', '.png', '.qoi',)
)


# Extensions that usually will require GDAL
GDAL_EXTENSIONS = (
    '.ntf', '.nitf', '.ptif', '.cog.tiff', '.cog.tif',
    '.r0', '.r1', '.r2', '.r3', '.r4', '.r5', '.nsf',
    '.jp2', '.vrt',
)

# TODO: ITK Image formats
# https://insightsoftwareconsortium.github.io/itk-js/docs/image_formats.html
ITK_EXTENSIONS = (
    '.mha',
    '.nrrd',  # http://teem.sourceforge.net/nrrd/format.html
    '.mgh',  # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
    '.mgz',
    '.nii',  # https://nifti.nimh.nih.gov/nifti-1
    '.img',
    '.mrb',  # multiple-resolution-bitmap https://whatext.com/mrb
)

# ITK Demo data:
# https://data.kitware.com/#collection/57b5c9e58d777f126827f5a1

IMAGE_EXTENSIONS = (
    _WELL_KNOWN_EXTENSIONS +
    ('.tif', '.tiff',) +
    GDAL_EXTENSIONS +
    ITK_EXTENSIONS
)


def imread(fpath, space='auto', backend='auto', **kw):
    """
    Reads image data in a specified format using some backend implementation.

    Args:
        fpath (str): path to the file to be read

        space (str):
            The desired colorspace of the image. Can by any colorspace accepted
            by `convert_colorspace`, or it can be 'auto', in which case the
            colorspace of the image is unmodified (except in the case where a
            color image is read by opencv, in which case we convert BGR to RGB
            by default). If None, then no modification is made to whatever
            backend is used to read the image. Defaults to 'auto'.

            New in version 0.7.10: when the backend does not resolve to "cv2"
            the "auto" space resolves to None, thus the image is read as-is.

        backend (str): which backend reader to use. By default
            the file extension is used to determine this, but it can be
            manually overridden. Valid backends are 'gdal', 'skimage', 'itk',
            'pil', and 'cv2'. Defaults to 'auto'.

        **kw : backend-specific arguments

    Returns:
        ndarray: the image data in the specified color space.

    Note:
        if space is something non-standard like HSV or LAB, then the file must
        be a normal 8-bit color image, otherwise an error will occur.

    Note:
        Some backends will respect EXIF orientation (skimage) and others will
        not (gdal, cv2).

        The scikit-image backend is itself another multi-backend plugin-based
        image reader/writer.

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
        >>> fpath = ub.grabdata(
        >>>     'http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif',
        >>>     hasher='sha256', hash_prefix='64522acba6f0fb7060cd4c202ed32c5163c34e63d386afdada4190cce51ff4d4')
        >>> img1 = kwimage.imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=splitext(fpath)[1])
        >>> kwimage.imwrite(tmp.name, img1)
        >>> img2 = kwimage.imread(tmp.name)
        >>> assert np.all(img2 == img1)
        >>> tmp.close()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img1, pnum=(1, 2, 1), fnum=1, norm=True, title='tif orig')
        >>> kwplot.imshow(img2, pnum=(1, 2, 2), fnum=1, norm=True, title='tif io round-trip')

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> import kwimage
        >>> img1 = kwimage.imread(ub.grabdata(
        >>>     'http://i.imgur.com/iXNf4Me.png', fname='ada.png', hasher='sha256',
        >>>     hash_prefix='898cf2588c40baf64d6e09b6a93b4c8dcc0db26140639a365b57619e17dd1c77'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> kwimage.imwrite(tmp_tif.name, img1)
        >>> kwimage.imwrite(tmp_png.name, img1)
        >>> tif_im = kwimage.imread(tmp_tif.name)
        >>> png_im = kwimage.imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)
        >>> tmp_tif.close()
        >>> tmp_png.close()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(png_im, pnum=(1, 2, 1), fnum=1, title='tif io')
        >>> kwplot.imshow(tif_im, pnum=(1, 2, 2), fnum=1, title='png io')

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> import kwimage
        >>> tif_fpath = ub.grabdata(
        >>>     'https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff',
        >>>     fname='pepper.tif', hasher='sha256',
        >>>     hash_prefix='31ff3a1f416cb7281acfbcbb4b56ee8bb94e9f91489602ff2806e5a49abc03c0')
        >>> img1 = kwimage.imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> kwimage.imwrite(tmp_tif.name, img1)
        >>> kwimage.imwrite(tmp_png.name, img1)
        >>> tif_im = kwimage.imread(tmp_tif.name)
        >>> png_im = kwimage.imread(tmp_png.name)
        >>> tmp_tif.close()
        >>> tmp_png.close()
        >>> assert np.all(tif_im == png_im)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(png_im / 2 ** 16, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(tif_im / 2 ** 16, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> # xdoctest: +REQUIRES(module:itk, --network)
        >>> import kwimage
        >>> import ubelt as ub
        >>> # Grab an image that ITK can read
        >>> fpath = ub.grabdata(
        >>>     url='https://data.kitware.com/api/v1/file/606754e32fa25629b9476f9e/download',
        >>>     fname='brainweb1e5a10f17Rot20Tx20.mha',
        >>>     hash_prefix='08f0812591691ae24a29788ba8cd1942e91', hasher='sha512')
        >>> # Read the image (this is actually a DxHxW stack of images)
        >>> img1_stack = kwimage.imread(fpath)
        >>> # Check that write + read preserves data
        >>> import tempfile
        >>> tmp_file = tempfile.NamedTemporaryFile(suffix='.mha')
        >>> kwimage.imwrite(tmp_file.name, img1_stack)
        >>> recon = kwimage.imread(tmp_file.name)
        >>> assert not np.may_share_memory(recon, img1_stack)
        >>> assert np.all(recon == img1_stack)
        >>> tmp_file.close()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.stack_images_grid(recon[0::20]),
        >>>               title='kwimage.imread with a .mha file')
        >>> kwplot.show_if_requested()

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
    """
    fpath = os.fspath(fpath)
    if backend == 'auto':
        # TODO: memoize the extensions?

        # TODO: each backend should maintain a list of supported (possibly
        # overlapping) formats, and that should be used to build a mapping from
        # formats to candidate backends. We should then filter down to a
        # backend that actually exists.

        # Determine the backend reader using the file extension
        _fpath_lower = fpath.lower()
        # Note: rset dataset (https://trac.osgeo.org/gdal/ticket/3457) support is hacked
        if _fpath_lower.endswith(ITK_EXTENSIONS):
            backend = 'itk'
        elif _fpath_lower.endswith(GDAL_EXTENSIONS):
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
        elif _fpath_lower.endswith('.svg'):
            backend = 'svg'  # a bit hacky, not a raster format
        else:
            # TODO: if we don't have an extension we could try to inspect the
            # file header
            USE_FILE_HEADER = 0
            if USE_FILE_HEADER:
                '''
                for key in kwimage.grab_test_image_fpath.keys():
                    fpath = kwimage.grab_test_image_fpath(key)
                    with open(fpath, 'rb') as file:
                        header_bytes = file.read(4)
                        print(header_bytes)
                '''
                JPEG_HEADER = b'\xff\xd8\xff'
                PNG_HEADER = b'\x89PNG'
                NITF_HEADER = b'NITF'
                with open(fpath, 'rb') as file:
                    header_bytes = file.read(4)
                if header_bytes.startswith(JPEG_HEADER):
                    backend = 'cv2'
                elif header_bytes.startswith(PNG_HEADER):
                    backend = 'cv2'
                elif header_bytes.startswith(NITF_HEADER):
                    backend = 'gdal'
                else:
                    backend = 'cv2'
            else:
                backend = 'cv2'

    if space == 'auto' and backend != 'cv2':
        # cv2 is the only backend that does weird things, we can
        # default to auto and save the user the headache of specifying this
        space = None

    try:
        if backend == 'gdal':
            image, src_space, auto_dst_space = _imread_gdal(fpath, **kw)
        elif backend == 'cv2':
            image, src_space, auto_dst_space = _imread_cv2(fpath)
        elif backend == 'turbojpeg':
            image, src_space, auto_dst_space = _imread_turbojpeg(fpath)
        elif backend == 'skimage':
            image, src_space, auto_dst_space = _imread_skimage(fpath)
        elif backend == 'pil':
            image, src_space, auto_dst_space = _imread_pil(fpath)
        elif backend == 'qoi':
            image, src_space, auto_dst_space = _imread_qoi(fpath)
        elif backend == 'itk':
            src_space, auto_dst_space = None, None
            import itk
            itk_obj = itk.imread(fpath)
            image = np.asarray(itk_obj)
        elif backend == 'svg':
            image, src_space, auto_dst_space = _imread_svg(fpath)
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


def _imread_qoi(fpath):
    """
    """
    import qoi
    image = qoi.read(fpath)
    src_space, auto_dst_space = None, None
    return image, src_space, auto_dst_space


def _imwrite_qoi(fpath, data):
    """
    Only seems to allow RGB 255.

    Ignore:
        >>> from kwimage.im_io import imread, _imread_qoi, _imwrite_qoi
        >>> import kwimage
        >>> data = kwimage.ensure_uint255(kwimage.checkerboard())
        >>> fpath = 'tmp.qoi'
        >>> _imwrite_qoi(fpath, data)
        >>> recon, _, _ = _imread_qoi(fpath)
    """
    import kwimage
    import qoi
    data = kwimage.atleast_3channels(data)
    qoi.write(fpath, data)
    return fpath


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


def _imread_pil(fpath):
    from PIL import Image
    pil_img = Image.open(fpath)
    image = np.array(pil_img)
    if pil_img.mode == 'RGB':
        src_space = 'rgb'
        auto_dst_space = 'rgb'
    elif pil_img.mode == 'RGBA':
        src_space = 'rgba'
        auto_dst_space = 'rgba'
    elif len(image.shape) == 2 or image.shape[2] == 1:
        src_space = 'gray'
        auto_dst_space = 'gray'
    elif len(image.shape) == 3 or image.shape[2] == 3:
        src_space = 'rgb'
        auto_dst_space = 'rgb'
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
        src_space = None
        # raise NotImplementedError('unknown number of channels')
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


def _imread_gdal(fpath, overview=None, ignore_color_table=False,
                 nodata_method=None, band_indices=None, nodata=None):
    """
    gdal imread backend

    Args:
        overview (int):
            if specified load a specific overview level

        ignore_color_table (bool):
            if True and the image has a color table, return its indexes
            instead of the colored image.

        nodata_method (None | str):
            if None, any nodata attributes are ignored. Otherwise specifies how
            nodata values should be handled. If "ma", returns a masked array
            instead of a normal ndarray. If "float", always returns a float
            array where masked values are replaced with nan.

        band_indices (None | List[int]):
            if None, all bands are read, otherwise only specified
            band indexes are read.

    References:
        [GDAL_Config_Options] https://gdal.org/user/configoptions.html
        https://gis.stackexchange.com/questions/180961/reading-a-specific-overview-layer-from-geotiff-file-using-gdal-python

    Ignore:
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath('amazon')

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test nodata values
        >>> import kwimage
        >>> from osgeo import gdal
        >>> from osgeo import osr
        >>> # Make a dummy geotiff
        >>> imdata = kwimage.grab_test_image('airport')
        >>> dpath = ub.Path.appdir('kwimage/test/geotiff').ensuredir()
        >>> geo_fpath = dpath / 'dummy_geotiff.tif'
        >>> # compute dummy values for a geotransform to CRS84
        >>> img_h, img_w = imdata.shape[0:2]
        >>> img_box = kwimage.Boxes([[0, 0, img_w, img_h]], 'xywh')
        >>> wld_box = kwimage.Boxes([[-73.7595528, 42.6552404, 0.0001, 0.0001]], 'xywh')
        >>> img_corners = img_box.corners()
        >>> wld_corners = wld_box.corners()
        >>> transform = kwimage.Affine.fit(img_corners, wld_corners)
        >>> nodata = -9999
        >>> srs = osr.SpatialReference()
        >>> srs.ImportFromEPSG(4326)
        >>> crs = srs.ExportToWkt()
        >>> # Set a region to be nodata
        >>> imdata = imdata.astype(np.int16)
        >>> imdata[-100:] = nodata
        >>> imdata[0:200:, -200:-180] = nodata
        >>> mask = (imdata == nodata)
        >>> kwimage.imwrite(geo_fpath, imdata, backend='gdal', nodata=-9999,
        >>>                 crs=crs, transform=transform)
        >>> # Read the geotiff with different methods
        >>> raw_recon = kwimage.imread(geo_fpath, nodata=None)
        >>> ma_recon = kwimage.imread(geo_fpath, nodata='ma')
        >>> nan_recon = kwimage.imread(geo_fpath, nodata='float')
        >>> # raw values should be read
        >>> assert np.all(raw_recon[mask] == nodata)
        >>> assert not np.any(raw_recon[~mask] == nodata)
        >>> # nans should be in nodata places
        >>> assert np.all(np.isnan(nan_recon[mask]))
        >>> assert not np.any(np.isnan(nan_recon[~mask]))
        >>> # check locations are masked correctly
        >>> assert np.all(ma_recon[mask].mask)
        >>> assert not np.any(ma_recon[~mask].mask)

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test band specification
        >>> import kwimage
        >>> import pytest
        >>> # Make a dummy geotiff
        >>> imdata = kwimage.grab_test_image('airport')
        >>> dpath = ub.Path.appdir('kwimage/test/geotiff').ensuredir()
        >>> fpath1 = dpath / 'dummy_overviews_rgb.tif'
        >>> kwimage.imwrite(fpath1, imdata, overviews=3, backend='gdal')
        >>> band0 = kwimage.imread(fpath1, backend='gdal', band_indices=[0, 1])
        >>> assert band0.shape[2] == 2

        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('time'):
            with timer:
                band0 = kwimage.imread(fpath1, backend='gdal', band_indices=[0, 1])

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test overview values
        >>> import kwimage
        >>> import pytest
        >>> # Make a dummy geotiff
        >>> imdata = kwimage.grab_test_image('airport')
        >>> dpath = ub.Path.appdir('kwimage/test/geotiff').ensuredir()
        >>> fpath1 = dpath / 'dummy_overviews_rgb.tif'
        >>> fpath2 = dpath / 'dummy_overviews_gray.tif'
        >>> kwimage.imwrite(fpath1, imdata, overviews=3, backend='gdal')
        >>> kwimage.imwrite(fpath2, imdata[:, :, 0], overviews=3, backend='gdal')
        >>> recon1_3a = kwimage.imread(fpath1, overview='coarsest', backend='gdal')
        >>> recon1_3b = kwimage.imread(fpath1, overview=3, backend='gdal')
        >>> recon1_None = kwimage.imread(fpath1, backend='gdal')
        >>> recon1_0 = kwimage.imread(fpath1, overview=0, backend='gdal')
        >>> recon1_1 = kwimage.imread(fpath1, overview=1, backend='gdal')
        >>> recon1_2 = kwimage.imread(fpath1, overview=2, backend='gdal')
        >>> recon1_3 = kwimage.imread(fpath1, overview=3, backend='gdal')
        >>> with pytest.raises(ValueError):
        >>>     kwimage.imread(fpath1, overview=4, backend='gdal')
        >>> assert recon1_0.shape == (868, 1156, 3)
        >>> assert recon1_1.shape == (434, 578, 3)
        >>> assert recon1_2.shape == (217, 289, 3)
        >>> assert recon1_3.shape == (109, 145, 3)
        >>> assert recon1_3a.shape == (109, 145, 3)
        >>> assert recon1_3b.shape == (109, 145, 3)
        >>> recon2_3a = kwimage.imread(fpath2, overview='coarsest', backend='gdal')
        >>> recon2_3b = kwimage.imread(fpath2, overview=3, backend='gdal')
        >>> recon2_0 = kwimage.imread(fpath2, overview=0, backend='gdal')
        >>> assert recon2_0.shape == (868, 1156)
        >>> assert recon2_3a.shape == (109, 145)
        >>> assert recon2_3b.shape == (109, 145)
        >>> # TODO: test an image with a color table
    """
    try:
        from osgeo import gdal
    except ImportError:
        import gdal
    try:
        if nodata is not None:
            ub.schedule_deprecation(
                modname='kwimage', name='nodata',
                type='argument to _imread_gdal',
                migration='use nodata_method instead',
                deprecate='0.9.5', error='0.10.0', remove='0.11.0')
            nodata_method = nodata

        if nodata_method is not None:
            if isinstance(nodata_method, str):
                if nodata_method not in {'ma', 'float'}:
                    raise KeyError('nodata_method={} must be ma, float, or None'.format(nodata_method))
            else:
                raise TypeError(type(nodata_method))

        gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        if gdal_dset is None:
            raise IOError('GDAL cannot read: {!r}'.format(fpath))

        gdalkw = {}  # xoff, yoff, win_xsize, win_ysize
        image, num_channels = _gdal_read(
            gdal_dset, overview=overview,
            ignore_color_table=ignore_color_table,
            band_indices=band_indices, gdalkw=gdalkw,
            nodata_method=nodata_method,
            nodata_value=None,
        )

        # note this isn't a safe assumption, but it is an OK default heuristic
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


def _gdal_read(gdal_dset, overview, nodata=None, ignore_color_table=None,
               band_indices=None, gdalkw=None, nodata_method=None,
               nodata_value=None):
    """
    Backend for reading data from an open gdal dataset
    """

    if nodata is not None:
        # backwards compat
        nodata_method = nodata

    # TODO:
    # - [ ] Handle SubDatasets (e.g. ones produced by scikit-image)
    # https://gdal.org/drivers/raster/gtiff.html#subdatasets
    # See ../tests/test_io.py for experiments that trigger this
    if len(gdal_dset.GetSubDatasets()):
        raise NotImplementedError('subdatasets are not handled correctly')
        # INTERLEAVE = gdal_dset.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE', '')
        # if INTERLEAVE == 'BAND':
        #     if len(gdal_dset.GetSubDatasets()) > 0:
        #         raise NotImplementedError('Cannot handle interleaved files yet')

    total_num_channels = gdal_dset.RasterCount

    if band_indices is None:
        band_indices = range(total_num_channels)
        num_channels = total_num_channels
    else:
        num_channels = len(band_indices)

    default_bands = [gdal_dset.GetRasterBand(i + 1) for i in band_indices]
    default_band0 = default_bands[0]

    if overview:
        overview_count = default_band0.GetOverviewCount()
        if isinstance(overview, str):
            if overview == 'coarsest':
                overview = overview_count
            else:
                raise KeyError(overview)
        if overview < 0:
            warnings.warn('Using negative overviews is deprecated. '
                          'Use coarset to get the lowest resolution overview')
            overview = max(overview_count + overview, 0)
        if overview > overview_count:
            raise ValueError('Image has no overview={}'.format(overview))
        if overview > 0:
            bands = [b.GetOverview(overview - 1) for b in default_bands]
            if any(b is None for b in bands):
                raise AssertionError(
                    'Band was None in {}'.format(gdal_dset.GetDescription()))
        else:
            bands = default_bands
    else:
        bands = default_bands

    if num_channels == 1:
        band = bands[0]

        color_table = None if ignore_color_table else band.GetColorTable()
        if color_table is None:
            buf = band.ReadAsArray(**gdalkw)
            if buf is None:
                # Sometimes this works if you try again. I don't know why.
                # It spits out annoying messages, not sure how to supress.
                # TODO: need MWE and an issue describing this workaround.
                buf = band.ReadAsArray(**gdalkw)
                if buf is None:
                    raise IOError('GDal was unable to read this band')
            image = np.array(buf)
        else:
            # The buffer is an index into the color table
            buf = band.ReadAsArray(**gdalkw)

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

        if nodata_method is not None:
            # TODO: not sure if this works right for
            # color table images
            band_nodata = band.GetNoDataValue()
            mask = band_nodata == buf
            if color_table is not None:
                # Fix mask to align with color table
                table_chans = idx_to_color.shape[1]
                mask = np.tile(mask[:, :, None], (1, 1, table_chans))
    else:
        band0 = bands[0]
        xsize = gdalkw.get('win_xsize', band0.XSize)
        ysize = gdalkw.get('win_ysize', band0.YSize)
        gdal_dtype = band0.DataType
        dtype = _gdal_to_numpy_dtype(gdal_dtype)
        shape = (ysize, xsize, num_channels)
        # Preallocate and populate image
        image = np.empty(shape, dtype=dtype)
        if nodata_method is not None:
            mask = np.empty(shape, dtype=bool)
        for idx, band in enumerate(bands):
            # load with less memory by specifing buf_obj
            buf = image[:, :, idx]
            ret = band.ReadAsArray(buf_obj=buf, **gdalkw)
            # ret = buf = band.ReadAsArray(**gdalkw)
            if ret is None:
                raise IOError(ub.paragraph(
                    '''
                    GDAL was unable to read band: {}, {}'
                    from {!r}
                    '''.format(idx, band, gdal_dset.GetDescription())))
            # image[:, :, idx] = buf
            if nodata_method is not None:
                band_nodata = band.GetNoDataValue()
                mask_buf = mask[:, :, idx]
                np.equal(buf, band_nodata, out=mask_buf)
                # mask[:, :, idx] = (buf == band_nodata)

    if nodata_method is not None:
        if nodata_method == 'ma':
            image = np.ma.array(image, mask=mask)
        elif nodata_method == 'float':
            promote_dtype = np.result_type(image.dtype, np.float32)
            image = image.astype(promote_dtype)
            image[mask] = np.nan
        else:
            raise KeyError('nodata_method={}'.format(nodata_method))

    return image, num_channels


def imwrite(fpath, image, space='auto', backend='auto', **kwargs):
    """
    Writes image data to disk.

    Args:
        fpath (PathLike): location to save the image

        image (ndarray): image data

        space (str | None):
            the colorspace of the image to save. Can by any colorspace accepted
            by `convert_colorspace`, or it can be 'auto', in which case we
            assume the input image is either RGB, RGBA or grayscale.  If None,
            then absolutely no color modification is made and whatever backend
            is used writes the image as-is.

            New in version 0.7.10: when the backend does not resolve to "cv2",
            the "auto" space resolves to None, thus the image is saved as-is.

        backend (str):
            Which backend writer to use. By default the file extension is used
            to determine this. Valid backends are 'gdal', 'skimage', 'itk', and
            'cv2'.

        **kwargs : args passed to the backend writer.
            When the backend is gdal, available options are:
            compress (str): Common options are auto, DEFLATE, LZW, JPEG.
            blocksize (int): size of tiled blocks (e.g. 256)
            overviews (None | str | int | list): Number of overviews.
            overview_resample (str): Common options NEAREST, CUBIC, LANCZOS
            options (List[str]): other gdal options.
            nodata (int): denotes a integer value as nodata.
            transform (kwimage.Affine): Transform to CRS from pixel space
            crs (str): The coordinate reference system for transform.
            See :func:`_imwrite_cloud_optimized_geotiff` for more details each options.
            When the backend is itk, see :func:`itk.imwrite` for options
            When the backend is skimage, see :func:`skimage.io.imsave` for options
            When the backend is cv2 see :func:`cv2.imwrite` for options.

    Returns:
        str: path to the written file

    Note:
        The image may be modified to preserve its colorspace depending on which
        backend is used to write the image.

        When saving as a jpeg or png, the image must be encoded with the uint8
        data type. When saving as a tiff, any data type is allowed.

        The scikit-image backend is itself another multi-backend plugin-based
        image reader/writer.

    Raises:
        Exception : if the image cannot be written

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # This should be moved to a unit test
        >>> from kwimage.im_io import _have_gdal  # NOQA
        >>> import kwimage
        >>> import tempfile
        >>> test_image_paths = [
        >>>    ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff', fname='pepper.tif'),
        >>>    ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'),
        >>>    #ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif'),
        >>>    ub.grabdata('https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png', fname='parrot.png')
        >>> ]
        >>> for fpath in test_image_paths:
        >>>     for space in ['auto', 'rgb', 'bgr', 'gray', 'rgba']:
        >>>         img1 = kwimage.imread(fpath, space=space)
        >>>         print('Test im-io consistency of fpath = {!r} in {} space, shape={}'.format(fpath, space, img1.shape))
        >>>         # Write the image in TIF and PNG format
        >>>         tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>         tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>>         kwimage.imwrite(tmp_tif.name, img1, space=space, backend='skimage')
        >>>         kwimage.imwrite(tmp_png.name, img1, space=space)
        >>>         tif_im = kwimage.imread(tmp_tif.name, space=space)
        >>>         png_im = kwimage.imread(tmp_png.name, space=space)
        >>>         assert np.all(tif_im == png_im), 'im-read/write inconsistency'
        >>>         if _have_gdal:
        >>>             tmp_tif2 = tempfile.NamedTemporaryFile(suffix='.tif')
        >>>             kwimage.imwrite(tmp_tif2.name, img1, space=space, backend='gdal')
        >>>             tif_im2 = kwimage.imread(tmp_tif2.name, space=space)
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
        >>>         tmp_tif.close()
        >>>         tmp_png.close()

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

    Example:
        >>> # Test saving a multi-band file
        >>> import kwimage
        >>> import pytest
        >>> import tempfile
        >>> # In this case the backend will not resolve to cv2, so
        >>> # we should not need to specify space.
        >>> data = np.random.rand(32, 32, 13).astype(np.float32)
        >>> temp = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> fpath = temp.name
        >>> kwimage.imwrite(fpath, data)
        >>> recon = kwimage.imread(fpath)
        >>> assert np.all(recon == data)
        >>> kwimage.imwrite(fpath, data, backend='skimage')
        >>> recon = kwimage.imread(fpath, backend='skimage')
        >>> assert np.all(recon == data)
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # gdal should error when trying to read an image written by skimage
        >>> with pytest.raises(NotImplementedError):
        >>>     kwimage.imread(fpath, backend='gdal')
        >>> # In this case the backend will resolve to cv2, and thus we expect
        >>> # a failure
        >>> temp.close()
        >>> temp = tempfile.NamedTemporaryFile(suffix='.png')
        >>> fpath = temp.name
        >>> with pytest.raises(NotImplementedError):
        >>>     kwimage.imwrite(fpath, data)
        >>> temp.close()

    Example:
        >>> import ubelt as ub
        >>> import kwimage
        >>> dpath = ub.Path(ub.ensure_app_cache_dir('kwimage/badwrite'))
        >>> dpath.delete().ensuredir()
        >>> imdata = kwimage.ensure_uint255(kwimage.grab_test_image())[:, :, 0]
        >>> import pytest
        >>> fpath = dpath / 'does-not-exist/img.jpg'
        >>> with pytest.raises(IOError):
        ...     kwimage.imwrite(fpath, imdata, backend='cv2')
        >>> with pytest.raises(IOError):
        ...     kwimage.imwrite(fpath, imdata, backend='skimage')
        >>> # xdoctest: +SKIP
        >>> # TODO: run tests conditionally
        >>> with pytest.raises(IOError):
        ...     kwimage.imwrite(fpath, imdata, backend='gdal')
        >>> with pytest.raises((IOError, RuntimeError)):
        ...     kwimage.imwrite(fpath, imdata, backend='itk')
    """
    fpath = os.fspath(fpath)

    if backend == 'auto':
        _fpath_lower = fpath.lower()
        if _fpath_lower.endswith(('.tif', '.tiff')):
            if _have_gdal():
                backend = 'gdal'
            else:
                backend = 'skimage'
        elif _fpath_lower.endswith(GDAL_EXTENSIONS):
            if _have_gdal():
                backend = 'gdal'
        elif _fpath_lower.endswith(ITK_EXTENSIONS):
            backend = 'itk'
        else:
            backend = 'cv2'

    if space == 'auto':
        if backend != 'cv2':
            # For non-cv2 backends, we can read / writ the image as is
            # without worrying about channel ordering conversions
            space = None

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
            flag = cv2.imwrite(fpath, image, **kwargs)
        except cv2.error as ex:
            if 'could not find a writer for the specified extension' in str(ex):
                raise ValueError(
                    'Image fpath {!r} does not have a known image extension '
                    '(e.g. png/jpg)'.format(fpath))
            else:
                raise
        else:
            # TODO: generalize error handling and diagnostics for all backends
            if not flag:
                if not exists(dirname(fpath)):
                    raise IOError((
                        'kwimage failed to write with opencv backend. '
                        'Reason: destination fpath {!r} is in a directory that '
                        'does not exist.').format(fpath))
                else:
                    raise IOError(
                        'kwimage failed to write with opencv backend. '
                        'Reason: unknown.')

    elif backend == 'skimage':
        import skimage.io
        skimage.io.imsave(fpath, image, **kwargs)
    elif backend == 'gdal':
        _imwrite_cloud_optimized_geotiff(fpath, image, **kwargs)
    elif backend == 'pil':
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img.save(fpath)
    elif backend == 'itk':
        import itk
        itk_obj = itk.image_view_from_array(image)
        itk.imwrite(itk_obj, fpath, **kwargs)
    elif backend == 'turbojpeg':
        raise NotImplementedError
    else:
        raise KeyError('Unknown imwrite backend={!r}'.format(backend))

    return fpath


def load_image_shape(fpath, backend='auto'):
    """
    Determine the height/width/channels of an image without reading the entire
    file.

    Args:
        fpath (str): path to an image
        backend (str): can be "auto", "pil", or "gdal".

    Returns:
        Tuple[int, int, int] - shape of the image
            Recall this library uses the convention that "shape" is refers to
            height,width,channels array-style ordering and "size" is
            width,height cv2-style ordering.

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test the loading the shape works the same as loading the image and
        >>> # testing the shape
        >>> import kwimage
        >>> import tempfile
        >>> temp_dir = tempfile.TemporaryDirectory()
        >>> temp_dpath = ub.Path(temp_dir.name)
        >>> data = kwimage.grab_test_image()
        >>> datas = {
        >>>     'rgb255': kwimage.ensure_uint255(data),
        >>>     'rgb01': kwimage.ensure_float01(data),
        >>>     'rgba01': kwimage.ensure_alpha_channel(data),
        >>> }
        >>> results = {}
        >>> # These should be consistent
        >>> # The was a problem where CV2_IMREAD_UNCHANGED read the alpha band,
        >>> # but PIL did not, but maybe this is fixed now?
        >>> for key, imdata in datas.items():
        >>>     fpath = temp_dpath / f'{key}.png'
        >>>     kwimage.imwrite(fpath, imdata)
        >>>     shapes = {}
        >>>     shapes['pil_load_shape'] = kwimage.load_image_shape(fpath, backend='pil')
        >>>     shapes['gdal_load_shape'] = kwimage.load_image_shape(fpath, backend='gdal')
        >>>     shapes['auto_load_shape'] = kwimage.load_image_shape(fpath, backend='auto')
        >>>     shapes['pil'] = kwimage.imread(fpath, backend='pil').shape
        >>>     shapes['cv2'] = kwimage.imread(fpath, backend='cv2').shape
        >>>     shapes['gdal'] = kwimage.imread(fpath, backend='gdal').shape
        >>>     shapes['skimage'] = kwimage.imread(fpath, backend='skimage').shape
        >>>     results[key] = shapes
        >>> print('results = {}'.format(ub.repr2(results, nl=2, align=':', sort=0)))
        >>> for shapes in results.values():
        >>>     assert ub.allsame(shapes.values())
        >>> temp_dir.cleanup()

    Benchmark:
        >>> # For large files, PIL is much faster
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> from osgeo import gdal
        >>> from PIL import Image
        >>> import timerit
        >>> #
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image_fpath()
        >>> #
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
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

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> import ubelt as ub
        >>> import kwimage
        >>> dpath = ub.Path.appdir('kwimage/tests', type='cache').ensuredir()
        >>> fpath = dpath / 'foo.tif'
        >>> kwimage.imwrite(fpath, np.random.rand(64, 64, 3))
        >>> shape = kwimage.load_image_shape(fpath)
        >>> assert shape == (64, 64, 3)

    Ignore:
        * Note: this seems to have an issue with PNG's with mode='LA',
          which means that there really are two underlying channels, but it
          kwimage.imread cv2 backend reads it as a 4 channel RGBA array.
    """
    if backend == 'auto':
        try:
            shape = load_image_shape(fpath, backend='pil')
        except Exception as pil_ex:
            if not _have_gdal():
                raise
            try:
                shape = load_image_shape(fpath, backend='gdal')
            except Exception:
                raise pil_ex
    elif backend == 'pil':
        from PIL import Image
        fpath = os.fspath(fpath)
        with Image.open(fpath) as pil_img:
            width, height = pil_img.size
            num_channels = len(pil_img.getbands())
        shape = (height, width, num_channels)
    elif backend == 'gdal':
        from osgeo import gdal
        fpath = os.fspath(fpath)
        gdal_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
        if gdal_dset is None:
            raise Exception(gdal.GetLastErrorMsg())
        width = gdal_dset.RasterXSize
        height = gdal_dset.RasterYSize
        num_channels = gdal_dset.RasterCount
        gdal_dset = None
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
        from osgeo import gdal  # NOQA
    except Exception:
        return False
    else:
        return True


def _imwrite_cloud_optimized_geotiff(fpath, data, compress='auto',
                                     blocksize=256, overviews=None,
                                     overview_resample='NEAREST',
                                     interleave='PIXEL',
                                     options=None,
                                     nodata=None,
                                     nodata_value=None,
                                     crs=None, transform=None):
    """
    Writes data as a cloud-optimized geotiff using gdal

    Args:
        fpath (PathLike): file path to save the COG to.

        data (ndarray[ndim=3]): Raw HWC image data to save. Dimensions should
            be height, width, channels.

        compress (bool): Can be JPEG (lossy) or LZW (lossless),
            or DEFLATE (lossless). Can also be 'auto', which will try to
            heuristically choose a sensible choice.

        blocksize (int): size of tiled blocks

        overviews (None | str | int | list):
            If specified as a string, can be 'auto'.
            if specified as a list, then uses exactly those overviews. If
            specified as an integer a list is created using powers of two.

        overview_resample (str): resampling method for
            overview pyramid. Valid choices are: 'NEAREST', 'AVERAGE',
            'BILINEAR', 'CUBIC', 'CUBICSPLINE', 'LANCZOS'.

        options (List[str]): other gdal options. See [GDAL_GTiff_Options]_ for
            details.

        nodata_value (int):
            if specified, uses this as the nodata value for each band.

        transform (kwimage.Affine):
            An affine transform from image coordinates into a specified
            coordinate reference system (must set crs).

        crs (str):
            The coordinate reference system for the geo_transform.

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
        .. [GDAL_GTiff_Options] https://gdal.org/drivers/raster/gtiff.html
        https://gdal.org/drivers/raster/cog.html

    Note:
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

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # Test GDAL options
        >>> from kwimage.im_io import *  # NOQA
        >>> from kwimage.im_io import _imwrite_cloud_optimized_geotiff
        >>> import kwimage
        >>> import tempfile
        >>> data = kwimage.grab_test_image()
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> fpath = tmp_tif.name
        >>> kwimage.imwrite(fpath, data, compress='LZW', interleave='PIXEL', blocksize=64, options=['NUM_THREADS=ALL_CPUS'])
        >>> _ = ub.cmd('gdalinfo ' + fpath, verbose=3)
        >>> loaded = kwimage.imread(fpath)
        >>> assert np.all(loaded.ravel() == data.ravel())
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> dinfo = np.iinfo(np.uint16)
        >>> kwplot.imshow(loaded / dinfo.max)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(module:osgeo)
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # Test GDAL options
        >>> from kwimage.im_io import *  # NOQA
        >>> from kwimage.im_io import _imwrite_cloud_optimized_geotiff
        >>> import kwimage
        >>> import tempfile
        >>> orig_data = kwimage.grab_test_image()
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> fpath = tmp_tif.name
        >>> imwrite_param_basis = {
        >>>     'interleave': ['BAND', 'PIXEL'],
        >>>     'compress': ['NONE', 'DEFLATE'],
        >>>     'blocksize': [64, 128, None],
        >>>     'overviews': [None, 'auto'],
        >>> }
        >>> data_param_basis = {
        >>>     'dsize': [(256, 256), (532, 202)],
        >>>     'dtype': ['float32', 'uint8'],
        >>> }
        >>> data_param_grid = list(ub.named_product(data_param_basis))
        >>> imwrite_param_grid = list(ub.named_product(imwrite_param_basis))
        >>> for data_kwargs in data_param_grid:
        >>>     data = kwimage.imresize(orig_data, dsize=data_kwargs['dsize'])
        >>>     data = data.astype(data_kwargs['dtype'])
        >>>     for imwrite_kwargs in imwrite_param_grid:
        >>>         print('data_kwargs = {}'.format(ub.repr2(data_kwargs, nl=1)))
        >>>         print('imwrite_kwargs = {}'.format(ub.repr2(imwrite_kwargs, nl=1)))
        >>>         kwimage.imwrite(fpath, data, **imwrite_kwargs)
        >>>         _ = ub.cmd('gdalinfo ' + fpath, verbose=3)
        >>>         loaded = kwimage.imread(fpath)
        >>>         assert np.all(loaded == data)
        >>> tmp_tif.close()
    """
    from osgeo import gdal
    if len(data.shape) == 2:
        data = data[:, :, None]

    if options is None:
        options = []

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
    if isinstance(overviews, str):
        if overviews == 'auto':
            smallest_overview_dim = 512
            # Compute as many overviews as needed to get both dimensions < smallest_overview_dim
            # unless that would cause one dimension to disolve
            max_x_overviews = int(np.log2(y_size))
            max_y_overviews = int(np.log2(x_size))
            max_overviews = min(max_x_overviews, max_y_overviews)
            y_overviews = y_size // smallest_overview_dim
            x_overviews = x_size // smallest_overview_dim
            request_overviews = max(y_overviews, x_overviews)
            overviews = min(max_overviews, request_overviews)

    if overviews is None:
        overviewlist = []
    elif isinstance(overviews, int):
        overviewlist = (2 ** np.arange(1, overviews + 1)).tolist()
    else:
        overviewlist = overviews

    _options = [
        # We are still using the GTiff Driver instead of COG to have control
        # over interleave
        'BIGTIFF=YES',
    ]
    if blocksize is not None:
        _options = [
            'TILED=YES',
            'BLOCKXSIZE={}'.format(blocksize),
            'BLOCKYSIZE={}'.format(blocksize),
        ]

    if compress == 'RAW':
        compress = 'NONE'

    _options += ['COMPRESS={}'.format(compress)]
    if compress == 'JPEG' and num_bands == 3:
        # Using YCBCR speeds up jpeg compression by quite a bit
        _options += ['PHOTOMETRIC=YCBCR']

    # https://gdal.org/drivers/raster/gtiff.html#creation-options
    if interleave == 'BAND':
        # For 1-band images I don' think this matters?
        _options += ['INTERLEAVE=BAND']
    elif interleave == 'PIXEL':
        _options += ['INTERLEAVE=PIXEL']
    else:
        raise KeyError(interleave)

    if overviewlist:
        _options.append('COPY_SRC_OVERVIEWS=YES')

    _options += options

    _options = list(map(str, _options))  # python2.7 support

    # Create an in-memory dataset where we will prepare the COG data structure
    driver = gdal.GetDriverByName(str('MEM'))
    data_set = driver.Create(str(''), x_size, y_size, num_bands, eType=eType)

    if transform is not None or crs is not None:
        import affine
        # TODO: add ability to add RPC
        if crs is None or transform is None:
            raise ValueError('Specify transform and crs together')
        # TODO: Allow transform to be a normal gdal object or something
        # coercable to an affine object.
        a, b, c, d, e, f = transform.matrix.ravel()[0:6]
        aff = affine.Affine(a, b, c, d, e, f)
        data_set.SetProjection(crs)
        data_set.SetGeoTransform(aff.to_gdal())

    if nodata is not None:
        ub.schedule_deprecation(
            modname='kwimage', name='nodata',
            type='argument to _imwrite_gdal',
            migration='use nodata_value instead',
            deprecate='0.9.5', error='0.10.0', remove='0.11.0')

    if nodata_value is None:
        nodata_value = nodata

    for i in range(num_bands):
        band_data = np.ascontiguousarray(data[:, :, i])
        band = data_set.GetRasterBand(i + 1)
        band.WriteArray(band_data)
        if nodata_value is not None:
            band.SetNoDataValue(nodata_value)
        # TODO:
        # could set the color interpretation here
        band = None

    if overviewlist:
        # Build the downsampled overviews (for fast zoom in / out)
        data_set.BuildOverviews(str(overview_resample), overviewlist)

    driver = None
    # Copy the in-memory dataset to an on-disk GeoTiff
    driver2 = gdal.GetDriverByName(str('GTiff'))
    data_set2 = driver2.CreateCopy(fpath, data_set, options=_options)
    data_set = None

    # OK, so setting things to None turns out to be important. Gah!
    # NOTE: if data_set2 is None here, that may be because the directory
    # we are trying to write to does not exist.
    if data_set2 is None:
        last_gdal_error = gdal.GetLastErrorMsg()
        if 'No such file or directory' in last_gdal_error:
            ex_cls = IOError
        else:
            ex_cls = Exception
        raise ex_cls(
            'Unable to create gtiff driver for fpath={}, options={}, last_gdal_error={}'.format(
                fpath, _options, last_gdal_error))
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
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2)) == 'DEFLATE'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 3).astype(np.uint8)) == 'DEFLATE'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 4).astype(np.uint8)) == 'DEFLATE'
        >>> assert _gdal_auto_compress(data=np.random.rand(3, 2, 1).astype(np.uint8)) == 'DEFLATE'
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

            # if num_channels is None:
            #     data_set.RasterCount == 3

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
        compress = 'DEFLATE'
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


def _imread_svg(fpath):
    """
    References:
        https://pypi.org/project/svglib/

    Ignore:
        # xdoctest: +REQUIRES(module:svglib)
        # xdoctest: +REQUIRES(module:reportlab)
        from kwimage.im_io import *  # NOQA
        from kwimage.im_io import _imread_svg  # NOQA
        import kwimage
        fpath = ub.grabdata('https://upload.wikimedia.org/wikipedia/commons/a/aa/Philips_PM5544.svg')
        # This doesnt work quite how I would expect it to.
        imdata, _, _ = _imread_svg(fpath)
        image = kwimage.imread(fpath)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(image)
    """
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg
    import io
    from PIL import Image
    file = io.BytesIO()
    drawing = svg2rlg(fpath)
    renderPM.drawToFile(drawing, file, fmt="PNG")
    file.seek(0)
    pil_img = Image.open(file)
    imdata = np.asarray(pil_img)
    src_space = auto_dst_space = 'rgb'
    return imdata, src_space, auto_dst_space
