import numpy as np
import cv2


try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile


def radial_fourier_mask(img_hwc, radius=11, axis=None, clip=None):
    """
    In [1] they use a radius of 11.0 on CIFAR-10.

    Args:
        img_hwc (ndarray): assumed to be float 01

    References:
        [1] Jo and Bengio "Measuring the tendency of CNNs to Learn Surface Statistical Regularities" 2017.
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

    Example:
        >>> from kwimage.im_filter import *  # NOQA
        >>> import kwimage
        >>> img_hwc = kwimage.grab_test_image()
        >>> img_hwc = kwimage.ensure_float01(img_hwc)
        >>> out_hwc = radial_fourier_mask(img_hwc, radius=11)
        >>> # xdoc: REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> def keepdim(func):
        >>>     def _wrap(im):
        >>>         needs_transpose = (im.shape[0] == 3)
        >>>         if needs_transpose:
        >>>             im = im.transpose(1, 2, 0)
        >>>         out = func(im)
        >>>         if needs_transpose:
        >>>             out = out.transpose(2, 0, 1)
        >>>         return out
        >>>     return _wrap
        >>> @keepdim
        >>> def rgb_to_lab(im):
        >>>     return kwimage.convert_colorspace(im, src_space='rgb', dst_space='lab')
        >>> @keepdim
        >>> def lab_to_rgb(im):
        >>>     return kwimage.convert_colorspace(im, src_space='lab', dst_space='rgb')
        >>> @keepdim
        >>> def rgb_to_yuv(im):
        >>>     return kwimage.convert_colorspace(im, src_space='rgb', dst_space='yuv')
        >>> @keepdim
        >>> def yuv_to_rgb(im):
        >>>     return kwimage.convert_colorspace(im, src_space='yuv', dst_space='rgb')
        >>> def show_data(img_hwc):
        >>>     # dpath = ub.ensuredir('./fouriertest')
        >>>     kwplot.imshow(img_hwc, fnum=1)
        >>>     pnum_ = kwplot.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = radial_fourier_mask(img_hwc, r, clip=(0, 1))
        >>>         kwplot.imshow(imgt, pnum=pnum_(), fnum=2)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>     kwplot.set_figtitle('RGB')
        >>>     # plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('rgb', x)))
        >>>     pnum_ = kwplot.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = lab_to_rgb(radial_fourier_mask(rgb_to_lab(img_hwc), r))
        >>>         kwplot.imshow(imgt, pnum=pnum_(), fnum=3)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>     kwplot.set_figtitle('LAB')
        >>>     # plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('lab', x)))
        >>>     pnum_ = kwplot.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = yuv_to_rgb(radial_fourier_mask(rgb_to_yuv(img_hwc), r))
        >>>         kwplot.imshow(imgt, pnum=pnum_(), fnum=4)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>     kwplot.set_figtitle('YUV')
        >>>     # plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('yuv', x)))
        >>> show_data(img_hwc)
        >>> kwplot.show_if_requested()
    """
    import cv2
    rows, cols = img_hwc.shape[0:2]

    diam = radius * 2
    left = int(np.floor((cols - diam) / 2))
    right = int(np.ceil((cols - diam) / 2))
    top = int(np.floor((rows - diam) / 2))
    bot = int(np.ceil((rows - diam) / 2))

    # element = skimage.morphology.disk(radius)
    # mask = np.pad(element, ((top, bot), (left, right)), 'constant')
    if diam > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
        mask = cv2.copyMakeBorder(element, top, bot, left, right,
                                  cv2.BORDER_CONSTANT, value=0)
    else:
        mask = 0

    out_hwc = fourier_mask(img_hwc, mask, axis=axis, clip=clip)

    return out_hwc


@profile
def fourier_mask(img_hwc, mask, axis=None, clip=None, backend='cv2'):
    """
    Applies a mask to the fourier spectrum of an image

    Args:
        img_hwc (ndarray): assumed to be float 01
        mask (ndarray): mask used to modulate the image in the fourier domain.
             Usually these are boolean values (hence the name mask), but any
             numerical value is technically allowed.
         backend (str):
             which implementation of DFT to use. Can be 'cv2' or 'numpy'.
             Defaults to cv2.

    CommandLine:
        XDEV_PROFILE=1 xdoctest -m kwimage.im_filter fourier_mask --show

    import kwimage
    img_hwc = kwimage.grab_test_image(space='gray')
    mask = np.random.rand(*img_hwc.shape[0:2])
    out_hwc = fourier_mask(img_hwc, mask)
    for timer in ti.reset('fft mask with numpy'):
        with timer:
            fourier_mask(out_hwc, mask, backend='numpy')

    for timer in ti.reset('fft mask with cv2'):
        with timer:
            fourier_mask(out_hwc, mask, backend='cv2')

    Example:
        >>> from kwimage.im_filter import *  # NOQA
        >>> import kwimage
        >>> img_hwc = kwimage.grab_test_image(space='gray')
        >>> mask = np.random.rand(*img_hwc.shape[0:2])
        >>> mask = (kwimage.gaussian_blur(mask) > 0.5)
        >>> out_hwc_cv2 = fourier_mask(img_hwc, mask, backend='numpy')
        >>> out_hwc_np = fourier_mask(img_hwc, mask, backend='cv2')
        >>> # xdoc: REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img_hwc, pnum=(1, 4, 1), fnum=1, title='input')
        >>> kwplot.imshow(mask, pnum=(1, 4, 2), fnum=1, title='mask')
        >>> kwplot.imshow(out_hwc_cv2, pnum=(1, 4, 3), fnum=1, title='numpy')
        >>> kwplot.imshow(out_hwc_np, pnum=(1, 4, 4), fnum=1, title='cv2')
        >>> kwplot.show_if_requested()

    Example:
        >>> from kwimage.im_filter import *  # NOQA
        >>> import kwimage
        >>> img_hwc = kwimage.grab_test_image(space='gray')
        >>> mask = kwimage.gaussian_patch(img_hwc.shape[0:2])
        >>> mask = (mask / mask.max()) ** 32
        >>> out_hwc_cv2 = fourier_mask(img_hwc, mask, backend='numpy')
        >>> out_hwc_np = fourier_mask(img_hwc, mask, backend='cv2')
        >>> # xdoc: REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img_hwc, pnum=(1, 4, 1), fnum=1, title='input')
        >>> kwplot.imshow(mask, pnum=(1, 4, 2), fnum=1, title='mask')
        >>> kwplot.imshow(out_hwc_cv2, pnum=(1, 4, 3), fnum=1, title='numpy')
        >>> kwplot.imshow(out_hwc_np, pnum=(1, 4, 4), fnum=1, title='cv2')
        >>> kwplot.show_if_requested()

    Example:
        >>> from kwimage.im_filter import *  # NOQA
        >>> import kwimage
        >>> img_hwc = kwimage.grab_test_image(space='gray')
        >>> mask = kwimage.gaussian_patch(img_hwc.shape[0:2])
        >>> mask = 1 - (mask / mask.max()) ** 32
        >>> out_hwc_cv2 = fourier_mask(img_hwc, mask, backend='numpy')
        >>> out_hwc_np = fourier_mask(img_hwc, mask, backend='cv2')
        >>> # xdoc: REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img_hwc, pnum=(1, 4, 1), fnum=1, title='input')
        >>> kwplot.imshow(mask, pnum=(1, 4, 2), fnum=1, title='mask')
        >>> kwplot.imshow(out_hwc_cv2, pnum=(1, 4, 3), fnum=1, title='numpy')
        >>> kwplot.imshow(out_hwc_np, pnum=(1, 4, 4), fnum=1, title='cv2')
        >>> kwplot.show_if_requested()
    """
    import kwarray
    img_hwc = kwarray.atleast_nd(img_hwc, 3, front=False)
    img_chw = img_hwc.transpose(2, 0, 1)

    if backend == 'numpy':
        _fourier = _np_fourier
        _inv_fourier = _np_inv_fourier
    elif backend == 'cv2':
        mask = kwarray.atleast_nd(mask, 3, front=False)
        _fourier = _cv2_fourier
        _inv_fourier = _cv2_inv_fourier
    else:
        raise Exception

    out_chw = np.empty_like(img_chw)
    if axis is None:
        for i, s in enumerate(img_chw):
            # hadamard product (aka simple element-wise multiplication)
            f = _fourier(s)
            f *= mask
            # f = _fourier(s) * mask
            out_chw[i] = _inv_fourier(f)
    else:
        for i, s in enumerate(img_chw):
            if i in axis:
                # hadamard product (aka simple element-wise multiplication)
                f = _fourier(s)
                f *= mask
                out_chw[i] = _inv_fourier(f)
            else:
                out_chw[i] = s

    if clip:
        out_chw = np.clip(out_chw, *clip, out=out_chw)
    out_hwc = out_chw.transpose(1, 2, 0)
    return out_hwc


def _np_fourier(s):
    return np.fft.fftshift(np.fft.fft2(s))


def _np_inv_fourier(f):
    # use real because LAB has negative components
    return np.real(np.fft.ifft2(np.fft.ifftshift(f)))


def _cv2_fourier(s):
    return np.fft.fftshift(cv2.dft(s.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT))


def _cv2_inv_fourier(f):
    # Real part will be in first element of the last dim
    return cv2.idft(np.fft.ifftshift(f), flags=cv2.DFT_SCALE)[..., 0]


def _benchmark():
    """
    References:
        https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    """
    import kwimage
    img_hwc = kwimage.grab_test_image(space='gray')

    s = img_hwc[..., 0].astype(np.float32)
    # s = np.arange(0, 16).reshape(4, 4)

    import cv2

    a1_complex = np.fft.fft2(s).astype(np.complex64)
    b1_complex = np.fft.fftshift(a1_complex)
    c1_complex = np.fft.ifftshift(b1_complex)
    d1_complex = np.fft.ifft2(c1_complex)

    a1_twodim = np.stack([np.real(a1_complex), np.imag(a1_complex)], axis=2)
    b1_twodim = np.stack([np.real(b1_complex), np.imag(b1_complex)], axis=2)
    c1_twodim = np.stack([np.real(c1_complex), np.imag(c1_complex)], axis=2)
    d1_twodim = np.stack([np.real(d1_complex), np.imag(d1_complex)], axis=2)

    a2_twodim = cv2.dft(s.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    b2_twodim = np.fft.fftshift(a2_twodim)[..., ::-1]
    c2_twodim = np.fft.ifftshift(b2_twodim[..., ::-1])
    d2_twodim = cv2.idft(c2_twodim, flags=cv2.DFT_SCALE)

    import ubelt as ub
    print('\nA1/A2')
    print(ub.hzcat(['a1/a2 : ', ub.urepr(a1_twodim), ub.urepr(a2_twodim)]))

    # The results here seem flipped for some reason, and I don't quite
    # understand why. I've adjust the above code to flip dim=2 so it does
    # agree.
    print('\nB1/B2')
    print(ub.hzcat(['b1/b2 : ', ub.urepr(b1_twodim), ub.urepr(b2_twodim)]))

    print('\nC1/C2')
    print(ub.hzcat(['c1/c2 : ', ub.urepr(c1_twodim), ub.urepr(c2_twodim)]))
    print('\nD1/D2')
    print(ub.hzcat(['d1/d2 : ', ub.urepr(d1_twodim), ub.urepr(d2_twodim)]))

    import kwarray
    a_delta = kwarray.stats_dict(a1_twodim - a2_twodim)
    b_delta = kwarray.stats_dict(b1_twodim - b2_twodim)
    c_delta = kwarray.stats_dict(c1_twodim - c2_twodim)
    d_delta = kwarray.stats_dict(d1_twodim - d2_twodim)
    print('a_delta = {}'.format(ub.urepr(a_delta, nl=1)))
    print('b_delta = {}'.format(ub.urepr(b_delta, nl=1)))
    print('c_delta = {}'.format(ub.urepr(c_delta, nl=1)))
    print('d_delta = {}'.format(ub.urepr(d_delta, nl=1)))

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('np fft'):
        with timer:
            fs_np = _np_fourier(s)

    for timer in ti.reset('np ifft'):
        with timer:
            _np_inv_fourier(fs_np)

    for timer in ti.reset('cv2 fft'):
        with timer:
            fs_cv2 = _cv2_fourier(s)

    for timer in ti.reset('cv2 ifft'):
        with timer:
            _cv2_inv_fourier(fs_cv2)


def _benchmark2():
    import kwimage
    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=3)
    img_hwc = kwimage.grab_test_image(space='gray')
    mask = np.random.rand(*img_hwc.shape[0:2])
    out_hwc = fourier_mask(img_hwc, mask)

    for timer in ti.reset('fft mask with cv2'):
        with timer:
            fourier_mask(out_hwc, mask, backend='cv2')

    # for timer in ti.reset('fft mask with numpy'):
    #     with timer:
    #         fourier_mask(out_hwc, mask, backend='numpy')
