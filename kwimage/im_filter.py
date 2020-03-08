import numpy as np


def radial_fourier_mask(img_hwc, radius=11, axis=None, clip=None):
    """
    In [1] they use a radius of 11.0 on CIFAR-10.

    Args:
        img_hwc (ndarray): assumed to be float 01

    References:
        [1] Jo and Bengio "Measuring the tendency of CNNs to Learn Surface Statistical Regularities" 2017.
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

    Example:
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


def fourier_mask(img_hwc, mask, axis=None, clip=None):
    """
    Applies a mask to the fourier spectrum of an image

    Args:
        img_hwc (ndarray): assumed to be float 01
        mask (ndarray): mask used to modulate the image in the fourier domain.
             Usually these are boolean values (hence the name mask), but any
             numerical value is technically allowed.

    CommandLine:
        xdoctest -m kwimage.im_filter fourier_mask --show

    Example:
        >>> import kwimage
        >>> img_hwc = kwimage.grab_test_image(space='gray')
        >>> mask = np.random.rand(*img_hwc.shape[0:2])
        >>> out_hwc = fourier_mask(img_hwc, mask)
        >>> # xdoc: REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img_hwc, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(out_hwc, pnum=(1, 2, 2), fnum=1)
        >>> kwplot.show_if_requested()
    """
    import kwarray
    img_hwc = kwarray.atleast_nd(img_hwc, 3, front=False)
    img_chw = img_hwc.transpose(2, 0, 1)
    def fourier(s):
        # note: cv2 functions would probably be faster here
        return np.fft.fftshift(np.fft.fft2(s))

    def inv_fourier(f):
        # use real because LAB has negative components
        return np.real(np.fft.ifft2(np.fft.ifftshift(f)))

    out_chw = np.empty_like(img_chw)
    if axis is None:
        for i, s in enumerate(img_chw):
            # hadamard product (aka simple element-wise multiplication)
            out_chw[i] = inv_fourier(fourier(s) * mask)
    else:
        for i, s in enumerate(img_chw):
            if i in axis:
                # hadamard product (aka simple element-wise multiplication)
                out_chw[i] = inv_fourier(fourier(s) * mask)
            else:
                out_chw[i] = s
    if clip:
        out_chw = np.clip(out_chw, *clip)
    out_hwc = out_chw.transpose(1, 2, 0)
    return out_hwc
