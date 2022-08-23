"""
Functions for stacking images (of potentially different sizes) together in a
single image.

Notes:
    * We may change the "bg_value" argument to "bg_color" in the future.
"""

import cv2
import numpy as np
import skimage
import skimage.transform
from . import im_core
from . import im_cv2


def stack_images(images, axis=0, resize=None, interpolation=None, overlap=0,
                 return_info=False, bg_value=None, pad=None,
                 allow_casting=True):
    """
    Make a new image with the input images side-by-side

    Args:
        images (Iterable[ndarray]):  image data

        axis (int): axis to stack on (either 0 or 1)

        resize (int | str | None):
            if None image sizes are not modified, otherwise resize resize can
            be either 0 or 1.  We resize the `resize`-th image to match the
            `1 - resize`-th image. In other words, resize=0 means the current
            image is resized to match the next image, and resize=1 means the
            next image is resized to match the current image.
            Can also be strings "larger" or "smaller". If resize="larger", the
            current image is only resized if the next image is larger. If
            resize='smaller' the current image is resized only if the next
            image is smaller.
            TODO: this needs to be reworked to be more intuitive.

        interpolation (int | str):
            string or cv2-style interpolation type.
            only used if resize or overlap > 0

        overlap (int):
            number of pixels to overlap. Using a negative number results in a
            border.

        pad (int):
            if specified overrides `overlap` as a the number of pixels to pad
            between images.

        return_info (bool):
            if True, returns transforms (scales and translations) to map from
            original image to its new location.

        bg_value (Number | ndarray | str):
            background value or color, if specified, uses this as a fill value.

        allow_casting (bool):
            if True, then if "uint255" and "float01" format images are given
            they are converted to "float01".  Defaults to True.

    Returns:
        ndarray: an image of stacked images side by side

        Tuple[ndarray, List]: where the first item is the aformentioned stacked
            image and the second item is a list of transformations for each
            input image mapping it to its location in the returned image.

    SeeAlso:
        :func:`kwimage.im_stack.stack_images_grid`

    TODO:
        - [ ] This is currently implemented by calling the "stack_two_images"
              function multiple times. This should be optimized by allocating
              the entire canvas first.

    Example:
        >>> import kwimage
        >>> img1 = kwimage.grab_test_image('carl', space='rgb')
        >>> img2 = kwimage.grab_test_image('astro', space='rgb')
        >>> images = [img1, img2]
        >>> imgB, transforms = kwimage.stack_images(
        >>>     images, axis=0, resize='larger', pad=10, return_info=True)
        >>> print('imgB.shape = {}'.format(imgB.shape))
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> kwplot.imshow(imgB, colorspace='rgb')
        >>> wh1 = np.multiply(img1.shape[0:2][::-1], transforms[0].scale)
        >>> wh2 = np.multiply(img2.shape[0:2][::-1], transforms[1].scale)
        >>> xoff1, yoff1 = transforms[0].translation
        >>> xoff2, yoff2 = transforms[1].translation
        >>> xywh1 = (xoff1, yoff1, wh1[0], wh1[1])
        >>> xywh2 = (xoff2, yoff2, wh2[0], wh2[1])
        >>> kwplot.draw_boxes(kwimage.Boxes([xywh1], 'xywh'), color=(1.0, 0, 0))
        >>> kwplot.draw_boxes(kwimage.Boxes([xywh2], 'xywh'), color=(1.0, 0, 0))
        >>> kwplot.show_if_requested()
    """
    imgiter = iter(images)
    img1 = next(imgiter)

    if pad is not None:
        overlap = -pad

    if return_info:
        transforms_ = [skimage.transform.AffineTransform(
            scale=[1.0, 1.0], translation=[0.0, 0.0]
        )]

    for img2 in imgiter:
        img1, offset_tup, sf_tup = _stack_two_images(img1, img2, axis=axis,
                                                     resize=resize,
                                                     bg_value=bg_value,
                                                     interpolation=interpolation,
                                                     overlap=overlap,
                                                     allow_casting=allow_casting)
        if return_info:
            off1, off2 = offset_tup
            sf1, sf2 = sf_tup

            tf1 = skimage.transform.AffineTransform(scale=sf1, translation=off1)
            tf2 = skimage.transform.AffineTransform(scale=sf2, translation=off2)

            # Apply transforms to the first image to all previous transforms
            for t in transforms_:
                t.params = t.params.dot(tf1.params)
            # Append the second transform
            transforms_.append(tf2)

    if return_info:
        return img1, transforms_
    else:
        return img1


def stack_images_grid(images, chunksize=None, axis=0, overlap=0, pad=None,
                      return_info=False, bg_value=None, resize=None,
                      allow_casting=True):
    """
    Stacks images in a grid. Optionally return transforms of original image
    positions in the output image.

    Args:
        images (Iterable[ndarray]):  image data

        chunksize (int): number of rows per column or columns per
            row depending on the value of `axis`.
            If unspecified, computes this as `int(sqrt(len(images)))`.

        axis (int):
            If 0, chunksize is columns per row.
            If 1, chunksize is rows per column.
            Defaults to 0.

        overlap (int): number of pixels to overlap. Using a negative
            number results in a border.

        pad (int): if specified overrides `overlap` as a the number of pixels
            to pad between images.

        return_info (bool): if True, returns transforms (scales and
            translations) to map from original image to its new location.

        resize (int | str | None):
            if None image sizes are not modified, otherwise can be set to
            "larger" or "smaller" to resize the images in each stack direction.

        bg_value (Number | ndarray | str):
            background value or color, if specified, uses this as a fill value.

        allow_casting (bool):
            if True, then if "uint255" and "float01" format images are given
            they are converted to "float01".  Defaults to True.

    Returns:
        ndarray: an image of stacked images in a grid pattern

        Tuple[ndarray, List]: where the first item is the aformentioned stacked
            image and the second item is a list of transformations for each
            input image mapping it to its location in the returned image.

    SeeAlso:
        :func:`kwimage.im_stack.stack_images`

    Example:
        >>> import kwimage
        >>> img1 = kwimage.grab_test_image('carl')
        >>> img2 = kwimage.grab_test_image('astro')
        >>> img3 = kwimage.grab_test_image('airport')
        >>> img4 = kwimage.grab_test_image('paraview')[..., 0:3]
        >>> img5 = kwimage.grab_test_image('pm5644')
        >>> images = [img1, img2, img3, img4, img5]
        >>> canvas, transforms = kwimage.stack_images_grid(
        ...     images, chunksize=3, axis=0, pad=10, bg_value='kitware_blue',
        ...     return_info=True, resize='larger')
        >>> print('canvas.shape = {}'.format(canvas.shape))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """
    import ubelt as ub
    if chunksize is None:
        chunksize = int(len(images) ** .5)
    if pad is not None:
        overlap = -pad
    stack1_list = []
    tfs1_list = []
    assert axis in [0, 1]
    for batch in ub.chunks(images, chunksize, bordermode='none'):
        stack1, tfs1 = stack_images(batch, overlap=overlap, return_info=True,
                                    bg_value=bg_value,
                                    resize=resize, axis=1 - axis)
        tfs1_list.append(tfs1)
        stack1_list.append(stack1)

    img_grid, tfs2 = stack_images(stack1_list, overlap=overlap,
                                  bg_value=bg_value,
                                  return_info=True, axis=axis)
    transforms_ = [tf1 + tf2
                   for tfs1, tf2 in zip(tfs1_list, tfs2)
                   for tf1 in tfs1]

    if return_info:
        return img_grid, transforms_
    else:
        return img_grid


def _stack_two_images(img1, img2, axis=0, resize=None, interpolation=None,
                      overlap=0, bg_value=None, allow_casting=True):
    """
    Returns:
        Tuple[ndarray, Tuple, Tuple]: imgB, offset_tup, sf_tup

    Ignore:
        import xinspect
        globals().update(xinspect.get_func_kwargs(_stack_two_images))
        resize = 1
        overlap = -10
    """

    def _rectify_axis(img1, img2, axis):
        """ determine if we are stacking in horzontally or vertically """
        (h1, w1) = img1.shape[0: 2]  # get chip dimensions
        (h2, w2) = img2.shape[0: 2]
        xoff2, yoff2 = 0, 0
        vert_wh  = max(w1, w2), h1 + h2
        horiz_wh = w1 + w2, max(h1, h2)
        if axis is None:
            # Display the orientation with the better (closer to 1) aspect ratio
            vert_ar  = max(vert_wh) / min(vert_wh)
            horiz_ar = max(horiz_wh) / min(horiz_wh)
            axis = 0 if vert_ar < horiz_ar else 1
        if axis == 0:  # vertical stack
            wB, hB = vert_wh
            yoff2 = h1
        elif axis == 1:
            wB, hB = horiz_wh
            xoff2 = w1
        else:
            raise ValueError('axis can only be 0 or 1')
        return axis, h1, h2, w1, w2, wB, hB, xoff2, yoff2

    def _round_dsize(dsize, scale):
        """
        Returns an integer size and scale that best approximates
        the floating point scale on the original size

        Args:
            dsize (tuple): original width height
            scale (float | tuple): desired floating point scale factor
        """
        try:
            sx, sy = scale
        except TypeError:
            sx = sy = scale
        w, h = dsize
        new_w = int(round(w * sx))
        new_h = int(round(h * sy))
        new_scale = new_w / w, new_h / h
        new_dsize = (new_w, new_h)
        return new_dsize, new_scale

    def _ramp(shape, axis):
        """ nd ramp function """
        newshape = [1] * len(shape)
        reps = list(shape)
        newshape[axis] = -1
        reps[axis] = 1
        basis = np.linspace(0, 1, shape[axis])
        data = basis.reshape(newshape)
        return np.tile(data, reps)

    def _blend(part1, part2, alpha):
        """ blending based on an alpha mask """
        part1, alpha = im_core.make_channels_comparable(part1, alpha)
        part2, alpha = im_core.make_channels_comparable(part2, alpha)
        partB = (part1 * (1.0 - alpha)) + (part2 * (alpha))
        return partB

    interpolation = im_cv2._coerce_interpolation(interpolation,
                                                 default=cv2.INTER_NEAREST)

    if allow_casting:
        if img1.dtype.kind == 'f' and img2.dtype.kind == 'u':
            img2 = im_core.ensure_float01(img2)
        elif img1.dtype.kind == 'u' and img2.dtype.kind == 'f':
            img1 = im_core.ensure_float01(img1)

        if img1.dtype != img2.dtype:
            if img1.dtype.kind == 'f' and img2.dtype.kind == 'f':
                common_dtype = np.promote_types(img1.dtype, img2.dtype)
                img1 = img1.astype(common_dtype)
                img2 = img2.astype(common_dtype)

    img1, img2 = im_core.make_channels_comparable(img1, img2)
    nChannels = im_core.num_channels(img1)

    if img1.dtype != img2.dtype:
        raise TypeError(
            'img1.dtype=%r, img2.dtype=%r' % (img1.dtype, img2.dtype))

    axis, h1, h2, w1, w2, wB, hB, xoff2, yoff2 = _rectify_axis(img1, img2, axis)

    # Rectify both images to they are the same dimension
    if resize:
        # Compre the lengths of the width and height
        length1 = img1.shape[1 - axis]
        length2 = img2.shape[1 - axis]

        if resize == 'larger':
            resize = 0 if length1 > length2 else 1
        elif resize == 'smaller':
            resize = 0 if length1 < length2 else 1

        if resize == 0:
            # Resize the first image
            sf2 = (1., 1.)
            scale = length2 / length1
            dsize, sf1 = _round_dsize(img1.shape[0:2][::-1], scale)
            img1 = cv2.resize(img1, dsize, interpolation=interpolation)
        elif resize == 1:
            # Resize the second image
            sf1 = (1., 1.)
            scale = length1 / length2
            dsize, sf2 = _round_dsize(img2.shape[0:2][::-1], scale)
            img2 = cv2.resize(img2, dsize, interpolation=interpolation)
        else:
            raise ValueError('resize can only be 0 or 1 - or a special key')
        axis, h1, h2, w1, w2, wB, hB, xoff2, yoff2 = _rectify_axis(img1, img2, axis)
    else:
        sf1 = (1.0, 1.0)
        sf2 = (1.0, 1.0)

    # allow for some overlap / blending of the images
    if overlap != 0:
        if axis == 0:
            hB -= overlap
        elif axis == 1:
            wB -= overlap

    # Do image concatentation
    if nChannels > 1 or len(img1.shape) > 2:
        newshape = (hB, wB, nChannels)
    else:
        newshape = (hB, wB)
    # Allocate new image for both
    imgB = np.zeros(newshape, dtype=img1.dtype)

    if bg_value is not None:
        if isinstance(bg_value, str):
            import kwimage
            bg_value = kwimage.Color(bg_value)._forimage(imgB)
        try:
            imgB[:, :] = bg_value
        except ValueError:
            imgB = im_core.atleast_3channels(imgB)
            imgB[:, :] = bg_value

    # Insert the images in the larger frame

    # Insert the first image
    xoff1 = yoff1 = 0
    imgB[yoff1:(yoff1 + h1), xoff1:(xoff1 + w1)] = img1

    if overlap:
        if axis == 0:
            yoff2 -= overlap
        elif axis == 1:
            xoff2 -= overlap

        imgB[yoff2:(yoff2 + h2), xoff2:(xoff2 + w2)] = img2

        if overlap > 0:
            # Blend the overlapping part
            if axis == 0:
                part1 = img1[-overlap:, :]
                part2 = imgB[yoff2:(yoff2 + overlap), 0:w1]
                alpha = _ramp(part1.shape[0:2], axis=axis)
                blended = _blend(part1, part2, alpha)
                imgB[yoff2:(yoff2 + overlap), 0:w1] = blended
            elif axis == 1:
                part1 = img1[:, -overlap:]
                part2 = imgB[0:h1, xoff2:(xoff2 + overlap)]
                alpha = _ramp(part1.shape[0:2], axis=axis)
                blended = _blend(part1, part2, alpha)
                imgB[0:h1, xoff2:(xoff2 + overlap)] = blended
    else:
        imgB[yoff2:(yoff2 + h2), xoff2:(xoff2 + w2)] = img2

    offset1 = (xoff1, yoff1)
    offset2 = (xoff2, yoff2)
    offset_tup = (offset1, offset2)
    sf_tup = (sf1, sf2)
    return imgB, offset_tup, sf_tup


def _efficient_rectangle_packing():
    """
    References:
        https://en.wikipedia.org/wiki/Packing_problems
        https://github.com/Penlect/rectangle-packer
        https://github.com/secnot/rectpack
        https://stackoverflow.com/questions/1213394/what-algorithm-can-be-used-for-packing-rectangles-of-different-sizes-into-the-sm
        https://www.codeproject.com/Articles/210979/Fast-optimizing-rectangle-packing-algorithm-for-bu

    Requires:
        pip install rectangle-packer

    Ignore:
        >>> import kwimage
        >>> anchors = anchors=[[1, 1], [3 / 4, 1], [1, 3 / 4]]
        >>> boxes = kwimage.Boxes.random(num=100, anchors=anchors).scale((100, 100)).to_xywh()
        >>> # Create a bunch of rectangles (width, height)
        >>> sizes = boxes.data[:, 2:4].astype(int).tolist()
        >>> import rpack
        >>> positions = rpack.pack(sizes)
        >>> boxes.data[:, 0:2] = positions
        >>> boxes = boxes.scale(0.95, about='center')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> boxes.draw()
        >>> # The result will be a list of (x, y) positions:
        >>> positions

        images = [kwimage.grab_test_image(key) for key in kwimage.grab_test_image.keys()]
        images = [kwimage.imresize(g, max_dim=256) for g in images]

        sizes = [g.shape[0:2][::-1] for g in images]

        import rpack
        positions = rpack.pack(sizes)

        !pip install rectpack

        import rectpack

        bin_width = 512

        packer = rectpack.newPacker(rotation=False)
        for rid, (w, h) in enumerate(sizes):
            packer.add_rect(w, h, rid=rid)

        max_w, max_h = np.array(sizes).sum(axis=0)
        # f = max_w / bin_width
        avail_height = max_h
        packer.add_bin(bin_width, avail_height)

        packer.pack()

        packer[0]

        all_rects = packer.rect_list()
        all_rects = np.array(all_rects)

        rids = all_rects[:, 5]
        tl_x = all_rects[:, 1]
        tl_y = all_rects[:, 2]
        w = all_rects[:, 3]
        h = all_rects[:, 4]

        ltrb = kwimage.Boxes(all_rects[:, 1:5], 'xywh').to_ltrb()
        canvas_w, canvas_h = ltrb.data[:, 2:4].max(axis=0)

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for b, x, y, w, h, rid in all_rects:
            img = images[rid]
            img = kwimage.ensure_float01(img)
            canvas, img = kwimage.make_channels_comparable(canvas, img)
            canvas[y: y + h, x: x + w] = img

        kwplot.imshow(canvas)

    """
