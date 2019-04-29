# -*- coding: utf-8 -*-
"""
TODO:
    - [ ] Remove doctest dependency on ndsampler?

CommandLine:
    xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py __doc__

Example:
    >>> # xdoctest: +REQUIRES(module:ndsampler)
    >>> from kwimage.structs.heatmap import *  # NOQA
    >>> from kwimage.structs.heatmap import _dets_to_fcmaps
    >>> import kwimage
    >>> import ndsampler
    >>> sampler = ndsampler.CocoSampler.demo('shapes')
    >>> iminfo, anns = sampler.load_image_with_annots(1)
    >>> image = iminfo['imdata']
    >>> input_dims = image.shape[0:2]
    >>> kp_classes = sampler.dset.keypoint_categories()
    >>> dets = kwimage.Detections.from_coco_annots(
    >>>     anns, sampler.dset.dataset['categories'],
    >>>     sampler.catgraph, kp_classes, shape=input_dims)
    >>> bg_size = [100, 100]
    >>> heatmap = dets.rasterize(bg_size, input_dims, soften=2)
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.figure(fnum=1, doclf=True)
    >>> kwplot.imshow(image)
    >>> heatmap.draw(invert=True, kpts=[0, 1, 2, 3, 4])

Example:
    >>> # xdoctest: +REQUIRES(module:ndsampler)
    >>> from kwimage.structs.heatmap import *  # NOQA
    >>> from kwimage.structs.heatmap import _dets_to_fcmaps
    >>> import kwimage
    >>> import ndsampler
    >>> sampler = ndsampler.CocoSampler.demo('shapes')
    >>> iminfo, anns = sampler.load_image_with_annots(1)
    >>> image = iminfo['imdata']
    >>> input_dims = image.shape[0:2]
    >>> kp_classes = sampler.dset.keypoint_categories()
    >>> dets = kwimage.Detections.from_coco_annots(
    >>>     anns, sampler.dset.dataset['categories'],
    >>>     sampler.catgraph, kp_classes, shape=input_dims)
    >>> bg_size = [100, 100]
    >>> bg_idxs = sampler.catgraph.index('background')
    >>> fcn_target = _dets_to_fcmaps(dets, bg_size, input_dims, bg_idxs)
    >>> fcn_target.keys()
    >>> print('fcn_target: ' + ub.repr2(ub.map_vals(lambda x: x.shape, fcn_target), nl=1))
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> size_mask = fcn_target['size']
    >>> dxdy_mask = fcn_target['dxdy']
    >>> cidx_mask = fcn_target['cidx']
    >>> kpts_mask = fcn_target['kpts']
    >>> def _vizmask(dxdy_mask):
    >>>     dx, dy = dxdy_mask
    >>>     mag = np.sqrt(dx ** 2 + dy ** 2)
    >>>     mag /= (mag.max() + 1e-9)
    >>>     mask = (cidx_mask != 0).astype(np.float32)
    >>>     angle = np.arctan2(dy, dx)
    >>>     orimask = kwplot.make_orimask(angle, mask, alpha=mag)
    >>>     vecmask = kwplot.make_vector_field(
    >>>         dx, dy, stride=4, scale=0.1, thickness=1, tipLength=.2,
    >>>         line_type=16)
    >>>     return [vecmask, orimask]
    >>> vecmask, orimask = _vizmask(dxdy_mask)
    >>> raster = kwimage.overlay_alpha_layers(
    >>>     [vecmask, orimask, image], keepalpha=False)
    >>> raster = dets.draw_on((raster * 255).astype(np.uint8),
    >>>                       labels=True, alpha=None)
    >>> kwplot.imshow(raster)
    >>> kwplot.show_if_requested()
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import numpy as np
import torch
import ubelt as ub
import skimage
import kwarray
import six
import functools
import xdev
from kwimage.structs.detections import _dets_to_fcmaps
from . import _generic


class _HeatmapDrawMixin(object):
    """
    mixin methods for drawing heatmap details
    """

    def colorize(self, channel, invert=False, with_alpha=1.0,
                 interpolation='linear', imgspace=False):
        """
        Creates a colorized version of a heatmap channel suitable for
        visualization

        Args:
            channel (int): category to visualize
            imgspace (bool, default=False): colorize the image after
                warping into the image space.

        CommandLine:
            xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py _HeatmapDrawMixin.colorize --show

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> colormask1 = self.colorize(0, imgspace=False)
            >>> colormask2 = self.colorize(0, imgspace=True)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> kwplot.show_if_requested()

        Example:
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> colormask1 = self.colorize('diameter', imgspace=False)
            >>> colormask2 = self.colorize('diameter', imgspace=True)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> kwplot.show_if_requested()
        """
        import kwplot
        if isinstance(channel, six.string_types):
            # TODO: this is a bit hacky / inefficient, probably needs minor cleanup
            if imgspace:
                a = self.warp()
            else:
                a = self
            if channel == 'offset':
                mask = np.linalg.norm(a.offset, axis=0)
            elif channel == 'diameter':
                mask = np.linalg.norm(a.diameter, axis=0)
            else:
                raise KeyError(channel)
            mask = mask / np.maximum(mask.max(), 1e-9)
        else:
            if imgspace:
                mask = self.upscale(channel, interpolation=interpolation)[0]
            else:
                mask = self.class_probs[channel]

            if invert:
                mask = 1 - mask
        colormask = kwplot.make_heatmask(mask, with_alpha=with_alpha)
        return colormask

    def draw_stacked(self, image=None, dsize=(224, 224), ignore_class_idxs={},
                     top=None, chosen_cxs=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> stacked = self.draw_stacked()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked)
        """
        import kwimage
        import matplotlib as mpl
        tf = self.tf_data_to_img
        mat = np.linalg.inv(tf.params)
        cmap = mpl.cm.get_cmap('magma')

        level_dsize = self.class_probs.shape[-2:][::-1]

        if chosen_cxs is None:
            if top is not None:
                # Find the categories with the most "heat"
                cx_to_score = self.class_probs.mean(2).mean(1)
                for cx in ignore_class_idxs:
                    cx_to_score[cx] = -np.inf
                chosen_cxs = kwarray.ArrayAPI.numpy(cx_to_score).argsort()[::-1][:top]
            else:
                chosen_cxs = np.arange(self.class_probs.shape[0])

        if image is not None:
            small_img = cv2.warpAffine(image, mat[0:2], dsize=level_dsize)
            small_img = cv2.resize(small_img, dsize)
            # small_img = cv2.resize(image, dsize)
            colorized = [small_img]
        else:
            colorized = []
        for cx in chosen_cxs:
            if cx in ignore_class_idxs:
                continue
            if self.classes:
                node = self.classes[cx]
            else:
                node = 'cx={}'.format(cx)
            c = self.class_probs[cx]
            c = cmap(c)
            c = (c[..., 0:3] * 255.0).astype(np.uint8)
            c = cv2.resize(c, dsize)
            c = kwimage.draw_text_on_image(c, '{}'.format(node), (0, 20), fontScale=.5)
            # kwplot.imshow(c, title=str(i), fnum=2)
            colorized.append(c)
        stacked = kwimage.stack_images(colorized, overlap=-3, axis=1)
        return stacked

    def draw(self, image=None, **kwargs):
        # If draw doesnt exist use draw_on
        import numpy as np
        if image is None:
            dims = self.bounds
            shape = tuple(dims) + (4,)
            image = np.zeros(shape, dtype=np.float32)
        image = self.draw_on(image, **kwargs)
        import kwplot
        kwplot.imshow(image)

    def draw_on(self, image, channel=0, invert=False, with_alpha=1.0,
                interpolation='linear', vecs=False, kpts=None, imgspace=None):
        """
        Overlays a heatmap channel on top of an image

        Args:
            image (ndarray): image to draw on
            channel (int): category to visualize
            imgspace (bool, default=False): colorize the image after
                warping into the image space.

        TODO:
            - [ ] Find a way to visualize offset, diameter, and class_probs
                  either individually or all at the same time

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwarray
            >>> import kwimage
            >>> image = kwimage.grab_test_image('astro')
            >>> probs = kwimage.gaussian_patch(image.shape[0:2])[None, :]
            >>> probs = probs / probs.max()
            >>> class_probs = kwarray.ArrayAPI.cat([probs, 1 - probs], axis=0)
            >>> self = kwimage.Heatmap(class_probs=class_probs, offset=5 * np.random.randn(2, *probs.shape[1:]))
            >>> toshow = self.draw_on(image, 0, vecs=True, with_alpha=0.85)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(toshow)

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwimage
            >>> self = kwimage.Heatmap.random(dims=(200, 200), dets='coco', keypoints=True)
            >>> image = kwimage.grab_test_image('astro')
            >>> toshow = self.draw_on(image, 0, vecs=False, with_alpha=0.85)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(toshow)

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwimage
            >>> self = kwimage.Heatmap.random(dims=(200, 200), dets='coco', keypoints=True)
            >>> kpts = [6]
            >>> self = self.warp(self.tf_data_to_img.params)
            >>> image = kwimage.grab_test_image('astro')
            >>> image = kwimage.ensure_alpha_channel(image)
            >>> toshow = self.draw_on(image, 0, with_alpha=0.85, kpts=kpts)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(toshow)
        """
        import kwimage
        import kwplot

        if imgspace is None:
            if np.all(image.shape[0:2] == np.array(self.img_dims)):
                imgspace = True

        colormask = self.colorize(channel, invert=invert,
                                  with_alpha=with_alpha,
                                  interpolation=interpolation,
                                  imgspace=imgspace)

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        image = kwimage.ensure_float01(image)
        layers = []

        vec_colors = kwplot.Color.distinct(2)
        vec_alpha = .5

        if kpts is not None:
            # TODO: make a nicer keypoint offset vector visuliazation
            if kpts is True:
                if self.data.get('keypoints', None) is not None:
                    keypoints = self.data['keypoints']
                    kpts = list(range(len(keypoints.shape[1])))
            if not ub.iterable(kpts):
                kpts = [kpts]
            print('kpts = {!r}'.format(kpts))
            E = int(bool(vecs))
            vec_colors = kwplot.Color.distinct(len(kpts) + E)
            print('vec_colors = {!r}'.format(vec_colors))

        if vecs:
            if self.data.get('offset', None) is not None:
                #Hack
                # Visualize center offset vectors
                dy, dx = kwarray.ArrayAPI.numpy(self.data['offset'])
                color = vec_colors[0]
                vecmask = kwplot.make_vector_field(dx, dy, stride=4, scale=1.0,
                                                   alpha=with_alpha * vec_alpha,
                                                   color=color)
                vec_alpha = max(.1, vec_alpha - .1)
                chw = torch.Tensor(vecmask.transpose(2, 0, 1))
                vecalign = self._warp_imgspace(chw, interpolation=interpolation)
                vecalign = vecalign.transpose(1, 2, 0)
                # print('vecalign = {!r}'.format(vecalign))
                layers.append(vecalign)

        if kpts is not None:
            # TODO: make a nicer keypoint offset vector visuliazation
            if self.data.get('keypoints', None) is not None:
                keypoints = self.data['keypoints']
                for i, k in enumerate(kpts):
                    # color = (np.array(vec_colors[k]) * 255).astype(np.uint8)
                    color = vec_colors[i + E]

                    dy, dx = kwarray.ArrayAPI.numpy(keypoints[:, k])
                    vecmask = kwplot.make_vector_field(dx, dy, stride=8,
                                                       scale=0.5,
                                                       alpha=with_alpha *
                                                       vec_alpha, color=color)
                    vec_alpha = max(.1, vec_alpha - .1)
                    chw = torch.Tensor(vecmask.transpose(2, 0, 1))
                    vecalign = self._warp_imgspace(chw, interpolation=interpolation)
                    vecalign = vecalign.transpose(1, 2, 0)
                    print('vecalign = {!r}'.format(vecalign))
                    layers.append(vecalign)

        layers.append(colormask)
        layers.append(image)

        overlaid = kwimage.overlay_alpha_layers(layers)
        overlaid = dtype_fixer(overlaid)
        return overlaid


class _HeatmapWarpMixin(object):
    """
    mixin method having to do with warping and aligning heatmaps
    """

    def _align_other(self, other):
        """
        Warp another Heatmap (with the same underlying imgdims) into the same
        space as this heatmap. This lets us perform elementwise operations on
        the two heatmaps (like geometric mean).

        Args:
            other (Heatmap): the heatmap to align with `self`

        Returns:
            Heatmap: warped version of `other` that aligns with `self`.

        Example:
            >>> self = Heatmap.random((120, 130), img_dims=(200, 210), classes=2, nblips=10, rng=0)
            >>> other = Heatmap.random((60, 70), img_dims=(200, 210), classes=2, nblips=10, rng=1)
            >>> other2 = self._align_other(other)
            >>> assert self.shape != other.shape
            >>> assert self.shape == other2.shape
            >>> # xdoctest: +REQUIRES(--show)
            >>> kwplot.autompl()
            >>> kwplot.imshow(self.colorize(0, imgspace=False), fnum=1, pnum=(3, 2, 1))
            >>> kwplot.imshow(self.colorize(1, imgspace=False), fnum=1, pnum=(3, 2, 2))
            >>> kwplot.imshow(other.colorize(0, imgspace=False), fnum=1, pnum=(3, 2, 3))
            >>> kwplot.imshow(other.colorize(1, imgspace=False), fnum=1, pnum=(3, 2, 4))
        """
        if self is other:
            return other
        # The heatmaps must belong to the same image space
        assert self.classes == other.classes
        assert np.all(self.img_dims == other.img_dims)

        img_to_self = np.linalg.inv(self.tf_data_to_img.params)
        other_to_img = other.tf_data_to_img.params
        other_to_self = np.matmul(img_to_self, other_to_img)

        mat = other_to_self
        output_dims = self.class_probs.shape[1:]

        # other now exists in the same space as self
        new_other = other.warp(mat, output_dims=output_dims)
        return new_other

    def _align(self, mask, interpolation='linear'):
        """
        Align a linear combination of heatmap channels with the original image

        DEPRICATE
        """
        import kwimage
        M = self.tf_data_to_img.params[0:3]
        dsize = tuple(map(int, self.img_dims[::-1]))
        # flags = kwimage.im_cv2._rectify_interpolation('lanczos')
        # flags = kwimage.im_cv2._rectify_interpolation('nearest')
        flags = kwimage.im_cv2._rectify_interpolation(interpolation)
        aligned = cv2.warpAffine(mask, M[0:2], dsize=tuple(dsize), flags=flags)
        aligned = np.clip(aligned, 0, 1)
        return aligned

    def _warp_imgspace(self, chw, interpolation='linear'):
        import kwimage
        if self.tf_data_to_img is None and self.img_dims is None:
            aligned = chw.cpu().numpy()
        else:
            output_dims = self.img_dims
            mat = torch.Tensor(self.tf_data_to_img.params[0:3])
            outputs = kwimage.warp_tensor(
                chw[None, :], mat, output_dims=output_dims, mode=interpolation
            )
            aligned = outputs[0].cpu().numpy()
        return aligned

    def upscale(self, channel=None, interpolation='linear'):
        """
        Warp the heatmap with the image dimensions

        Example:
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> colormask = self.upscale()

        """
        if channel is None:
            chw = torch.Tensor(self.class_probs)
        else:
            chw = torch.Tensor(self.class_probs[channel])[None, :]
        aligned = self._warp_imgspace(chw, interpolation=interpolation)
        return aligned

    @xdev.profile
    def warp(self, mat=None, input_dims=None, output_dims=None,
             interpolation='linear', modify_spatial_coords=True,
             mat_is_xy=True):
        """
        Warp all spatial maps. If the map contains spatial data, that data is
        also warped (ignoring the translation component).

        Args:
            mat (ArrayLike): transformation matrix
            input_dims (tuple): unused, only exists for compatibility
            output_dims (tuple): size of the output heatmap
            interpolation (str): see `kwimage.warp_tensor`
            mat_is_xy (bool, default=True): set to false if the matrix
                is in yx space instead of xy space

        Returns:
            Heatmap: this heatmap warped into a new spatial dimension

        Ignore:
            # Verify swapping rows 0 and 1 and then swapping columns 0 and 1
            # Produces a matrix that works with permuted coordinates
            # It does.
            import sympy
            a, b, c, d, e, f, g, h, i, x, y, z = sympy.symbols('a, b, c, d, e, f, g, h, i, x, y, z')
            M1 = sympy.Matrix([[a, b, c], [d, e, f], [g, h, i]])
            M2 = sympy.Matrix([[e, d, f], [b, a, c], [h, g, i]])
            xy = sympy.Matrix([[x], [y], [z]])
            yx = sympy.Matrix([[y], [x], [z]])

            R1 = M1.multiply(xy)
            R2 = M2.multiply(yx)
            R3 = sympy.Matrix([[R1[1]], [R1[0]], [R1[2]],])
            assert R2 == R3

        Example:
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> self = Heatmap.random(rng=0, keypoints=True)
            >>> S = 3.0
            >>> mat = np.eye(3) * S
            >>> mat[-1, -1] = 1
            >>> newself = self.warp(mat, np.array(self.dims) * S).numpy()
            >>> assert newself.offset.shape[0] == 2
            >>> assert newself.diameter.shape[0] == 2
            >>> f1 = newself.offset.max() / self.offset.max()
            >>> assert f1 == S
            >>> f2 = newself.diameter.max() / self.diameter.max()
            >>> assert f2 == S

        Example:
            >>> import kwimage
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = kwimage.Heatmap.random(dims=(100, 100), dets='coco', keypoints=True)
            >>> image = np.zeros(self.img_dims)
            >>> toshow = self.draw_on(image, 1, vecs=True, with_alpha=0.85)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(toshow)
        """
        import kwimage

        if mat is None:
            mat = self.tf_data_to_img.params

        if output_dims is None:
            output_dims = self.img_dims

        newdata = {}
        newmeta = self.meta.copy()

        impl = kwarray.ArrayAPI.coerce('tensor')

        if mat_is_xy:
            # If the matrix is in X/Y coords, modify it to be in Y/X coords
            mat = mat[[1, 0, 2], :][:, [1, 0, 2]]

        mat = impl.asarray(mat)

        tf = skimage.transform.AffineTransform(matrix=mat)
        # hack: need to get a version of the matrix without any translation
        tf_notrans = _remove_translation(tf)
        mat_notrans = torch.Tensor(tf_notrans.params)

        # Modify data_to_img so the new heatmap will also properly upscale to
        # the image coordinates.
        inv_tf = skimage.transform.AffineTransform(matrix=tf._inv_matrix)
        newmeta['tf_data_to_img'] = self.tf_data_to_img + inv_tf

        for k, v in self.data.items():
            if v is not None:
                v = kwarray.ArrayAPI.tensor(v)
                # For spatial keys we need to transform the underlying values as well
                if modify_spatial_coords:
                    if k in self.__spatialkeys__:
                        pts = impl.contiguous(impl.T(v))
                        pts = kwimage.warp_points(mat_notrans, pts)
                        v = impl.contiguous(impl.T(pts))

                new_v = kwimage.warp_tensor(
                    v[None, :].float(), mat, output_dims=output_dims,
                    mode=interpolation)[0]
                newdata[k] = impl.asarray(new_v)

        newself = self.__class__(newdata, newmeta)
        return newself

    def scale(self, factor, output_dims=None, interpolation='linear'):
        if not ub.iterable(factor):
            s1 = s2 = factor
        else:
            s1, s2 = factor
        mat = np.array([
            [s1,  0, 0],
            [ 0, s2, 0],
            [ 0,  0, 1.],
        ])
        return self.warp(mat, output_dims=output_dims,
                         interpolation=interpolation)

    def translate(self, offset, output_dims=None, interpolation='linear'):
        if not ub.iterable(offset):
            tx = ty = offset
        else:
            tx, ty = offset
        mat = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1.],
        ])
        return self.warp(mat, output_dims=output_dims,
                         interpolation=interpolation)


class _HeatmapAlgoMixin(object):
    """
    Algorithmic operations on heatmaps
    """

    @classmethod
    def combine(cls, heatmaps, root_index=None, dtype=np.float32):
        """
        Combine multiple heatmaps into a single heatmap.

        Args:
            heatmaps (Sequence[Heatmap]): multiple heatmaps to combine into one
            root_index (int): which heatmap in the sequence to align other
                heatmaps with

        Returns:
            Heatmap: the combined heatmap

        Example:
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> a = Heatmap.random((120, 130), img_dims=(200, 210), classes=2, nblips=10, rng=0)
            >>> b = Heatmap.random((60, 70), img_dims=(200, 210), classes=2, nblips=10, rng=1)
            >>> c = Heatmap.random((40, 30), img_dims=(200, 210), classes=2, nblips=10, rng=1)
            >>> heatmaps = [a, b, c]
            >>> newself = Heatmap.combine(heatmaps, root_index=2)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(a.colorize(0, imgspace=1), fnum=1, pnum=(4, 2, 1))
            >>> kwplot.imshow(a.colorize(1, imgspace=1), fnum=1, pnum=(4, 2, 2))
            >>> kwplot.imshow(b.colorize(0, imgspace=1), fnum=1, pnum=(4, 2, 3))
            >>> kwplot.imshow(b.colorize(1, imgspace=1), fnum=1, pnum=(4, 2, 4))
            >>> kwplot.imshow(c.colorize(0, imgspace=1), fnum=1, pnum=(4, 2, 5))
            >>> kwplot.imshow(c.colorize(1, imgspace=1), fnum=1, pnum=(4, 2, 6))
            >>> kwplot.imshow(newself.colorize(0, imgspace=1), fnum=1, pnum=(4, 2, 7))
            >>> kwplot.imshow(newself.colorize(1, imgspace=1), fnum=1, pnum=(4, 2, 8))
            >>> # xdoctest: +REQUIRES(--show)
            >>> kwplot.imshow(a.colorize('offset', imgspace=1), fnum=2, pnum=(4, 1, 1))
            >>> kwplot.imshow(b.colorize('offset', imgspace=1), fnum=2, pnum=(4, 1, 2))
            >>> kwplot.imshow(c.colorize('offset', imgspace=1), fnum=2, pnum=(4, 1, 3))
            >>> kwplot.imshow(newself.colorize('offset', imgspace=1), fnum=2, pnum=(4, 1, 4))
            >>> # xdoctest: +REQUIRES(--show)
            >>> kwplot.imshow(a.colorize('diameter', imgspace=1), fnum=3, pnum=(4, 1, 1))
            >>> kwplot.imshow(b.colorize('diameter', imgspace=1), fnum=3, pnum=(4, 1, 2))
            >>> kwplot.imshow(c.colorize('diameter', imgspace=1), fnum=3, pnum=(4, 1, 3))
            >>> kwplot.imshow(newself.colorize('diameter', imgspace=1), fnum=3, pnum=(4, 1, 4))
        """
        # define arithmetic and geometric mean
        amean = functools.partial(np.mean, axis=0)

        # If the root is not specified use the largest heatmap
        if root_index is None:
            root_index = ub.argmax([np.prod(h.shape) for h in heatmaps])
        root = heatmaps[root_index]
        aligned_heatmaps = [root._align_other(h).numpy() for h in heatmaps]
        aligned_root = aligned_heatmaps[root_index]

        # Use the appropriate mean for each type of data
        newdata = {}
        if 'class_probs' in aligned_root.data:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'divide by zero')
                tmp = np.array([h.class_probs.astype(dtype) for h in aligned_heatmaps], dtype=dtype)
                newdata['class_probs'] = _gmean(tmp, clobber=True)
                tmp = None
        if 'offset' in aligned_root.data:
            newdata['offset'] = amean([h.offset for h in aligned_heatmaps])
        if 'diameter' in aligned_root.data:
            newdata['diameter'] = amean([h.diameter for h in aligned_heatmaps])
        if 'keypoints' in aligned_root.data and aligned_root.data['keypoints'] is not None:
            newdata['keypoints'] = amean([h.data['keypoints'] for h in aligned_heatmaps])
        newself = aligned_root.__class__(newdata, aligned_root.meta)
        return newself

    @xdev.profile
    def detect(self, channel, invert=False, min_score=0.01, num_min=10,
               max_dims=None, min_dims=None):
        """
        Lossy conversion from a Heatmap to a Detections object.

        For efficiency, the detections are returned in the same space as the
        heatmap, which usually some downsampled version of the image space.
        This is because it is more efficient to transform the detections into
        image-space after non-max supression is applied.

        Args:
            channel (int | ArrayLike[*DIMS]): class index to detect objects in.
                Alternatively, channel can be a custom probability map as long
                as its dimension agree with the heatmap.

            invert (bool, default=False): if True, inverts the probabilities in
                the chosen channel. (Useful if you have a background channel
                but want to detect foreground objects).

            min_score (float, default=0.1): probability threshold required
                for a pixel to be converted into a detection.

            num_min (int, default=10):
                always return at least `nmin` of the highest scoring detections
                even if they aren't above the `min_score` threshold.

            max_dims (Tuple[int, int]): maximum height / width of detections

            min_dims (Tuple[int, int]): minimum height / width of detections

        Returns:
            kwimage.Detections: raw detections.

                Note that these detections will not have class_idx populated

                It is the users responsbility to run non-max suppression on
                these results to remove duplicate detections.

        SeeAlso:
            Detections.rasterize

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> import ndsampler
            >>> self = Heatmap.random(rng=2, dims=(32, 32), diameter=10, offset=0)
            >>> dets = self.detect(channel=0)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dets1 = dets.sort().take(range(10))
            >>> colormask1 = self.colorize(0, imgspace=False)
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> dets1.draw()
            >>> # Transform heatmap and detections into image space.
            >>> colormask2 = self.colorize(0, imgspace=True)
            >>> dets2 = dets1.warp(self.tf_data_to_img)
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> dets2.draw()

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> import ndsampler
            >>> catgraph = ndsampler.CategoryTree.demo()
            >>> class_energy = torch.rand(len(catgraph), 32, 32)
            >>> class_probs = catgraph.heirarchical_softmax(class_energy, dim=0)
            >>> self = Heatmap.random(rng=0, dims=(32, 32), classes=catgraph, keypoints=True)
            >>> self.data['class_probs'] = class_probs.numpy()
            >>> channel = catgraph.index('background')
            >>> dets = self.detect(channel, invert=True)
            >>> class_idxs, scores = catgraph.decision(dets.probs, dim=1)
            >>> dets.data['class_idxs'] = class_idxs
            >>> dets.data['scores'] = scores
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dets1 = dets.sort().take(range(10))
            >>> colormask1 = self.colorize(0, imgspace=False)
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> dets1.draw(radius=1.0)
            >>> # Transform heatmap and detections into image space.
            >>> colormask2 = self.colorize(0, imgspace=True)
            >>> dets2 = dets1.warp(self.tf_data_to_img)
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> dets2.draw(radius=1.0)
        """
        if isinstance(channel, int):
            probs = self.class_probs[channel]
        else:
            probs = channel
        if invert:
            probs = 1 - probs

        dets = _prob_to_dets(
            probs, diameter=self.diameter, offset=self.offset,
            class_probs=self.class_probs,
            keypoints=self.data.get('keypoints', None),
            min_score=min_score, num_min=num_min,
            max_dims=max_dims, min_dims=min_dims,
        )
        if dets.data.get('keypoints', None) is not None:
            kp_classes = self.meta['kp_classes']
            dets.data['keypoints'].meta['classes'] = kp_classes
            dets.meta['kp_classes'] = kp_classes

        dets.meta['classes'] = self.classes
        return dets


class Heatmap(_generic.Spatial, _HeatmapDrawMixin,
              _HeatmapWarpMixin, _HeatmapAlgoMixin):
    """
    Keeps track of a downscaled heatmap and how to transform it to overlay the
    original input image. Heatmaps generally are used to estimate class
    probabilites at each pixel. This data struction additionally contains logic
    to augment pixel with offset (dydx) and scale (diamter) information.

    Attributes:
        data (Dict[str, ArrayLike]): dictionary containing spatially aligned
            heatmap data. Valid keys are as follows.

            class_probs (ArrayLike[C, H, W] | ArrayLike[C, D, H, W]):
                A probability map for each class. C is the number of classes.

            offset (ArrayLike[2, H, W] | ArrayLike[3, D, H, W], optional):
                object center position offset in y,x / t,y,x coordinates

            diamter (ArrayLike[2, H, W] | ArrayLike[3, D, H, W], optional):
                object bounding box sizes in h,w / d,h,w coordinates

            keypoints (ArrayLike[2, K, H, W] | ArrayLike[3, K, D, H, W], optional):
                y/x offsets for K different keypoint classes

        data (Dict[str, object]): dictionary containing miscellanious metadata
            about the heatmap data. Valid keys are as follows.

            img_dims (Tuple[H, W] | Tuple[D, H, W]):
                original image dimension

            tf_data_to_image (skimage.transform._geometric.GeometricTransform):
                transformation matrix (typically similarity or affine) that
                projects the given heatmap onto the image dimensions such that
                the image and heatmap are spatially aligned.

            classes (List[str] | ndsampler.CategoryTree):
                information about which index in `data['class_probs']`
                corresponds to which semantic class.

        **kwargs: any key that is accepted by the `data` or `meta` dictionaries
            can be specified as a keyword argument to this class and it will
            be properly placed in the appropriate internal dictionary.

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py Heatmap --show

    Example:
        >>> import kwimage
        >>> class_probs = kwimage.grab_test_image(dsize=(32, 32), space='gray')[None, ] / 255.0
        >>> img_dims = (220, 220)
        >>> tf_data_to_img = skimage.transform.AffineTransform(translation=(-18, -18), scale=(8, 8))
        >>> self = Heatmap(class_probs=class_probs, img_dims=img_dims,
        >>>                tf_data_to_img=tf_data_to_img)
        >>> aligned = self.upscale()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(aligned[0])
        >>> kwplot.show_if_requested()
    """
    # Valid keys for the data dictionary
    __datakeys__ = ['class_probs', 'offset', 'diameter', 'keypoints']

    # Valid keys for the meta dictionary
    __metakeys__ = ['img_dims', 'tf_data_to_img', 'classes', 'kp_classes']

    __spatialkeys__ = ['offset', 'diameter', 'keypoints']

    def __init__(self, data=None, meta=None, **kwargs):
        # Standardize input format
        if kwargs:
            if data or meta:
                raise ValueError('Cannot specify kwargs AND data/meta dicts')
            _datakeys = self.__datakeys__
            _metakeys = self.__metakeys__
            # Allow the user to specify custom data and meta keys
            if 'datakeys' in kwargs:
                _datakeys = _datakeys + list(kwargs.pop('datakeys'))
            if 'metakeys' in kwargs:
                _metakeys = _metakeys + list(kwargs.pop('metakeys'))
            # Perform input checks whenever kwargs is given
            data = {key: kwargs.pop(key) for key in _datakeys if key in kwargs}
            meta = {key: kwargs.pop(key) for key in _metakeys if key in kwargs}
            if kwargs:
                raise ValueError(
                    'Unknown kwargs: {}'.format(sorted(kwargs.keys())))
        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}

        self.data = data
        self.meta = meta

    def __nice__(self):
        shape = None if self.class_probs is None else self.class_probs.shape
        return '{} on img_dims={}'.format(shape, self.img_dims)

    def __getitem__(self, index):
        return self.class_probs[index]

    def __len__(self):
        return len(self.class_probs)

    @property
    def shape(self):
        return self.class_probs.shape

    @property
    def bounds(self):
        return self.class_probs.shape[1:]

    @property
    def dims(self):
        """ space-time dimensions of this heatmap """
        return self.class_probs.shape[1:]

    def is_numpy(self):
        return self._impl.is_numpy

    def is_tensor(self):
        return self._impl.is_tensor

    @property
    def _impl(self):
        """
        Returns the internal tensor/numpy ArrayAPI implementation
        """
        return kwarray.ArrayAPI.coerce(self.data['class_probs'])

    # @property
    # def device(self):
    #     """ If the backend is torch returns the data device, otherwise None """
    #     return self.data['class_probs'].device

    @classmethod
    def random(cls, dims=(10, 10), classes=3, diameter=True, offset=True,
               keypoints=False, img_dims=None, dets=None, nblips=10, noise=0.0,
               rng=None):
        """
        Creates dummy data, suitable for use in tests and benchmarks

        Args:
            dims (Tuple): dimensions of the heatmap
            img_dims (Tuple): dimensions of the image the heatmap corresponds to

        Example:
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> self = Heatmap.random((128, 128), img_dims=(200, 200),
            >>>     classes=3, nblips=10, rng=0, noise=0.1)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(self.colorize(0, imgspace=0), fnum=1, pnum=(1, 4, 1), doclf=1)
            >>> kwplot.imshow(self.colorize(1, imgspace=0), fnum=1, pnum=(1, 4, 2))
            >>> kwplot.imshow(self.colorize(2, imgspace=0), fnum=1, pnum=(1, 4, 3))
            >>> kwplot.imshow(self.colorize(3, imgspace=0), fnum=1, pnum=(1, 4, 4))

        Ignore:
            self.detect(0).sort().non_max_supress()[-np.arange(1, 4)].draw()
            from kwimage.structs.heatmap import *  # NOQA
            import xdev
            globals().update(xdev.get_func_kwargs(Heatmap.random))

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwimage
            >>> self = kwimage.Heatmap.random(dims=(50, 200), dets='coco',
            >>>                               keypoints=True)
            >>> image = np.zeros(self.img_dims)
            >>> toshow = self.draw_on(image, 1, vecs=True, kpts=0, with_alpha=0.85)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(toshow)

        Ignore:
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(image)
            >>> dets.draw()
            >>> dets.data['keypoints'].draw(radius=6)
            >>> dets.data['segmentations'].draw()

            >>> self.draw()
        """
        import kwimage
        rng = kwarray.ensure_rng(rng)

        if dets == 'coco':
            # special detections from the ndsampler coco demo
            import ndsampler
            sampler = ndsampler.CocoSampler.demo('photos')
            iminfo, anns = sampler.load_image_with_annots(1)
            image = iminfo['imdata']
            input_dims = image.shape[:2]
            kp_classes = sampler.dset.keypoint_categories()
            dets = kwimage.Detections.from_coco_annots(
                anns, cats=sampler.dset.dataset['categories'], shape=input_dims,
                kp_classes=kp_classes)
            img_dims = input_dims

        if isinstance(classes, int):
            classes = ['class_{}'.format(c) for c in range(classes)]

        # Pretend this heatmap corresponds to some upscaled subregion
        if img_dims is None:
            scale = 1 + rng.rand(2) * 2
            translation = rng.rand(2) * np.array(dims[::-1]) / 2
            tf_data_to_img = skimage.transform.AffineTransform(
                scale=scale, translation=translation)

            wh_dims = dims[::-1]
            img_wh_dims = tuple(np.ceil(tf_data_to_img([wh_dims]))[0].astype(np.int).tolist())
            img_dims = img_wh_dims[::-1]
        else:
            img_dims = np.array(img_dims)
            tf_data_to_img = skimage.transform.AffineTransform(
                scale=(img_dims / dims)[::-1],
                translation=(0, 0),
            )

        # TODO: clean up method of making heatmap from detections
        if dets is None:
            # We are either given detections, or we make random ones
            dets = kwimage.Detections.random(num=nblips, scale=img_dims,
                                             keypoints=keypoints,
                                             classes=classes, rng=rng)
            if 'background' not in dets.classes:
                dets.classes.append('background')

            classes = dets.classes
        else:
            classes = dets.classes
        # assume we have background
        bg_idx = dets.classes.index('background')

        # Warp detections into heatmap space
        transform = np.linalg.inv(tf_data_to_img.params)
        warped_dets = dets.warp(transform, input_dims=img_dims,
                                output_dims=dims)

        tf_notrans = _remove_translation(tf_data_to_img)
        bg_size = tf_notrans.inverse([100, 100])[0]

        _target = _dets_to_fcmaps(warped_dets, bg_size, dims,
                                  bg_idx=bg_idx, soft=True)
        class_probs = _target['class_probs']
        noise = (rng.randn(*class_probs.shape) * noise)
        class_probs += noise
        np.clip(class_probs, 0, None, out=class_probs)
        # class_probs = class_probs / class_probs.sum(axis=0)
        class_probs = np.array([smooth_prob(p) for p in class_probs])
        class_probs = class_probs / np.maximum(class_probs.sum(axis=0), 1e-9)

        if offset is True:
            offset = _target['dxdy'][[1, 0]]

        if keypoints is True:
            if 'kp_classes' not in locals():
                kp_classes = list(range(_target['kpts'].shape[1]))  # HACK

            keypoints = _target['kpts'][[1, 0]]

        if diameter is True:
            diameter = _target['size'][[1, 0]]

        self = cls(class_probs=class_probs, offset=offset,
                   diameter=diameter, img_dims=img_dims, classes=classes,
                   tf_data_to_img=tf_data_to_img)

        if keypoints is not False and keypoints is not None:
            self.data['keypoints'] = keypoints
            self.meta['kp_classes'] = kp_classes

        return self

    # --- Data Properties ---

    @property
    def class_probs(self):
        return self.data['class_probs']

    @property
    def offset(self):
        return self.data['offset']

    @property
    def diameter(self):
        return self.data['diameter']

    # --- Meta Properties ---

    @property
    def img_dims(self):
        return self.meta.get('img_dims', None)

    @property
    def tf_data_to_img(self):
        return self.meta.get('tf_data_to_img', None)

    @property
    def classes(self):
        return self.meta.get('classes', None)

    # ---

    @xdev.profile
    def numpy(self):
        """
        Converts underlying data to numpy arrays
        """
        newdata = {}
        for key, val in self.data.items():
            if val is None:
                newval = val
            else:
                newval = kwarray.ArrayAPI.numpy(val)
            newdata[key] = newval
        newself = self.__class__(newdata, self.meta)
        return newself

    @xdev.profile
    def tensor(self, device=ub.NoParam):
        """
        Converts underlying data to torch tensors
        """
        newdata = {}
        for key, val in self.data.items():
            if val is None:
                newval = val
            else:
                newval = kwarray.ArrayAPI.tensor(val, device=device)
            newdata[key] = newval
        newself = self.__class__(newdata, self.meta)
        return newself


@xdev.profile
def _prob_to_dets(probs, diameter=None, offset=None, class_probs=None,
                  keypoints=None, min_score=0.01, num_min=10,
                  max_dims=None, min_dims=None):
    """
    Directly convert a one-channel probability map into a Detections object.

    Helper for Heatmap.detect

    It does this by converting each pixel above a threshold in a probability
    map to a detection with a specified diameter.

    Args:
        probs (ArrayLike[H, W]) a one-channel probability map indicating the
            liklihood that each particular pixel should be detected as an
            object.

        diameter (ArrayLike[2, H, W] | Tuple):
            H, W sizes for the bounding box at each pixel location.
            If passed as a tuple, then all boxes receive that diameter.

        offset (Tuple | ArrayLike[2, H, W], default=0):
           Y, X offsets from the pixel location to the bounding box center.
           If passed as a tuple, then all boxes receive that offset.

        class_probs (ArrayLike[C, H, W], optional):
            probabilities for each class at each pixel location.
            If specified, this will populate the `probs` attribute of the
            returned Detections object.

        keypoints (ArrayLike[2, K, H, W], optional):
            Keypoint predictions for all keypoint classes

        min_score (float, default=0.1): probability threshold required
            for a pixel to be converted into a detection.

        num_min (int, default=10):
            always return at least `nmin` of the highest scoring detections
            even if they aren't above the `min_score` threshold.

    Returns:
        kwimage.Detections: raw detections. It is the users responsbility to
            run non-max suppression on these results to remove duplicate
            detections.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> probs = rng.rand(3, 3).astype(np.float32)
        >>> min_score = .5
        >>> diameter = [10, 10]
        >>> dets = _prob_to_dets(probs, diameter, min_score=min_score)
        >>> assert dets.boxes.data.dtype.kind == 'f'
        >>> assert len(dets) == 9
        >>> dets = _prob_to_dets(torch.FloatTensor(probs), diameter, min_score=min_score)
        >>> assert dets.boxes.data.dtype.is_floating_point
        >>> assert len(dets) == 9

    Example:
        >>> import kwimage
        >>> from kwimage.structs.heatmap import *
        >>> from kwimage.structs.heatmap import _prob_to_dets
        >>> heatmap = kwimage.Heatmap.random(rng=0, dims=(3, 3), keypoints=True)
        >>> # Try with numpy
        >>> min_score = .5
        >>> dets = _prob_to_dets(heatmap.class_probs[0], heatmap.diameter,
        >>>                            heatmap.offset, heatmap.class_probs,
        >>>                            heatmap.data['keypoints'],
        >>>                            min_score)
        >>> assert dets.boxes.data.dtype.kind == 'f'
        >>> assert len(dets) == 9
        >>> dets_np = dets
        >>> # Try with torch
        >>> heatmap = heatmap.tensor()
        >>> dets = _prob_to_dets(heatmap.class_probs[0], heatmap.diameter,
        >>>                            heatmap.offset, heatmap.class_probs,
        >>>                            heatmap.data['keypoints'],
        >>>                            min_score)
        >>> assert dets.boxes.data.dtype.is_floating_point
        >>> assert len(dets) == 9
        >>> dets_torch = dets
        >>> assert np.all(dets_torch.numpy().boxes.data == dets_np.boxes.data)

    Ignore:
        import kwil
        kwil.autompl()
        dets.draw(setlim=True, radius=.1)

    Example:
        >>> heatmap = Heatmap.random(rng=0, dims=(3, 3), diameter=1)
        >>> probs = heatmap.class_probs[0]
        >>> diameter = heatmap.diameter
        >>> offset = heatmap.offset
        >>> class_probs = heatmap.class_probs
        >>> min_score = 0.5
        >>> dets = _prob_to_dets(probs, diameter, offset, class_probs, None, min_score)
    """
    impl = kwarray.ArrayAPI.impl(probs)

    if diameter is None:
        diameter = 1

    if offset is None:
        offset = 0

    diameter_is_uniform = tuple(getattr(diameter, 'shape', []))[1:] != tuple(probs.shape)
    offset_is_uniform = tuple(getattr(offset, 'shape', []))[1:] != tuple(probs.shape)

    if diameter_is_uniform:
        if not ub.iterable(diameter):
            diameter = [diameter, diameter]

    if offset_is_uniform:
        if not ub.iterable(offset):
            offset = impl.asarray([offset, offset])

    flags = probs > min_score
    if not diameter_is_uniform:
        if max_dims is not None:
            max_height, max_width = max_dims
            if max_height is not None:
                flags &= diameter[0] <= max_height
            if max_width is not None:
                flags &= diameter[1] <= max_width
        if min_dims is not None:
            min_height, min_width = min_dims
            if min_height is not None:
                flags &= diameter[0] >= min_height
            if min_width is not None:
                flags &= diameter[1] >= min_width

    # Ensure that some detections are returned even if none are above the
    # threshold.
    if num_min is not None:
        numel = impl.numel(flags)
        if flags.sum() < num_min:
            if impl.is_tensor:
                topxs = probs.view(-1).argsort()[max(0, numel - num_min):numel]
                flags.view(-1)[topxs] = 1
            else:
                idxs = kwarray.argmaxima(probs, num=num_min, ordered=False)
                # idxs = probs.argsort(axis=None)[-num_min:]
                flags.ravel()[idxs] = True

    yc, xc = impl.nonzero(flags)
    yc_ = impl.astype(yc, np.float32)
    xc_ = impl.astype(xc, np.float32)
    if diameter_is_uniform:
        h = impl.full_like(yc_, fill_value=diameter[0])
        w = impl.full_like(xc_, fill_value=diameter[1])
    else:
        h = impl.astype(diameter[0][flags], np.float32)
        w = impl.astype(diameter[1][flags], np.float32)
    cxywh = impl.cat([xc_[:, None], yc_[:, None], w[:, None], h[:, None]], axis=1)

    import kwimage
    tlbr = kwimage.Boxes(cxywh, 'cxywh').toformat('tlbr')
    scores = probs[flags]

    # TODO:
    # Can we extract the detected segmentation mask/poly here as well?

    dets = kwimage.Detections(boxes=tlbr, scores=scores)

    # Get per-class probs for each detection
    if class_probs is not None:
        det_probs = impl.T(class_probs[:, yc, xc])
        dets.data['probs'] = det_probs

    if offset is not None:
        if offset_is_uniform:
            det_dxdy = offset[[1, 0]]
        else:
            det_dxdy = impl.T(offset[:, yc, xc][[1, 0]])
        dets.boxes.translate(det_dxdy, inplace=True)

    if keypoints is not None:
        # Take keypoint predictions for each remaining detection
        det_kpts_xy = impl.contiguous(impl.T(keypoints[:, :, yc, xc][[1, 0]]))
        # Translate keypoints to absolute coordinates
        det_kpts_xy[..., 0] += xc_[:, None]
        det_kpts_xy[..., 1] += yc_[:, None]

        # The shape of det_kpts_xy is [N, K, 2]

        # TODO: need to package kp_classes as well
        # TODO: can we make this faster? It is bottlenecking, in this instance
        # the points list wont be jagged, so perhaps we can use a denser data
        # structure?
        if 1:
            # Try using a dense homogenous data structure
            det_coords = kwimage.Coords(det_kpts_xy)
            det_kpts = kwimage.Points({'xy': det_coords})
        else:
            # Using a jagged non-homogenous data structure is slow
            det_coords = [
                kwimage.Coords(xys) for xys in det_kpts_xy
            ]
            det_kpts = kwimage.PointsList([
                kwimage.Points({'xy': xy}) for xy in det_coords
            ])

        dets.data['keypoints'] = det_kpts

    assert len(dets.scores.shape) == 1
    return dets


def smooth_prob(prob, k=3, inplace=False, eps=1e-9):
    """
    Smooths the probability map, but preserves the magnitude of the peaks.

    Notes:
        even if inplace is true, we still need to make a copy of the input
        array, however, we do ensure that it is cleaned up before we leave the
        function scope.

        sigma=0.8 @ k=3, sigma=1.1 @ k=5, sigma=1.4 @ k=7
    """
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # opencv formula
    blur = cv2.GaussianBlur(prob, (k, k), sigma)
    # Shift and scale the intensities so the maximum and minimum
    # pixel value in the blurred image match the original image
    minpos = np.unravel_index(blur.argmin(), blur.shape)
    maxpos = np.unravel_index(blur.argmax(), blur.shape)
    shift = prob[minpos] - blur[minpos]
    scale = prob[maxpos] / np.maximum((blur[maxpos] + shift), eps)
    if inplace:
        prob[:] = blur
        blur = prob
    np.add(blur, shift, out=blur)
    np.multiply(blur, scale, out=blur)
    return blur


def _remove_translation(tf):
    """
    Removes the translation component of a transform

    TODO:
        - [ ] Is this possible in more general cases? E.g. projective transforms?
    """
    if isinstance(tf, skimage.transform.AffineTransform):
        tf_notrans = skimage.transform.AffineTransform(
            scale=tf.scale, rotation=tf.rotation, shear=tf.shear)
    elif isinstance(tf, skimage.transform.SimilarityTransform):
        tf_notrans = skimage.transform.SimilarityTransform(
            scale=tf.scale, rotation=tf.rotation)
    elif isinstance(tf, skimage.transform.EuclideanTransform):
        tf_notrans = skimage.transform.EuclideanTransform(
            scale=tf.scale, rotation=tf.rotation)
    else:
        raise TypeError(tf)
    return tf_notrans


def _gmean(a, axis=0, clobber=False):
    """
    Compute the geometric mean along the specified axis.

    Modification of the scipy.mstats method to be more memory efficient

    Example
        >>> rng = np.random.RandomState(0)
        >>> C, H, W = 8, 32, 32
        >>> axis = 0
        >>> a = rng.rand(2, C, H, W)
        >>> _gmean(a)
    """
    assert isinstance(a, np.ndarray)

    if clobber:
        # NOTE: we reuse (a), we clobber the input array!
        log_a = np.log(a, out=a)
    else:
        log_a = np.log(a)

    # attempt to reuse memory when computing mean
    mem = log_a[tuple([slice(None)] * axis + [0])]
    mean_log_a = log_a.mean(axis=axis, out=mem)

    # And reuse memory again when computing the final result
    result = np.exp(mean_log_a, out=mean_log_a)

    return result
