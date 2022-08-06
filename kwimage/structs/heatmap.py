"""
TODO:
    - [ ] Remove doctest dependency on ndsampler?

    - [ ] Remove the datakeys that tries to define what heatmap should represent
          (e.g. class_probs, keypoints, etc...) and instead just focus on a
          data structure that stores a [C, H, W] or [H, W] tensor?

CommandLine:
    xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py __doc__

Example:
    >>> # xdoctest: +REQUIRES(module:ndsampler)
    >>> # xdoctest: +REQUIRES(--mask)
    >>> from kwimage.structs.heatmap import *  # NOQA
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
    >>> # xdoctest: +REQUIRES(--mask)
    >>> from kwimage.structs.heatmap import *  # NOQA
    >>> from kwimage.structs.detections import _dets_to_fcmaps
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
import cv2
import numpy as np
import ubelt as ub
import skimage
import kwarray
import functools
from . import _generic

try:
    import torch
except Exception:
    torch = None


class _HeatmapDrawMixin(object):
    """
    mixin methods for drawing heatmap details
    """

    def _colorize_class_idx(self):
        """
        """
        # Ignore cases where index is negative?
        cidxs = kwarray.ArrayAPI.numpy(self.data['class_idx']).astype(int)

        import networkx as nx
        import kwimage

        classes = self.meta['classes']
        backup_colors = iter(kwimage.Color.distinct(len(classes)))

        name_to_color = {}

        if hasattr(classes, 'graph'):
            name_to_color = nx.get_node_attributes(classes.graph, 'color')
            for node in classes.graph.nodes:
                color = classes.graph.nodes[node].get('color', None)
                if color is None:
                    color = next(backup_colors)
                name_to_color[node] = kwimage.Color(color).as01()
        else:
            name_to_color = ub.dzip(classes, backup_colors)

        cx_to_color = np.array([name_to_color[cname] for cname in classes])
        colorized = cx_to_color[cidxs]
        return colorized

    def colorize(self, channel=None, invert=False, with_alpha=1.0,
                 interpolation='linear', imgspace=False, cmap=None):
        """
        Creates a colorized version of a heatmap channel suitable for
        visualization

        Args:
            channel (int | str): index of category to visualize, or a special
                code indicating how to visualize multiple classes.
                Can be class_idx, class_probs, or class_energy.

            imgspace (bool): colorize the image after
                warping into the image space.

        CommandLine:
            xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py _HeatmapDrawMixin.colorize --show

        Ignore:
            import xdev
            from kwimage.structs.heatmap import *  # NOQA
            globals().update(xdev.get_func_kwargs(Heatmap.colorize))

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
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> colormask1 = self.colorize('diameter', imgspace=False)
            >>> colormask2 = self.colorize('diameter', imgspace=True)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> kwplot.show_if_requested()

        Ignore:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> self.data['class_energy'] = (self.data['class_probs'] - .5) * 10
            >>> colormask1 = self.colorize('class_energy_color', imgspace=False)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(colormask1, fnum=1, title='output space')
            >>> kwplot.show_if_requested()

        Ignore:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwarray
            >>> import kwimage
            >>> rng = kwarray.ensure_rng(0)
            >>> class_probs = np.zeros((2, 32, 32))
            >>> class_probs[0] = kwimage.Polygon.random(rng=rng).scale(16).translate(16).fill(class_probs[0], value=0.5)
            >>> class_probs[1] = kwimage.Polygon.random(rng=rng).scale(32).fill(class_probs[1], value=0.5)
            >>> self = kwimage.Heatmap(class_probs=class_probs)
            >>> canvas = self.colorize()
            >>> canvas = kwimage.overlay_alpha_images(canvas, np.zeros_like(canvas[:, :, 0:3]))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
        """
        import kwplot

        if channel is None:
            if 'class_idx' in self.data:
                channel = 'class_idx'
            elif 'class_probs' in self.data:
                channel = 'class_probs'
            elif 'class_energy' in self.data:
                channel = 'class_energy'
            else:
                raise Exception('unsure how to default channel')

        def _per_channel_color(data, with_alpha, classes=None):
            # Another hacky mode
            # data = a.data['class_energy']
            import kwimage

            if len(data.shape) == 2:
                # add in prefix channel if its not there
                data = data[None, :, :]

            # Define default colors
            default_cidx_to_color = kwimage.Color.distinct(len(data))

            # try and read colors from classes CategoryTree
            try:
                cidx_to_color = []
                for cidx in range(len(data)):
                    node = classes[cidx]
                    color = classes.graph.nodes[node].get('color', None)
                    if True:
                        assert color is not None
                    if color is None:
                        # fallback, ignore conflicts
                        color = default_cidx_to_color[cidx]
                    else:
                        color = kwimage.Color(color).as01()
                    cidx_to_color.append(color)
            except Exception:
                # fallback on default colors
                cidx_to_color = default_cidx_to_color

            # Each class gets its own color, and modulates the alpha
            layers = []
            for cidx, chan in enumerate(data):
                color = cidx_to_color[cidx]
                layer = np.empty(tuple(chan.shape) + (4,))
                layer[..., 3] = chan
                layer[..., 0:3] = color
                layers.append(layer)

            colormask = kwimage.overlay_alpha_layers(layers)
            colormask[..., 3] *= with_alpha
            return colormask

        if channel in ['class_idx', 'idx']:
            # HACK
            import kwimage
            colormask = self._colorize_class_idx()
            colormask = kwimage.ensure_alpha_channel(colormask, with_alpha)
            if imgspace:
                chw = torch.Tensor(colormask.transpose(2, 0, 1))
                colormask = self._warp_imgspace(chw, interpolation=interpolation).transpose(1, 2, 0)
            return colormask

        if isinstance(channel, str):
            # TODO: this is a bit hacky / inefficient, needs cleanup
            if imgspace:
                mat = self.tf_data_to_img.params
                output_dims = self.img_dims
                a = self.warp(mat, version='old',
                              output_dims=output_dims).numpy()
            else:
                a = self
            if channel == 'offset':
                mask = np.linalg.norm(a.offset, axis=0)
            elif channel == 'diameter':
                mask = np.linalg.norm(a.diameter, axis=0)
            elif channel == 'class_probs_max':
                if 'class_probs' in a.data:
                    data = a.data['class_probs']
                else:
                    # HACK HACK HACK
                    data = a.data['class_energy']
                    low = min(0, data.min())
                    high = max(1, data.max())
                    data = (data - low) / (high - low)
                mask = data.max(axis=0)
            elif channel == 'class_energy_max':
                mask = a.data['class_energy'].max(axis=0)
                mask -= mask.min()
            elif channel == 'class_probs_color' or channel == 'class_probs':
                if 'class_probs' in a.data:
                    data = a.data['class_probs']
                else:
                    # HACK HACK HACK
                    data = a.data['class_energy']
                    low = min(0, data.min())
                    high = max(1, data.max())
                    data = (data - low) / (high - low)
                classes = self.classes
                colormask = _per_channel_color(data, with_alpha, classes)
                return colormask
            elif channel == 'class_energy_color' or channel == 'class_energy':
                # Another hacky mode
                import scipy
                import scipy.special
                data = a.data['class_energy']
                if 1:
                    # Assume 0-1 range, but stretch beyond if needed
                    low = min(0, data.min())
                    high = max(1, data.max())
                    data = (data - low) / (high - low)
                else:
                    data = scipy.special.softmax(data, axis=0)
                classes = self.classes
                colormask = _per_channel_color(data, with_alpha, classes)
                return colormask
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

        if cmap is None:
            cmap = 'plasma'
        colormask = kwplot.make_heatmask(mask, with_alpha=with_alpha, cmap=cmap)
        return colormask

    def draw_stacked(self, image=None, dsize=(224, 224), ignore_class_idxs={},
                     top=None, chosen_cxs=None):
        """
        Draws per-class probabilities and stacks them into a single image

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

        mat = None
        if image is not None:
            tf = self.tf_data_to_img
            if tf is not None:
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
            if mat is not None:
                # warp image into dataspace
                dataspace_img = cv2.warpAffine(image, mat[0:2], dsize=level_dsize)
            else:
                dataspace_img = image
            small_img = cv2.resize(dataspace_img, dsize)
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

    def draw(self, channel=None, image=None, imgspace=None, **kwargs):
        """
        Accepts same args as draw_on, but uses maplotlib

        Args:
            channel (int | str): category index to visualize, or special key

        """
        # If draw doesnt exist use draw_on
        import numpy as np
        import kwplot
        if image is None:
            if imgspace:
                dims = self.img_dims
            else:
                dims = self.bounds
            shape = tuple(dims) + (4,)
            image = np.zeros(shape, dtype=np.float32)
        image = self.draw_on(image, channel=channel, imgspace=imgspace,
                             **kwargs)
        kwplot.imshow(image)

    def draw_on(self, image=None, channel=None, invert=False, with_alpha=1.0,
                interpolation='linear', vecs=False, kpts=None, imgspace=None):
        """
        Overlays a heatmap channel on top of an image

        Args:
            image (ndarray): image to draw on, if unspecified one is created.

            channel (int | str): category index to visualize, or special key.
                special keys are: class_idx, class_probs, class_idx

            imgspace (bool): colorize the image after
                warping into the image space.

        TODO:
            - [ ] Find a way to visualize offset, diameter, and class_probs
                  either individually or all at the same time

          CommandLine:
              xdoctest -m /home/joncrall/code/kwimage/kwimage/structs/heatmap.py

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

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwimage
            >>> mask = np.random.rand(32, 32)
            >>> self = kwimage.Heatmap(
            >>>     class_probs=mask,
            >>>     img_dims=mask.shape[0:2],
            >>>     tf_data_to_img=np.eye(3),
            >>> )
            >>> canvas = self.draw_on()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(Heatmap.draw_on))

        """
        import kwimage

        if image is None:
            if imgspace:
                image = np.zeros(self.img_dims)
            else:
                image = np.zeros((*self.shape[-2:], 3))

        if channel is None:
            if 'class_idx' in self.data:
                channel = 'class_idx'
            elif 'class_probs' in self.data:
                channel = 'class_probs'
            elif 'class_energy' in self.data:
                channel = 'class_energy'
            else:
                raise Exception('unsure how to default channel')

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

        vec_colors = kwimage.Color.distinct(2)
        vec_alpha = .5

        if kpts is not None:
            # TODO: make a nicer keypoint offset vector visuliazation
            if kpts is True:
                if self.data.get('keypoints', None) is not None:
                    keypoints = self.data['keypoints']
                    kpts = list(range(len(keypoints.shape[1])))
            if not ub.iterable(kpts):
                kpts = [kpts]
            E = int(bool(vecs))
            vec_colors = kwimage.Color.distinct(len(kpts) + E)

        if vecs:
            if self.data.get('offset', None) is not None:
                #Hack
                # Visualize center offset vectors
                dy, dx = kwarray.ArrayAPI.numpy(self.data['offset'])
                color = vec_colors[0]
                vecmask = kwimage.make_vector_field(
                    dx, dy, stride=4, scale=1.0, alpha=with_alpha * vec_alpha,
                    color=color)
                vec_alpha = max(.1, vec_alpha - .1)
                chw = torch.Tensor(vecmask.transpose(2, 0, 1))
                vecalign = self._warp_imgspace(chw, interpolation=interpolation)
                vecalign = vecalign.transpose(1, 2, 0)
                layers.append(vecalign)

        if kpts is not None:
            # TODO: make a nicer keypoint offset vector visuliazation
            if self.data.get('keypoints', None) is not None:
                keypoints = self.data['keypoints']
                for i, k in enumerate(kpts):
                    # color = (np.array(vec_colors[k]) * 255).astype(np.uint8)
                    color = vec_colors[i + E]

                    dy, dx = kwarray.ArrayAPI.numpy(keypoints[:, k])
                    vecmask = kwimage.make_vector_field(dx, dy, stride=8,
                                                        scale=0.5,
                                                        alpha=with_alpha *
                                                        vec_alpha, color=color)
                    vec_alpha = max(.1, vec_alpha - .1)
                    chw = torch.Tensor(vecmask.transpose(2, 0, 1))
                    vecalign = self._warp_imgspace(chw, interpolation=interpolation)
                    vecalign = vecalign.transpose(1, 2, 0)
                    layers.append(vecalign)

        layers.append(colormask)
        layers.append(image)

        overlaid = kwimage.overlay_alpha_layers(layers)
        overlaid = dtype_fixer(overlaid, copy=False)
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
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> self = Heatmap.random((120, 130), img_dims=(200, 210), classes=2, nblips=10, rng=0)
            >>> other = Heatmap.random((60, 70), img_dims=(200, 210), classes=2, nblips=10, rng=1)
            >>> other2 = self._align_other(other)
            >>> assert self.shape != other.shape
            >>> assert self.shape == other2.shape
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(self.colorize(0, imgspace=False), fnum=1, pnum=(2, 2, 1))
            >>> kwplot.imshow(self.colorize(1, imgspace=False), fnum=1, pnum=(2, 2, 2))
            >>> kwplot.imshow(other.colorize(0, imgspace=False), fnum=1, pnum=(2, 2, 3))
            >>> kwplot.imshow(other.colorize(1, imgspace=False), fnum=1, pnum=(2, 2, 4))
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
        flags = kwimage.im_cv2._coerce_interpolation(interpolation)
        aligned = cv2.warpAffine(mask, M[0:2], dsize=tuple(dsize), flags=flags)
        aligned = np.clip(aligned, 0, 1)
        return aligned

    def _warp_imgspace(self, chw, interpolation='linear'):
        import kwimage
        if self.tf_data_to_img is None and self.img_dims is None:
            aligned = chw.cpu().numpy()
        else:
            if self.tf_data_to_img is None:
                # If img dims are the same then we dont need a transform we
                # know its identity
                if self.img_dims == self.dims:
                    return chw.cpu().numpy()

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

        Args:
            channel (ndarray | None):
                if None, use class probs, else chw data.

        TODO:
            - [ ] Needs refactor

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Heatmap.random(rng=0, dims=(32, 32))
            >>> colormask = self.upscale()

        """
        if channel is None:
            chw = torch.Tensor(self.class_probs)
        else:
            chw = torch.Tensor(self.class_probs[channel])[None, :]
        aligned = self._warp_imgspace(chw, interpolation=interpolation)
        return aligned

    # @profile
    def warp(self, mat=None, input_dims=None, output_dims=None,
             interpolation='linear', modify_spatial_coords=True,
             int_interpolation='nearest', mat_is_xy=True, version=None):
        """
        Warp all spatial maps. If the map contains spatial data, that data is
        also warped (ignoring the translation component).

        Args:
            mat (ArrayLike): transformation matrix
            input_dims (tuple): unused, only exists for compatibility
            output_dims (tuple): size of the output heatmap
            interpolation (str): see `kwimage.warp_tensor`
            int_interpolation (str): interpolation used for interger types (should be nearest)
            mat_is_xy (bool): set to false if the matrix
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
            >>> # xdoctest: +REQUIRES(module:torch)
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
            >>> # xdoctest: +REQUIRES(module:kwplot)
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

        if isinstance(mat, skimage.transform.AffineTransform):
            mat = mat.params
        elif isinstance(mat, kwimage.Affine):
            mat = mat.matrix

        newdata = {}
        newmeta = self.meta.copy()

        impl = kwarray.ArrayAPI.coerce('tensor')

        if version is None:
            import warnings
            warnings.warn(ub.paragraph(
                '''
                The old mat_is_xy logic has changed. Please ensure your
                application works with the old logic. Then set version='old'
                or 'new'. Both disable this warning message.
                '''))
            version = 'old'

        # Change if matrix is in X/Y or Y/X coords.
        if version == 'new':
            if not mat_is_xy:
                mat = mat[[1, 0, 2], :][:, [1, 0, 2]]
        elif version == 'old':
            if mat_is_xy:
                mat = mat[[1, 0, 2], :][:, [1, 0, 2]]
        else:
            raise KeyError(version)

        mat = impl.asarray(mat)

        mat_np = impl.numpy(mat)
        tf = skimage.transform.AffineTransform(matrix=mat_np)
        # hack: need to get a version of the matrix without any translation
        tf_notrans = _remove_translation(tf)
        mat_notrans = torch.Tensor(tf_notrans.params)

        if output_dims is None:
            # If output dimensions are not specified warp the existing dims
            # according to scale. NOTE: old behavior was to use the img_dims
            # but this has problems when we are making something smaller.
            def _auto_select_warped_output_shape(mat):
                h, w = self.dims
                # Warp corners of the box and determine a new output shape
                corners = kwimage.Coords(np.array([
                    [0., 0], [w, 0], [w, h], [0, h],
                ]))
                corners2 = corners.warp(mat.numpy())
                wh2 = corners2.data.clip(1, None).max(axis=0)
                w2, h2 = np.ceil(wh2).astype(int).tolist()
                output_dims = (w2, h2)
                return output_dims
            output_dims = _auto_select_warped_output_shape(mat_notrans)
            if self.img_dims is not None:
                import warnings
                warnings.warn(
                    'NOTE: automatic selection of output_dims has changed. '
                    'Previously it would use the img_dims, but now it calculates '
                    'the output_dims based on mat and the current dims. '
                    'Please check that your code still works and specify '
                    'output_dims explicitly to supress this message.')
                # output_dims = self.img_dims

        # Modify data_to_img so the new heatmap will also properly upscale to
        # the image coordinates.
        inv_tf = skimage.transform.AffineTransform(matrix=tf._inv_matrix)
        # newmeta['tf_data_to_img'] = self.tf_data_to_img + inv_tf
        # NOTE: The old models were working with the above code, but I think
        # thats because there was no translation factor. I'm pretty sure the
        # code on the bottom is correct. Obviously if something messes up, it
        # should probably be reverted. Left-vs-right is hard.
        if self.tf_data_to_img is not None:
            newmeta['tf_data_to_img'] = inv_tf + self.tf_data_to_img

        for k, v in self.data.items():
            if v is not None:
                v = kwarray.ArrayAPI.tensor(v)
                # For spatial keys we need to transform the underlying values
                # in addition to where those values are located.
                if modify_spatial_coords:
                    if k in self.__spatialkeys__:
                        pts = impl.contiguous(impl.T(v))
                        pts = kwimage.warp_points(mat_notrans, pts)
                        v = impl.contiguous(impl.T(pts))

                if kwarray.ArrayAPI.dtype_kind(v) == 'i':
                    # use different interpolation for integer types
                    if int_interpolation != 'nearest':
                        warnings.warn('Using non-nearest int interpolation')
                    new_v = kwimage.warp_tensor(
                        v[None, :].float(), mat, output_dims=output_dims,
                        mode=int_interpolation)[0]
                else:
                    new_v = kwimage.warp_tensor(
                        v[None, :].float(), mat, output_dims=output_dims,
                        mode=interpolation)[0]

                newdata[k] = impl.asarray(new_v)

        newself = self.__class__(newdata, newmeta)
        return newself

    def scale(self, factor, output_dims=None, interpolation='linear'):
        """
        Scale the heatmap
        """
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
                         version='old',
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
                         version='old',
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
            >>> # xdoctest: +REQUIRES(module:torch)
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

    def detect(self, channel, invert=False, min_score=0.01, num_min=10,
               max_dims=None, min_dims=None, dim_thresh_space='image'):
        """
        Lossy conversion from a Heatmap to a Detections object.

        For efficiency, the detections are returned in the same space as the
        heatmap, which usually some downsampled version of the image space.
        This is because it is more efficient to transform the detections into
        image-space after non-max supression is applied.

        Args:
            channel (int | ArrayLike):
                class index to detect objects in.
                Alternatively, channel can be a custom probability map as long
                as its dimension agree with the heatmap.

            invert (bool): if True, inverts the probabilities in
                the chosen channel. (Useful if you have a background channel
                but want to detect foreground objects).

            min_score (float): probability threshold required
                for a pixel to be converted into a detection. Defaults to 0.1

            num_min (int):
                always return at least `nmin` of the highest scoring detections
                even if they aren't above the `min_score` threshold. Defaults
                to 10.

            max_dims (Tuple[int, int]): maximum height / width of detections
                By default these are expected to be in image-space.

            min_dims (Tuple[int, int]): minimum height / width of detections
                By default these are expected to be in image-space.

            dim_thresh_space (str):
                When dim_thresh_space=='native', dimension thresholds (e.g.
                min_dims and max_dims) are specified in the native heatmap
                space (i.e.  usually a downsampled space). If
                dim_thresh_space=='image', then dimension thresholds are
                interpreted in the original image space. Defaults to 'image'

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
            >>> self = Heatmap.random(rng=2, dims=(32, 32))
            >>> dets = self.detect(channel=0, max_dims=7, num_min=None)
            >>> img_dets = dets.warp(self.tf_data_to_img)
            >>> assert img_dets.boxes.to_xywh().width.max() <= 7
            >>> assert img_dets.boxes.to_xywh().height.max() <= 7
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dets1 = dets.sort().take(range(30))
            >>> colormask1 = self.colorize(0, imgspace=False)
            >>> kwplot.imshow(colormask1, pnum=(1, 2, 1), fnum=1, title='output space')
            >>> dets1.draw()
            >>> # Transform heatmap and detections into image space.
            >>> dets2 = dets1.warp(self.tf_data_to_img)
            >>> colormask2 = self.colorize(0, imgspace=True)
            >>> kwplot.imshow(colormask2, pnum=(1, 2, 2), fnum=1, title='image space')
            >>> dets2.draw()

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwimage.structs.heatmap import *  # NOQA
            >>> import ndsampler
            >>> catgraph = ndsampler.CategoryTree.demo()
            >>> class_energy = torch.rand(len(catgraph), 32, 32)
            >>> class_probs = catgraph.hierarchical_softmax(class_energy, dim=0)
            >>> self = Heatmap.random(rng=0, dims=(32, 32), classes=catgraph, keypoints=True)
            >>> print(ub.repr2(ub.map_vals(lambda x: x.shape, self.data), nl=1))
            >>> self.data['class_probs'] = class_probs.numpy()
            >>> channel = catgraph.index('background')
            >>> dets = self.detect(channel, invert=True)
            >>> class_idx, scores = catgraph.decision(dets.probs, dim=1)
            >>> dets.data['class_idx'] = class_idx
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

        if max_dims is not None:
            max_dims = max_dims if ub.iterable(max_dims) else (max_dims, max_dims)
            max_dims = np.array(max_dims)

        elif min_dims is not None:
            min_dims = min_dims if ub.iterable(min_dims) else (min_dims, min_dims)
            min_dims = np.array(min_dims)

        # Convert the dims to a native space if necessary
        if dim_thresh_space == 'image':
            # convert thresholds to native space
            # NOT SURE IF WE NEED TO INVERT XY HERE OR NOT
            scale_dims = self.tf_data_to_img.scale[::-2]
            if max_dims is not None:
                max_dims = max_dims / scale_dims
            if min_dims is not None:
                min_dims = min_dims / scale_dims
        elif dim_thresh_space != 'native':
            raise KeyError(dim_thresh_space)

        dets = _prob_to_dets(
            probs,
            diameter=self.data.get('diameter', None),
            offset=self.data.get('offset', None),
            class_probs=self.data.get('class_probs', None),
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

        meta (Dict[str, object]): dictionary containing miscellanious metadata
            about the heatmap data. Valid keys are as follows.

            img_dims (Tuple[H, W] | Tuple[D, H, W]):
                original image dimension

            tf_data_to_image (skimage.transform._geometric.GeometricTransform):
                transformation matrix (typically similarity or affine) that
                projects the given, heatmap onto the image dimensions such that
                the image and heatmap are spatially aligned.

            classes (List[str] | ndsampler.CategoryTree):
                information about which index in ``data['class_probs']``
                corresponds to which semantic class.

        dims (Tuple): dimensions of the heatmap (See ``image_dims``) for the
            original image dimensions.

        **kwargs: any key that is accepted by the `data` or `meta` dictionaries
            can be specified as a keyword argument to this class and it will
            be properly placed in the appropriate internal dictionary.

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/structs/heatmap.py Heatmap --show

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> from kwimage.structs.heatmap import *  # NOQA
        >>> import kwimage
        >>> class_probs = kwimage.grab_test_image(dsize=(32, 32), space='gray')[None, ..., 0] / 255.0
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

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import kwimage
        >>> self = Heatmap.random()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw()
    """
    # Valid keys for the data dictionary
    __datakeys__ = ['class_probs', 'offset', 'diameter', 'keypoints',
                    'class_idx', 'class_energy']

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

            tf_data_to_img = meta.get('tf_data_to_img', None)
            if tf_data_to_img is not None:
                if isinstance(tf_data_to_img, np.ndarray):
                    meta['tf_data_to_img'] = skimage.transform.AffineTransform(
                        matrix=tf_data_to_img)

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
        return '{} on img_dims={}'.format(self.shape, self.img_dims)

    def __getitem__(self, index):
        return self.class_probs[index]

    def __len__(self):
        return len(self.class_probs)

    @property
    def shape(self):
        shape = None
        try:
            shape = self.class_probs.shape
        except Exception:
            for key, value in self.data.items():
                try:
                    shape = value.shape
                except AttributeError:
                    pass
        return shape

    @property
    def bounds(self):
        return self.shape[-2:]
        # return self.class_probs.shape[1:]

    @property
    def dims(self):
        """ space-time dimensions of this heatmap """
        return self.shape[-2:]
        # return self.class_probs.shape[1:]

    def is_numpy(self):
        return self._impl.is_numpy

    def is_tensor(self):
        return self._impl.is_tensor

    @property
    def _impl(self):
        """
        Returns the internal tensor/numpy ArrayAPI implementation

        Returns:
            kwarray.ArrayAPI
        """
        return kwarray.ArrayAPI.coerce(self.data['class_probs'])

    # @property
    # def device(self):
    #     """ If the backend is torch returns the data device, otherwise None """
    #     return self.data['class_probs'].device

    @classmethod
    def random(cls, dims=(10, 10), classes=3, diameter=True, offset=True,
               keypoints=False, img_dims=None, dets=None, nblips=10, noise=0.0,
               smooth_k=3, rng=None, ensure_background=True):
        """
        Creates dummy data, suitable for use in tests and benchmarks

        Args:
            dims (Tuple[int, int]): dimensions of the heatmap

            classes (int | List[str] | kwcoco.CategoryTree):
                foreground classes

            diameter (bool): if True, include a "diameter" heatmap

            offset (bool): if True, include an "offset" heatmap

            keypoints (bool):

            smooth_k (int): kernel size for gaussian blur to smooth out
                the heatmaps.

            img_dims (Tuple):
                dimensions of an upscaled image the heatmap corresponds to.
                (This should be removed and simply handled with a transform
                 in the future).

        Returns:
            Heatmap

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
            >>> # xdoctest: +REQUIRES(module:kwplot)
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
            img_wh_dims = tuple(np.ceil(tf_data_to_img([wh_dims]))[0].astype(int).tolist())
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
            if ensure_background:
                if 'background' not in dets.classes:
                    dets.classes.append('background')

            classes = dets.classes
        else:
            classes = dets.classes
        # assume we have background
        # bg_idx = dets.classes.index('background')

        # Warp detections into heatmap space
        transform = np.linalg.inv(tf_data_to_img.params)
        warped_dets = dets.warp(transform, input_dims=img_dims,
                                output_dims=dims)

        tf_notrans = _remove_translation(tf_data_to_img)
        bg_size = tf_notrans.inverse([100, 100])[0]

        self = warped_dets.rasterize(bg_size, input_dims=dims, soften=1,
                                     img_dims=img_dims,
                                     tf_data_to_img=tf_data_to_img)

        class_probs = self.data['class_probs']

        noise = (rng.randn(*class_probs.shape) * noise)
        class_probs += noise
        np.clip(class_probs, 0, None, out=class_probs)
        # class_probs = class_probs / class_probs.sum(axis=0)
        class_probs = np.array([smooth_prob(p, k=smooth_k) for p in class_probs])
        class_probs = class_probs / np.maximum(class_probs.sum(axis=0), 1e-9)

        if not offset:
            self.data.pop('offset')

        if not diameter:
            self.data.pop('diameter')

        if keypoints is not False and keypoints is not None:
            # self.data['keypoints'] = keypoints
            if 'kp_classes' not in locals():
                kp_classes = list(range(self.data['keypoints'].shape[1]))  # HACK
            self.meta['kp_classes'] = kp_classes

        self.meta['classes'] = classes

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

        offset (Tuple | ArrayLike[2, H, W]):
           Y, X offsets from the pixel location to the bounding box center.
           If passed as a tuple, then all boxes receive that offset.

        class_probs (ArrayLike[C, H, W], optional):
            probabilities for each class at each pixel location.
            If specified, this will populate the `probs` attribute of the
            returned Detections object.

        keypoints (ArrayLike[2, K, H, W], optional):
            Keypoint predictions for all keypoint classes

        min_score (float): probability threshold required
            for a pixel to be converted into a detection.
            Defaults to 0.1

        num_min (int):
            always return at least `nmin` of the highest scoring detections
            even if they aren't above the `min_score` threshold.  Defaults to
            10

    Returns:
        kwimage.Detections: raw detections. It is the users responsbility to
            run non-max suppression on these results to remove duplicate
            detections.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
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
        >>> # xdoctest: +REQUIRES(module:torch)
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
        >>> assert 'keypoints' in dets.data
        >>> dets_np = dets
        >>> # Try with torch
        >>> heatmap = heatmap.tensor()
        >>> dets = _prob_to_dets(heatmap.class_probs[0], heatmap.diameter,
        >>>                            heatmap.offset, heatmap.class_probs,
        >>>                            heatmap.data['keypoints'],
        >>>                            min_score)
        >>> assert dets.boxes.data.dtype.is_floating_point
        >>> assert len(dets) == len(dets_np)
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

        if hasattr(diameter, 'shape'):
            if len(diameter.shape) > 2:
                raise Exception('Trailing diameter shape={} does not agree with probs.shape={}'.format(
                    diameter.shape, probs.shape))

        if not ub.iterable(diameter):
            diameter = [diameter, diameter]

    if offset_is_uniform:
        if not ub.iterable(offset):
            offset = impl.asarray([offset, offset])

    flags = probs > min_score
    if not diameter_is_uniform:
        if max_dims is not None:
            max_dims = max_dims if ub.iterable(max_dims) else (max_dims, max_dims)
            max_height, max_width = max_dims

            if max_height is not None:
                flags &= diameter[0] <= max_height
            if max_width is not None:
                flags &= diameter[1] <= max_width
        if min_dims is not None:
            min_dims = min_dims if ub.iterable(min_dims) else (min_dims, min_dims)
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
    ltrb = kwimage.Boxes(cxywh, 'cxywh').toformat('ltrb')
    scores = probs[flags]

    # TODO:
    # Can we extract the detected segmentation mask/poly here as well?

    dets = kwimage.Detections(boxes=ltrb, scores=scores)

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

    Note:
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
