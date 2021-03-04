# -*- coding: utf-8 -*-
"""
Structure for efficient access and modification of bounding boxes with
associated scores and class labels. Builds on top of the `kwimage.Boxes`
structure.

Also can optionally incorporate `kwimage.PolygonList` for segmentation masks
and `kwimage.PointsList` for keypoints.


If you want to visualize boxes and scores you can do this:
    >>> # Given data
    >>> data = np.random.rand(10, 4) * 224
    >>> scores = np.random.rand(10,)
    >>> class_idxs = np.random.randint(0, 3, size=10)
    >>> classes = ['class1', 'class2', 'class3']
    >>> #
    >>> # Wrap your data with a Detections object
    >>> import kwimage
    >>> dets = kwimage.Detections(
    >>>     boxes=kwimage.Boxes(data, format='xywh'),
    >>>     scores=scores,
    >>>     class_idxs=class_idxs,
    >>>     classes=classes,
    >>> )
    >>> dets.draw()
    >>> import matplotlib.pyplot as plt
    >>> plt.gca().set_xlim(0, 224)
    >>> plt.gca().set_ylim(0, 224)

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import ubelt as ub
from kwimage.structs import boxes as _boxes
from kwimage.structs import _generic
from distutils.version import LooseVersion


try:
    from xdev import profile
except Exception:
    from ubelt import identity as profile

try:
    import torch
except Exception:
    torch = None
    _TORCH_HAS_BOOL_COMP = False
else:
    _TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


class _DetDrawMixin:
    """
    Non critical methods for visualizing detections
    """
    def draw(self, color='blue', alpha=None, labels=True, centers=False, lw=2,
             fill=False, ax=None, radius=5, kpts=True, sseg=True,
             setlim=False, boxes=True):
        """
        Draws boxes using matplotlib

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> self = Detections.random(num=10, scale=512.0, rng=0, classes=['a', 'b', 'c'])
            >>> self.boxes.translate((-128, -128), inplace=True)
            >>> image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> self.draw(color='blue', alpha=None)
            >>> # xdoc: +REQUIRES(--show)
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()
        """
        segmentations = self.data.get('segmentations', None)
        if sseg and segmentations is not None:
            segmentations.draw(color=color, alpha=.4)

        labels = self._make_labels(labels)
        alpha = self._make_alpha(alpha)
        if boxes:
            self.boxes.draw(labels=labels, color=color, alpha=alpha, fill=fill,
                            centers=centers, ax=ax, lw=lw)

        keypoints = self.data.get('keypoints', None)
        if kpts and keypoints is not None:
            keypoints.draw(color=color, radius=radius)

        if setlim:
            x1, y1, x2, y2 = self.boxes.to_ltrb().components
            xmax = x2.max()
            xmin = x1.min()
            ymax = y2.max()
            ymin = y1.min()
            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    def draw_on(self, image, color='blue', alpha=None, labels=True, radius=5,
                kpts=True, sseg=True, boxes=True, ssegkw=None,
                label_loc='top_left', thickness=2):
        """
        Draws boxes directly on the image using OpenCV

        Args:
            image (ndarray[uint8]): must be in uint8 format

            color (str | ColorLike | List[ColorLike]):
                one color for all boxes or a list of colors for each box

            alpha (float): Transparency of overlay. can be a scalar or a list
                for each box

            labels (bool | str | List[str]):
                if True, use categorie names as the labels. See _make_labels
                for details. Otherwise a manually specified text label for each
                box.

            boxes (bool): if True draw the boxes

            kpts (bool): if True draw the keypoints

            sseg (bool): if True draw the segmentations

            ssegkw (dict): extra arguments passed to `segmentations.draw_on`

            radius (float): passed to `keypoints.draw_on`

            label_loc (str): indicates where labels (if specified) should be
                drawn. passed to `boxes.draw_on`

            thickness (int, default=2): rectangle thickness, negative values
                will draw a filled rectangle. passed to `boxes.draw_on`

        Returns:
            ndarray[uint8]: image with labeled boxes drawn on it

        CommandLine:
            xdoctest -m kwimage.structs.detections _DetDrawMixin.draw_on:1 --profile --show

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> self = Detections.random(num=10, scale=512, rng=0)
            >>> image = (np.random.rand(512, 512) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image, color='blue')
            >>> # xdoc: +REQUIRES(--show)
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image2)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> from kwimage.structs.detections import *  # NOQA
            >>> import kwplot
            >>> self = Detections.random(num=10, scale=512, rng=0)
            >>> image = (np.random.rand(512, 512) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image, color='classes')
            >>> # xdoc: +REQUIRES(--show)
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image2)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoc: +REQUIRES(module:kwplot)
            >>> # xdoc: +REQUIRES(--profile)
            >>> import kwplot
            >>> self = Detections.random(num=100, scale=512, rng=0, keypoints=True, segmentations=True)
            >>> image = (np.random.rand(512, 512) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image, color='blue')
            >>> # xdoc: +REQUIRES(--show)
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image2)
            >>> kwplot.show_if_requested()

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(kwimage.Detections.draw_on))
        """
        labels = self._make_labels(labels)
        alpha = self._make_alpha(alpha)
        color = self._make_colors(color)

        dtype_fixer = _generic._consistent_dtype_fixer(image)

        segmentations = self.data.get('segmentations', None)
        if sseg and segmentations is not None:
            if ssegkw is None:
                ssegkw = {
                    'alpha': 0.4,
                    'color': color,
                }
            image = segmentations.draw_on(image, **ssegkw)

        if boxes:
            image = self.boxes.draw_on(image, color=color, alpha=alpha,
                                       labels=labels, label_loc=label_loc,
                                       thickness=thickness)

        keypoints = self.data.get('keypoints', None)
        if kpts and keypoints is not None:
            # image = kwimage.ensure_float01(image)
            image = keypoints.draw_on(image, radius=radius, color=color)
            # kwimage.ensure_float01(image)

        image = dtype_fixer(image, copy=False)
        return image

    def _make_colors(self, color):
        """
        Handles special settings of color.

        If color == 'classes', then choose a distinct color for each category
        """
        # Draw each category as a different color
        if color == 'classes':
            import kwimage
            class_idxs = self.class_idxs
            if class_idxs is None:
                color = 'blue'
            else:
                classes = self.classes
                if classes is None:
                    classes = list(range(max(class_idxs) + 1))

                # TODO: allow specified color scheme
                backup_colors = iter(kwimage.Color.distinct(len(classes)))

                # Respect colors stored in classes if given
                if hasattr(classes, 'idx_to_node'):
                    cname_to_color = {
                        cid: cat.get('color', None)
                        for cid, cat in classes.cats.items()
                    }
                    cidx_to_color = [
                        cname_to_color[cname]
                        for cname in classes.idx_to_node
                    ]
                else:
                    cidx_to_color = [None] * len(classes)

                for cidx, color in enumerate(cidx_to_color):
                    if color is None:
                        cidx_to_color[cidx] = next(backup_colors)

                color = [cidx_to_color[cidx] for cidx in class_idxs]
        return color

    def _make_alpha(self, alpha):
        """
        Either passes through user specified alpha or chooses a sensible
        default
        """
        if alpha in ['score', 'scores']:
            alpha = np.sqrt(self.scores)
        else:
            if alpha is None or alpha is False:
                alpha = 1.0
            alpha = [float(alpha)] * self.num_boxes()
        return alpha

    def _make_labels(self, labels):
        """
        Either passes through user specified labels or chooses a sensible
        default
        """
        def _fixsore(s):
            return float('nan') if s is None else s

        if labels:
            if labels is True:
                parts = []
                if self.data.get('class_idxs', None) is not None:
                    parts.append('class')
                # Choose sensible default
                if self.data.get('scores', None) is not None:
                    parts.append('score')
                labels = '+'.join(parts)

            if isinstance(labels, six.string_types):
                if labels in ['class', 'class+score']:
                    if self.classes:
                        identifers = list(ub.take(self.classes, self.class_idxs))
                    else:
                        identifers = self.class_idxs
                if labels in ['class']:
                    labels = identifers
                elif labels in ['score']:
                    labels = ['{:.4f}'.format(_fixsore(score)) for score in self.scores]
                elif labels in ['class+score']:
                    labels = ['{} @ {:.4f}'.format(cid, _fixsore(score))
                              for cid, score in zip(identifers, self.scores)]
                else:
                    raise KeyError('unknown labels key {!r}'.format(labels))
        return labels


class _DetAlgoMixin:
    """
    Non critical methods for algorithmic manipulation of detections
    """

    def non_max_supression(self, thresh=0.0, perclass=False, impl='auto',
                           daq=False, device_id=None):
        """
        Find high scoring minimally overlapping detections

        Args:
            thresh (float): iou threshold between 0 and 1. A box is removed if
                it overlaps with a previously chosen box by more than this
                threshold. Higher values are are more permissive (more boxes
                are returned). A value of 0 means that returned boxes will have
                no overlap.
            perclass (bool): if True, works on a per-class basis
            impl (str): nms implementation to use
            daq (Bool | Dict): if False, uses reqgular nms, otherwise uses
                divide and conquor algorithm. If `daq` is a Dict, then
                it is used as the kwargs to `kwimage.daq_spatial_nms`

            device_id : try not to use. only used if impl is gpu

        Returns:
            ndarray[int]: indices of boxes to keep
        """
        import kwimage
        classes = self.class_idxs if perclass else None

        if len(self) <= 0:
            return []

        ltrb = self.boxes.to_ltrb().data
        scores = self.data.get('scores', None)
        if scores is None:
            scores = np.ones(len(self), dtype=np.float32)
        if daq:
            daqkw = {} if daq is True else daq.copy()
            daqkw['impl'] = daqkw.get('impl', impl)
            daqkw['stop_size'] = daqkw.get('stop_size', 2048)
            daqkw['max_depth'] = daqkw.get('max_depth', 12)
            daqkw['thresh'] = daqkw.get('thresh', thresh)
            if 'diameter' not in daqkw:
                if len(self.boxes) > 0:
                    daqkw['diameter'] = max(self.boxes.width.max(),
                                            self.boxes.height.max())
                else:
                    daqkw['diameter'] = 10  # hack

            keep = kwimage.daq_spatial_nms(ltrb, scores, device_id=device_id,
                                           **daqkw)
        else:
            keep = kwimage.non_max_supression(ltrb, scores, thresh=thresh,
                                              classes=classes, impl=impl,
                                              device_id=device_id)
        return keep

    def non_max_supress(self, thresh=0.0, perclass=False, impl='auto',
                        daq=False):
        """
        Convinience method. Like `non_max_supression`, but returns to supressed
        boxes instead of the indices to keep.
        """
        keep = self.non_max_supression(thresh=thresh, perclass=perclass,
                                       impl=impl, daq=daq)
        return self.take(keep)

    def rasterize(self, bg_size, input_dims, soften=1, tf_data_to_img=None,
                  img_dims=None, exclude=[]):
        """
        Ambiguous conversion from a Heatmap to a Detections object.

        SeeAlso:
            Heatmap.detect

        Returns:
            kwimage.Heatmap: raster-space detections.

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwimage.structs.detections import *  # NOQA
            >>> self, iminfo, sampler = Detections.demo()
            >>> image = iminfo['imdata'][:]
            >>> input_dims = iminfo['imdata'].shape[0:2]
            >>> bg_size = [100, 100]
            >>> heatmap = self.rasterize(bg_size, input_dims)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, pnum=(2, 2, 1))
            >>> heatmap.draw(invert=True)
            >>> kwplot.figure(fnum=1, pnum=(2, 2, 2))
            >>> kwplot.imshow(heatmap.draw_on(image))
            >>> kwplot.figure(fnum=1, pnum=(2, 1, 2))
            >>> kwplot.imshow(heatmap.draw_stacked())
        """
        import kwarray
        import skimage
        import kwimage
        classes = self.meta['classes']

        bg_idx = classes.index('background')
        fcn_target = _dets_to_fcmaps(
            self, bg_size=bg_size, input_dims=input_dims, bg_idx=bg_idx,
            soft=False, exclude=exclude)

        if tf_data_to_img is None:
            tf_data_to_img = skimage.transform.AffineTransform(
                scale=(1, 1), translation=(0, 0),
            )

        if img_dims is None:
            img_dims = np.array(input_dims)
        # print(fcn_target.keys())
        # print('fcn_target: ' + ub.repr2(ub.map_vals(lambda x: x.shape, fcn_target), nl=1))

        impl = kwarray.ArrayAPI.coerce(fcn_target['cidx'])

        # class_probs = nh.criterions.focal.one_hot_embedding(
        #     fcn_target['cidx'].reshape(-1),
        #     num_classes=len(classes), dim=1)

        class_idx = fcn_target['cidx']

        if 'class_probs' not in exclude:
            class_probs = kwarray.one_hot_embedding(
                class_idx, num_classes=len(classes), dim=0)

            if soften > 0:
                k = 31
                sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # opencv formula
                data = impl.contiguous(class_probs.T)
                import cv2
                cv2.GaussianBlur(data, (k, k), sigma, dst=data)
                class_probs = impl.contiguous(data.T)

            if soften > 1:
                class_probs = impl.softmax(class_probs, axis=0)

        dims = tuple(class_idx.shape)

        kw_heat = {
            'class_idx': class_idx,
            'classes': classes,
            'img_dims': img_dims,
            'tf_data_to_img': tf_data_to_img,
            'datakeys': ['kpts_ignore', 'class_idx'],
        }

        if 'class_probs' not in exclude:
            kw_heat['class_probs'] = class_probs

        if 'diameter' not in exclude:
            if 'size' in fcn_target:
                kw_heat['diameter'] = fcn_target['size'][[1, 0]]

        if 'offset' not in exclude:
            if 'dxdy' in fcn_target:
                kw_heat['offset'] = fcn_target['dxdy'][[1, 0]]

        if 'keypoints' not in exclude:
            if 'kpts' in fcn_target:
                kp_classes = self.meta['kp_classes']
                K = len(kp_classes)
                # TODO: add noise or do some bluring?
                kw_heat['keypoints'] = impl.view(fcn_target['kpts'], (2, K,) + dims)[[1, 0]]
                kw_heat['kpts_ignore'] = fcn_target['kpts_ignore']

        self = kwimage.Heatmap(**kw_heat)
        # print('self.data: ' + ub.repr2(ub.map_vals(lambda x: x.shape, self.data), nl=1))
        return self


class Detections(ub.NiceRepr, _DetAlgoMixin, _DetDrawMixin):
    """
    Container for holding and manipulating multiple detections.

    Attributes:
        data (Dict): dictionary containing corresponding lists. The length of
            each list is the number of detections. This contains the bounding
            boxes, confidence scores, and class indices. Details of the most
            common keys and types are as follows:

                boxes (kwimage.Boxes[ArrayLike]): multiple bounding boxes
                scores (ArrayLike): associated scores
                class_idxs (ArrayLike): associated class indices
                segmentations (ArrayLike): segmentations masks for each box,
                    members can be :class:`Mask` or :class:`MultiPolygon`.
                keypoints (ArrayLike): keypoints for each box. Members should
                    be :class:`Points`.

            Additional custom keys may be specified as long as (a) the values
            are array-like and the first axis corresponds to the standard data
            values and (b) are custom keys are listed in the `datakeys` kwargs
            when constructing the Detections.

        meta (Dict):
            This contains contextual information about the detections.  This
            includes the class names, which can be indexed into via the class
            indexes.

    Example:
        >>> import kwimage
        >>> dets = kwimage.Detections(
        >>>     # there are expected keys that do not need registration
        >>>     boxes=kwimage.Boxes.random(3),
        >>>     class_idxs=[0, 1, 1],
        >>>     classes=['a', 'b'],
        >>>     # custom data attrs must align with boxes
        >>>     myattr1=np.random.rand(3),
        >>>     myattr2=np.random.rand(3, 2, 8),
        >>>     # there are no restrictions on metadata
        >>>     mymeta='a custom metadata string',
        >>>     # Note that any key not in kwimage.Detections.__datakeys__ or
        >>>     # kwimage.Detections.__metakeys__ must be registered at the
        >>>     # time of construction.
        >>>     datakeys=['myattr1', 'myattr2'],
        >>>     metakeys=['mymeta'],
        >>>     checks=True,
        >>> )
        >>> print('dets = {}'.format(dets))
        dets = <Detections(3)>
    """
    # __slots__ = ('data', 'meta',)

    # Valid keys for the data dictionary
    # NOTE: I'm not sure its productive to restrict to a set of specified
    # properties. It might be better to allow detections to have arbitrary data
    # properties like: velocity, as long as they are array-like. However, I'm
    # not sure how to best structure the code to allow this so it is both clear
    # and efficient. Currently I've allowed the user to specify custom datakeys
    # and metakeys as kwargs, but that design might change.
    __datakeys__ = ['boxes', 'scores', 'class_idxs', 'probs', 'weights',
                    'keypoints', 'segmentations']

    # Valid keys for the meta dictionary
    __metakeys__ = ['classes']

    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None,
                 checks=True, **kwargs):
        """
        Construct a Detections object by either explicitly specifying the
        internal data and meta dictionary structures or by passing expected
        attribute names as kwargs.

        Args:
            data (Dict[str, ArrayLike]): explicitly specify the data dictionary
            meta (Dict[str, object]): explicitly specify the meta dictionary
            datakeys (List[str]): a list of custom attributes that should be
               considered as data (i.e. must be an array aligned with boxes).
            metakeys (List[str]): a list of custom attributes that should be
               considered as metadata (i.e. can be arbitrary).
            checks (bool, default=True): if True and arguments are passed by
                kwargs, then check / ensure that all types are compatible
            **kwargs:
                specify any key for the data or meta dictionaries.

        Notes:
            Custom data and metadata can be specified as long as you pass the
            names of these keys in the `datakeys` and/or `metakeys` kwargs.

            In the case where you specify a custom attribute as a list, it will
            "currently" (we may change this behavior in the future) be coerced
            into a numpy or torch array. If you want to store a generic Python
            list, wrap the custom list in a ``_generic.ObjectList``.

        Example:
            >>> # Coerce to numpy
            >>> import kwimage
            >>> dets = Detections(
            >>>     boxes=kwimage.Boxes.random(3).numpy(),
            >>>     class_idxs=[0, 1, 1],
            >>>     checks=True,
            >>> )
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> # Coerce to tensor
            >>> dets = Detections(
            >>>     boxes=kwimage.Boxes.random(3).tensor(),
            >>>     class_idxs=[0, 1, 1],
            >>>     checks=True,
            >>> )
            >>> # Error on incompatible types
            >>> import pytest
            >>> with pytest.raises(TypeError):
            >>>     dets = Detections(
            >>>         boxes=kwimage.Boxes.random(3).tensor(),
            >>>         scores=np.random.rand(3),
            >>>         class_idxs=[0, 1, 1],
            >>>         checks=True,
            >>>     )

        Example:
            >>> self = Detections.random(10)
            >>> other = Detections(self)
            >>> assert other.data == self.data
            >>> assert other.data is self.data, 'try not to copy unless necessary'

        """
        # Standardize input format
        if kwargs:
            if data or meta:
                raise ValueError('Cannot specify kwargs AND data/meta dicts')
            _datakeys = self.__datakeys__
            _metakeys = self.__metakeys__
            # Allow the user to specify custom data and meta keys
            if datakeys is not None:
                _datakeys = _datakeys + list(datakeys)
            if metakeys is not None:
                _metakeys = _metakeys + list(metakeys)
            # Perform input checks whenever kwargs is given
            data = {key: kwargs.pop(key) for key in _datakeys if key in kwargs}
            meta = {key: kwargs.pop(key) for key in _metakeys if key in kwargs}
            if kwargs:
                raise ValueError(
                    'Unknown kwargs: {}'.format(sorted(kwargs.keys())))

            if checks:
                import kwarray
                # Check to make sure all types in `data` are compatible
                ndarrays = []
                tensors = []
                other = []
                objlist = []

                ### Make it easier to specify keypoints and segmentations
                if 'segmentations' in data:
                    import kwimage
                    data['segmentations'] = kwimage.SegmentationList.coerce(
                        data['segmentations'])

                for k, v in data.items():
                    if v is None:
                        objlist.append(v)
                    elif _generic._isinstance2(v, _generic.ObjectList):
                        objlist.append(v)
                    elif _generic._isinstance2(v, _boxes.Boxes):
                        if v.is_numpy():
                            ndarrays.append(k)
                        else:
                            tensors.append(k)
                    elif isinstance(v, np.ndarray):
                        ndarrays.append(k)
                    elif torch is not None and isinstance(v, torch.Tensor):
                        tensors.append(k)
                    else:
                        other.append(k)

                if bool(ndarrays) and bool(tensors):
                    raise TypeError(
                        'Detections can hold numpy.ndarrays or torch.Tensors, '
                        'but not both')
                if tensors:
                    impl = kwarray.ArrayAPI.coerce('tensor')
                else:
                    impl = kwarray.ArrayAPI.coerce('numpy')
                for k in other:
                    data[k] = impl.asarray(data[k])

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
        return self.num_boxes()

    def __len__(self):
        return self.num_boxes()

    def copy(self):
        """
        Returns a deep copy of this Detections object
        """
        import copy
        return copy.deepcopy(self)

    @classmethod
    def coerce(cls, data=None, **kwargs):
        """
        The "try-anything to get what I want" constructor

        Args:
            data:
            **kwargs: currently boxes and cnames

        Example:
            >>> from kwimage.structs.detections import *  # NOQA
            >>> import kwimage
            >>> kwargs = dict(
            >>>     boxes=kwimage.Boxes.random(4),
            >>>     cnames=['a', 'b', 'c', 'c'],
            >>> )
            >>> data = {}
            >>> self = kwimage.Detections.coerce(data, **kwargs)
        """
        if data is None:
            data = {}
        if 'boxes' in kwargs:
            data['boxes'] = kwargs['boxes']

        cnames = kwargs.get('cnames', kwargs.get('class_names', kwargs.get('catnames', None)))
        if cnames is not None:
            if len(cnames) and isinstance(ub.peek(cnames), six.string_types):
                if 'classes' not in data:
                    data['classes'] = sorted(set(cnames))
                if 'class_idxs' not in data:
                    classes = data['classes']
                    data['class_idxs'] = list(map(classes.index, cnames))

        self = cls(**data)
        return self

    @classmethod
    def from_coco_annots(cls, anns, cats=None, classes=None, kp_classes=None,
                         shape=None, dset=None):
        """
        Create a Detections object from a list of coco-like annotations.

        Args:
            anns (List[Dict]): list of coco-like annotation objects

            dset (CocoDataset): if specified, cats, classes, and kp_classes
                can are ignored.

            cats (List[Dict]): coco-format category information.
                Used only if `dset` is not specified.

            classes (ndsampler.CategoryTree): category tree with coco class
                info. Used only if `dset` is not specified.

            kp_classes (ndsampler.CategoryTree): keypoint category tree with
                coco keypoint class info. Used only if `dset` is not specified.

            shape (tuple): shape of parent image

        Returns:
            Detections: a detections object

        Example:
            >>> from kwimage.structs.detections import *  # NOQA
            >>> # xdoctest: +REQUIRES(--module:ndsampler)
            >>> anns = [{
            >>>     'id': 0,
            >>>     'image_id': 1,
            >>>     'category_id': 2,
            >>>     'bbox': [2, 3, 10, 10],
            >>>     'keypoints': [4.5, 4.5, 2],
            >>>     'segmentation': {
            >>>         'counts': '_11a04M2O0O20N101N3L_5',
            >>>         'size': [20, 20],
            >>>     },
            >>> }]
            >>> dataset = {
            >>>     'images': [],
            >>>     'annotations': [],
            >>>     'categories': [
            >>>         {'id': 0, 'name': 'background'},
            >>>         {'id': 2, 'name': 'class1', 'keypoints': ['spot']}
            >>>     ]
            >>> }
            >>> #import ndsampler
            >>> #dset = ndsampler.CocoDataset(dataset)
            >>> cats = dataset['categories']
            >>> dets = Detections.from_coco_annots(anns, cats)

        Example:
            >>> # xdoctest: +REQUIRES(--module:ndsampler)
            >>> # Test case with no category information
            >>> from kwimage.structs.detections import *  # NOQA
            >>> anns = [{
            >>>     'id': 0,
            >>>     'image_id': 1,
            >>>     'category_id': None,
            >>>     'bbox': [2, 3, 10, 10],
            >>>     'prob': [.1, .9],
            >>> }]
            >>> cats = [
            >>>     {'id': 0, 'name': 'background'},
            >>>     {'id': 2, 'name': 'class1'}
            >>> ]
            >>> dets = Detections.from_coco_annots(anns, cats)

        Example:
            >>> import kwimage
            >>> # xdoctest: +REQUIRES(--module:ndsampler)
            >>> import ndsampler
            >>> sampler = ndsampler.CocoSampler.demo('photos')
            >>> iminfo, anns = sampler.load_image_with_annots(1)
            >>> shape = iminfo['imdata'].shape[0:2]
            >>> kp_classes = sampler.dset.keypoint_categories()
            >>> dets = kwimage.Detections.from_coco_annots(
            >>>     anns, sampler.dset.dataset['categories'], sampler.catgraph,
            >>>     kp_classes, shape=shape)
        """
        import kwimage
        cnames = None
        if dset is not None:
            try:
                classes = dset.object_categories()
            except Exception:
                pass
            cats = dset.dataset['categories']
            try:
                kp_classes = dset.keypoint_categories()
            except Exception:
                pass
                # kp_classes = None
        else:
            if cats is None:
                cnames = []
                for ann in anns:
                    if 'category_name' in ann:
                        cnames.append(ann['category_name'])
                    else:
                        raise Exception('Specify dset or cats or category_name in each annotation')
                if classes is None:
                    classes = sorted(set(cnames))
                assert set(cnames).issubset(set(classes))

                # make dummy cats
                cats = [{'name': name, 'id': cid}
                        for cid, name in enumerate(classes, start=1) ]

        if classes is None:
            classes = list(ub.oset([cat['name'] for cat in cats]))

        if cnames is None:
            cids = [ann['category_id'] for ann in anns]
            cid_to_cat = {c['id']: c for c in cats}  # Hack
            cnames = [None if cid is None else cid_to_cat[cid]['name']
                      for cid in cids]

        xywh = np.array([ann['bbox'] for ann in anns], dtype=np.float32)
        boxes = kwimage.Boxes(xywh, 'xywh')
        try:
            class_idxs = [classes.index(cname) for cname in cnames]
        except (KeyError, ValueError):
            class_idxs = [None if cname is None else classes.index(cname)
                          for cname in cnames]

        dets = Detections(
            boxes=boxes,
            class_idxs=np.array(class_idxs),
            classes=classes,
        )

        if len(anns):
            if 'score' in anns[0]:
                dets.data['scores'] = np.array([ann.get('score', np.nan) for ann in anns])

            if 'prob' in anns[0]:
                dets.data['probs'] = np.array([ann.get('prob', np.nan) for ann in anns])

            if 'weight' in anns[0]:
                dets.data['weights'] = np.array([ann.get('weight', np.nan) for ann in anns])

        if True:
            ss = [ann.get('segmentation', None) for ann in anns]
            masks = [
                None if s is None else
                kwimage.MultiPolygon.coerce(s, dims=shape)
                for s in ss
            ]
            dets.data['segmentations'] = kwimage.PolygonList(masks)

        if True:
            name_to_cat = {c['name']: c for c in cats}
            def _lookup_kp_class_idxs(cid):
                kpnames = None
                while kpnames is None:
                    cat = cid_to_cat[cid]
                    parent = cat.get('supercategory', None)
                    if 'keypoints' in cat:
                        kpnames = cat['keypoints']
                    elif parent is not None:
                        cid = name_to_cat[cat['supercategory']]['id']
                    else:
                        raise KeyError(cid)
                kpcidxs = [kp_classes.index(n) for n in kpnames]
                return kpcidxs
            kpts = []
            for ann in anns:
                k = ann.get('keypoints', None)
                if k is None:
                    kpts.append(k)
                elif len(k) == 0:
                    kpcidxs = []
                else:
                    kpcidxs = None
                    # TODO: correctly handle newstyle keypoints
                    if dset is not None:
                        pass
                    kpcidxs = None
                    if not (isinstance(k, list) and len(k) and isinstance(ub.peek(k), dict)):
                        # oldstyle
                        if kp_classes is not None:
                            # These are only needed for old-style coco
                            kpcidxs = _lookup_kp_class_idxs(ann['category_id'])

                    pts = kwimage.Points.from_coco(
                        k, class_idxs=kpcidxs, classes=kp_classes)
                    kpts.append(pts)
            dets.data['keypoints'] = kwimage.PointsList(kpts)

            if kp_classes is not None:
                dets.data['keypoints'].meta['classes'] = kp_classes
                dets.meta['kp_classes'] = kp_classes
        return dets

    def to_coco(self, cname_to_cat=None, style='orig', image_id=None, dset=None):
        """
        Converts this set of detections into coco-like annotation dictionaries.

        Notes:
            Not all aspects of the MS-COCO format can be accurately
            represented, so some liberties are taken. The MS-COCO standard
            defines that annotations should specifiy a category_id field, but
            in some cases this information is not available so we will populate
            a 'category_name' field if possible and in the worst case fall back
            to 'category_index'.

            Additionally, detections may contain additional information beyond
            the MS-COCO standard, and this information (e.g. weight, prob,
            score) is added as forign fields.

        Args:
            cname_to_cat: currently ignored.

            style (str, default='orig'): either 'orig' (for the original coco
                format) or 'new' for the more general kwcoco-style coco
                format.

            image_id (int, default=None):
                if specified, populates the image_id field of each image

            dset (CocoDataset, default=None):
                if specified, attempts to populate the category_id field
                to be compatible with this coco dataset.

        Yields:
            dict: coco-like annotation structures

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from kwimage.structs.detections import *
            >>> self = Detections.demo()[0]
            >>> cname_to_cat = None
            >>> list(self.to_coco())
        """
        import kwarray
        to_collate = {}
        if 'boxes' in self.data:
            to_collate['bbox'] = list(self.data['boxes'].to_coco(style=style))

        if 'class_idxs' in self.data:
            if 'classes' in self.meta:
                classes = self.meta['classes']
                catnames = [classes[cidx] for cidx in self.class_idxs]
                if cname_to_cat is not None:
                    pass
                if dset is not None:
                    cids = [dset._resolve_to_cat(c)['id'] for c in catnames]
                    to_collate['category_id'] = cids
                else:
                    to_collate['category_name'] = catnames
            else:
                if dset is not None:
                    raise NotImplementedError(
                        'Passed a dset to resolve category id, but this '
                        'detection object has no classes meta attribute')
                to_collate['category_index'] = kwarray.ArrayAPI.tolist(
                    self.data['class_idxs'])

        if 'keypoints' in self.data:
            to_collate['keypoints'] = list(self.data['keypoints'].to_coco(
                style=style))

        if 'segmentations' in self.data:
            to_collate['segmentation'] = list(self.data['segmentations'].to_coco(
                style=style))

        if 'scores' in self.data:
            to_collate['score'] = kwarray.ArrayAPI.tolist(self.data['scores'])

        if 'weights' in self.data:
            to_collate['weight'] = kwarray.ArrayAPI.tolist(self.data['weights'])

        if 'probs' in self.data:
            to_collate['prob'] = kwarray.ArrayAPI.tolist(self.data['probs'])

        if image_id is not None:
            to_collate['image_id'] = [image_id] * len(self)

        keys = list(to_collate.keys())
        for item_vals in zip(*to_collate.values()):
            ann = ub.dzip(keys, item_vals)
            yield ann

    # --- Data Properties ---

    @property
    def boxes(self):
        return self.data['boxes']

    @property
    def class_idxs(self):
        return self.data['class_idxs']

    @property
    def scores(self):
        """ typically only populated for predicted detections """
        return self.data['scores']

    @property
    def probs(self):
        """ typically only populated for predicted detections """
        return self.data['probs']

    @property
    def weights(self):
        """ typically only populated for groundtruth detections """
        return self.data['weights']

    # --- Meta Properties ---

    @property
    def classes(self):
        return self.meta.get('classes', None)

    def num_boxes(self):
        return len(self.boxes)

    # --- Modifiers ---

    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        """
        Spatially warp the detections.

        Example:
            >>> import skimage
            >>> transform = skimage.transform.AffineTransform(scale=(2, 3), translation=(4, 5))
            >>> self = Detections.random(2)
            >>> new = self.warp(transform)
            >>> assert new.boxes == self.boxes.warp(transform)
            >>> assert new != self
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['boxes'] = new.data['boxes'].warp(transform,
                                                   input_dims=input_dims,
                                                   inplace=inplace)
        if 'keypoints' in new.data:
            new.data['keypoints'] = new.data['keypoints'].warp(
                transform, input_dims=input_dims, output_dims=output_dims,
                inplace=inplace)
        if 'segmentations' in new.data:
            new.data['segmentations'] = new.data['segmentations'].warp(
                transform, input_dims=input_dims, output_dims=output_dims,
                inplace=inplace)
        return new

    @profile
    def scale(self, factor, output_dims=None, inplace=False):
        """
        Spatially warp the detections.

        Example:
            >>> import skimage
            >>> transform = skimage.transform.AffineTransform(scale=(2, 3), translation=(4, 5))
            >>> self = Detections.random(2)
            >>> new = self.warp(transform)
            >>> assert new.boxes == self.boxes.warp(transform)
            >>> assert new != self
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['boxes'] = new.data['boxes'].scale(factor, inplace=inplace)
        if 'keypoints' in new.data:
            new.data['keypoints'] = new.data['keypoints'].scale(
                factor, output_dims=output_dims, inplace=inplace)
        if 'segmentations' in new.data:
            new.data['segmentations'] = new.data['segmentations'].scale(
                factor, output_dims=output_dims, inplace=inplace)
        return new

    @profile
    def translate(self, offset, output_dims=None, inplace=False):
        """
        Spatially warp the detections.

        Example:
            >>> import skimage
            >>> self = Detections.random(2)
            >>> new = self.translate(10)
        """
        new = self if inplace else self.__class__(self.data.copy(), self.meta)
        new.data['boxes'] = new.data['boxes'].translate(offset, inplace=inplace)
        if 'keypoints' in new.data:
            new.data['keypoints'] = new.data['keypoints'].translate(
                offset, output_dims=output_dims)
        if 'segmentations' in new.data:
            new.data['segmentations'] = new.data['segmentations'].translate(
                offset, output_dims=output_dims)
        return new

    @classmethod
    def concatenate(cls, dets):
        """
        Args:
            boxes (Sequence[Detections]): list of detections to concatenate

        Returns:
            Detections: stacked detections

        Example:
            >>> self = Detections.random(2)
            >>> other = Detections.random(3)
            >>> dets = [self, other]
            >>> new = Detections.concatenate(dets)
            >>> assert new.num_boxes() == 5

            >>> self = Detections.random(2, segmentations=True)
            >>> other = Detections.random(3, segmentations=True)
            >>> dets = [self, other]
            >>> new = Detections.concatenate(dets)
            >>> assert new.num_boxes() == 5
        """
        if len(dets) == 0:
            raise ValueError('need at least one detection to concatenate')
        newdata = {}
        first = dets[0]
        for key in first.data.keys():
            if first.data[key] is None:
                newdata[key] = None
            else:
                try:
                    tocat = [d.data[key] for d in dets]
                    try:
                        # Use class concatenate if it exists,
                        cat = tocat[0].__class__.concatenate
                    except AttributeError:
                        # otherwise use numpy/torch
                        cat = _boxes._cat
                    newdata[key] = cat(tocat, axis=0)
                except Exception:
                    msg = ('Error when trying to concat {}'.format(key))
                    print(msg)
                    raise

        newmeta = dets[0].meta
        new = cls(newdata, newmeta)
        return new

    def argsort(self, reverse=True):
        """
        Sorts detection indices by descending (or ascending) scores

        Returns:
            ndarray[int]: sorted indices
        """
        sortx = self.scores.argsort()
        if reverse:
            if torch is not None and torch.is_tensor(sortx):
                sortx = torch.flip(sortx, dims=(0,))
            else:
                sortx = sortx[::-1]
        return sortx

    def sort(self, reverse=True):
        """
        Sorts detections by descending (or ascending) scores

        Returns:
            kwimage.structs.Detections: sorted copy of self
        """
        sortx = self.argsort(reverse=reverse)
        return self.take(sortx)

    def compress(self, flags, axis=0):
        """
        Returns a subset where corresponding locations are True.

        Args:
            flags (ndarray[bool]): mask marking selected items

        Returns:
            kwimage.structs.Detections: subset of self

        CommandLine:
            xdoctest -m kwimage.structs.detections Detections.compress

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import kwimage
            >>> dets = kwimage.Detections.random(keypoints='dense')
            >>> flags = np.random.rand(len(dets)) > 0.5
            >>> subset = dets.compress(flags)
            >>> assert len(subset) == flags.sum()
            >>> subset = dets.tensor().compress(flags)
            >>> assert len(subset) == flags.sum()
        """
        if flags is Ellipsis:
            return self

        if len(flags) != len(self):
            raise IndexError('compress must get a flag for every item')

        if self.is_tensor():
            if isinstance(flags, np.ndarray):
                if flags.dtype.kind == 'b':
                    flags = flags.astype(np.uint8)
            if isinstance(flags, torch.Tensor):
                if _TORCH_HAS_BOOL_COMP:
                    if flags.dtype != torch.bool:
                        flags = flags.bool()
                else:
                    if flags.dtype != torch.uint8:
                        flags = flags.byte()
                if flags.device != flags.device:
                    flags = flags.to(self.device)
            else:
                if _TORCH_HAS_BOOL_COMP:
                    flags = torch.BoolTensor(flags).to(self.device)
                else:
                    flags = torch.ByteTensor(flags).to(self.device)
        newdata = {k: _generic._safe_compress(v, flags, axis)
                   for k, v in self.data.items()}
        return self.__class__(newdata, self.meta)

    def take(self, indices, axis=0):
        """
        Returns a subset specified by indices

        Args:
            indices (ndarray[int]): indices to select

        Returns:
            kwimage.structs.Detections: subset of self

        Example:
            >>> import kwimage
            >>> dets = kwimage.Detections(boxes=kwimage.Boxes.random(10))
            >>> subset = dets.take([2, 3, 5, 7])
            >>> assert len(subset) == 4
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> subset = dets.tensor().take([2, 3, 5, 7])
            >>> assert len(subset) == 4
        """
        if self.is_tensor():
            indices = torch.LongTensor(indices).to(self.device)
        newdata = {k: _generic._safe_take(v, indices, axis)
                   for k, v in self.data.items()}
        return self.__class__(newdata, self.meta)

    def __getitem__(self, index):
        """
        Fancy slicing / subset / indexing.

        Note: scalar indices are always coerced into index lists of length 1.

        Example:
            >>> import kwimage
            >>> import kwarray
            >>> dets = kwimage.Detections(boxes=kwimage.Boxes.random(10))
            >>> indices = [2, 3, 5, 7]
            >>> flags = kwarray.boolmask(indices, len(dets))
            >>> assert dets[flags].data == dets[indices].data
        """
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))
        if ub.iterable(index):
            import kwarray
            impl = kwarray.ArrayAPI.coerce('numpy')
            indices = impl.asarray(index)
        else:
            indices = np.array([index])
        if indices.dtype.kind == 'b':
            return self.compress(indices)
        else:
            return self.take(indices)

    @property
    def device(self):
        """ If the backend is torch returns the data device, otherwise None """
        return self.boxes.device

    def is_tensor(self):
        """ is the backend fueled by torch? """
        return self.boxes.is_tensor()

    def is_numpy(self):
        """ is the backend fueled by numpy? """
        return self.boxes.is_numpy()

    def numpy(self):
        """
        Converts tensors to numpy. Does not change memory if possible.

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> self = Detections.random(3).tensor()
            >>> newself = self.numpy()
            >>> self.scores[0] = 0
            >>> assert newself.scores[0] == 0
            >>> self.scores[0] = 1
            >>> assert self.scores[0] == 1
            >>> self.numpy().numpy()
        """
        newdata = {}
        for key, val in self.data.items():
            if val is None:
                newval = val
            else:
                if torch is not None and torch.is_tensor(val):
                    newval = val.data.cpu().numpy()
                elif hasattr(val, 'numpy'):
                    newval = val.numpy()
                else:
                    newval = val
            newdata[key] = newval
        newself = self.__class__(newdata, self.meta)
        return newself

    @property
    def dtype(self):
        dtypes = set()
        for key, val in self.data.items():
            if val is not None:
                try:
                    child_dtype = val.dtype
                    if isinstance(child_dtype, set):
                        dtypes.update(child_dtype)
                    else:
                        dtypes.add(child_dtype)
                except AttributeError:
                    dtypes.add('unknown-for-{}'.format(type(val)))
        if len(dtypes) == 1:
            return ub.peek(dtypes)
        else:
            return dtypes

    def tensor(self, device=ub.NoParam):
        """
        Converts numpy to tensors. Does not change memory if possible.

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> from kwimage.structs.detections import *
            >>> self = Detections.random(3)
            >>> newself = self.tensor()
            >>> self.scores[0] = 0
            >>> assert newself.scores[0] == 0
            >>> self.scores[0] = 1
            >>> assert self.scores[0] == 1
            >>> self.tensor().tensor()
        """
        newdata = {}
        for key, val in self.data.items():
            if val is None:
                newval = val
            elif hasattr(val, 'tensor'):
                newval = val.tensor(device)
            else:
                if torch is not None and torch.is_tensor(val):
                    newval = val
                else:
                    newval = torch.from_numpy(val)
                if device is not ub.NoParam:
                    newval = newval.to(device)
            newdata[key] = newval
        newself = self.__class__(newdata, self.meta)
        return newself

    # --- Non-core methods ----

    @classmethod
    def demo(Detections):
        import ndsampler
        sampler = ndsampler.CocoSampler.demo('photos')
        iminfo, anns = sampler.load_image_with_annots(1)
        input_dims = iminfo['imdata'].shape[0:2]
        kp_classes = sampler.dset.keypoint_categories()
        self = Detections.from_coco_annots(
            anns, sampler.dset.dataset['categories'],
            sampler.catgraph, kp_classes, shape=input_dims)

        # TODO: should this extra info belong in the metadata field?
        return self, iminfo, sampler

    @classmethod
    def random(cls, num=10, scale=1.0, classes=3, keypoints=False,
               segmentations=False, tensor=False, rng=None):
        """
        Creates dummy data, suitable for use in tests and benchmarks

        Args:
            num (int): number of boxes
            scale (float | tuple, default=1.0): bounding image size
            classes (int | Sequence): list of class labels or number of classes
            keypoints (bool, default=False):
                if True include random keypoints for each box.
            segmentations (bool, default=False):
                if True include random segmentations for each box.
            tensor (bool, default=False): determines backend.
                DEPRECATED.  Call tensor on resulting object instead.
            rng (np.random.RandomState): random state

        Example:
            >>> import kwimage
            >>> dets = kwimage.Detections.random(keypoints='jagged')
            >>> dets.data['keypoints'].data[0].data
            >>> dets.data['keypoints'].meta
            >>> dets = kwimage.Detections.random(keypoints='dense')
            >>> dets = kwimage.Detections.random(keypoints='dense', segmentations=True).scale(1000)
            >>> # xdoctest:+REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dets.draw(setlim=True)

        Example:
            >>> import kwimage
            >>> dets = kwimage.Detections.random(
            >>>     keypoints='jagged', segmentations=True, rng=0).scale(1000)
            >>> print('dets = {}'.format(dets))
            dets = <Detections(10)>
            >>> dets.data['boxes'].quantize(inplace=True)
            >>> print('dets.data = {}'.format(ub.repr2(
            >>>     dets.data, nl=1, with_dtype=False, strvals=True)))
            dets.data = {
                'boxes': <Boxes(xywh,
                             array([[548, 544,  55, 172],
                                    [423, 645,  15, 247],
                                    [791, 383, 173, 146],
                                    [ 71,  87, 498, 839],
                                    [ 20, 832, 759,  39],
                                    [461, 780, 518,  20],
                                    [118, 639,  26, 306],
                                    [264, 414, 258, 361],
                                    [ 18, 568, 439,  50],
                                    [612, 616, 332,  66]], dtype=int32))>,
                'class_idxs': [1, 2, 0, 0, 2, 0, 0, 0, 0, 0],
                'keypoints': <PointsList(n=10)>,
                'scores': [0.3595079 , 0.43703195, 0.6976312 , 0.06022547, 0.66676672, 0.67063787,0.21038256, 0.1289263 , 0.31542835, 0.36371077],
                'segmentations': <SegmentationList(n=10)>,
            }
            >>> # xdoctest:+REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> dets.draw(setlim=True)

        Example:
            >>> # Boxes position/shape within 0-1 space should be uniform.
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> fig.gca().set_xlim(0, 128)
            >>> fig.gca().set_ylim(0, 128)
            >>> import kwimage
            >>> kwimage.Detections.random(num=10, segmentations=True).scale(128).draw()
        """
        import kwimage
        import kwarray
        rng = kwarray.ensure_rng(rng)
        boxes = kwimage.Boxes.random(num=num, rng=rng)
        if isinstance(classes, int):
            num_classes = classes
            classes = ['class_{}'.format(c) for c in range(classes)]
            # hack: ensure that we have a background class
            classes.append('background')
        else:
            num_classes = len(classes)
        scores = rng.rand(len(boxes))
        class_idxs = rng.randint(0, num_classes, size=len(boxes))
        self = cls(boxes=boxes, scores=scores, class_idxs=class_idxs,
                   classes=classes)
        self.meta['classes'] = classes

        if keypoints is True:
            keypoints = 'jagged'

        if segmentations:
            sseg_list = []
            for xywh in self.boxes.to_xywh().data:
                box_scale = xywh[2:]
                box_offset = xywh[0:2]
                sseg = kwimage.MultiPolygon.random(n=1, tight=True, rng=rng)
                sseg = sseg.scale(box_scale).translate(box_offset)
                sseg_list.append(sseg)
            self.data['segmentations'] = kwimage.SegmentationList.coerce(sseg_list)

        if isinstance(keypoints, six.string_types):
            kp_classes = [1, 2, 3, 4]
            self.meta['kp_classes'] = kp_classes
            if keypoints == 'jagged':
                kpts_list = kwimage.PointsList([
                    kwimage.Points.random(
                        num=rng.randint(len(kp_classes)),
                        classes=kp_classes, rng=rng,
                    )
                    for _ in range(len(boxes))
                ])
                kpts_list.meta['classes'] = kp_classes
                self.data['keypoints'] = kpts_list
            elif keypoints == 'dense':
                keypoints = kwimage.Points.random(
                    num=(len(boxes), len(kp_classes)), rng=rng,
                    classes=kp_classes,)
                self.data['keypoints'] = keypoints

        self = self.scale(scale)

        if tensor:
            self = self.tensor()

        return self


def _dets_to_fcmaps(dets, bg_size, input_dims, bg_idx=0, pmin=0.6, pmax=1.0,
                    soft=True, exclude=[]):
    """
    Construct semantic segmentation detection targets from annotations in
    dictionary format.

    Rasterize detections.

    Args:
        dets (kwimage.Detections):
        bg_size (tuple): size (W, H) to predict for backgrounds
        input_dims (tuple): window H, W

    Returns:
        dict: with keys
            size : 2D ndarray containing the W,H of the object
            dxdy : 2D ndarray containing the x,y offset of the object
            cidx : 2D ndarray containing the class index of the object

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(_dets_to_fcmaps))

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from kwimage.structs.detections import *  # NOQA
        >>> from kwimage.structs.detections import _dets_to_fcmaps
        >>> import kwimage
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo('photos')
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
        fcn_target: {
            'cidx': (512, 512),
            'class_probs': (10, 512, 512),
            'dxdy': (2, 512, 512),
            'kpts': (2, 7, 512, 512),
            'kpts_ignore': (7, 512, 512),
            'size': (2, 512, 512),
        }
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

        raster = (kwimage.overlay_alpha_layers(_vizmask(kpts_mask[:, 5]) + [image], keepalpha=False) * 255).astype(np.uint8)
        kwplot.imshow(raster, pnum=(1, 3, 2), fnum=1)
        raster = (kwimage.overlay_alpha_layers(_vizmask(kpts_mask[:, 6]) + [image], keepalpha=False) * 255).astype(np.uint8)
        kwplot.imshow(raster, pnum=(1, 3, 3), fnum=1)
        raster = (kwimage.overlay_alpha_layers(_vizmask(dxdy_mask) + [image], keepalpha=False) * 255).astype(np.uint8)
        raster = dets.draw_on(raster, labels=True, alpha=None)
        kwplot.imshow(raster, pnum=(1, 3, 1), fnum=1)
        raster = kwimage.overlay_alpha_layers(
            [vecmask, orimask, image], keepalpha=False)
        raster = dets.draw_on((raster * 255).astype(np.uint8),
                              labels=True, alpha=None)
        kwplot.imshow(raster)
        kwplot.show_if_requested()
    """
    import cv2
    # In soft mode we made a one-channel segmentation target mask
    cidx_mask = np.full(input_dims, dtype=np.int32, fill_value=bg_idx)

    if 'class_probs' not in exclude:
        if soft:
            # In soft mode we add per-class channel probability blips
            num_obj_classes = len(dets.classes)
            cidx_probs = np.full((num_obj_classes,) + tuple(input_dims),
                                 dtype=np.float32, fill_value=0)

    if 'diameter' not in exclude:
        size_mask = np.empty((2,) + tuple(input_dims), dtype=np.float32)
        size_mask[:] = np.array(bg_size)[:, None, None]

    if 'offset' not in exclude:
        dxdy_mask = np.zeros((2,) + tuple(input_dims), dtype=np.float32)

    dets = dets.numpy()

    cxywh = dets.boxes.to_cxywh().data
    class_idxs = dets.class_idxs
    import kwimage

    if 'segmentations' in dets.data:
        sseg_list = [None if p is None else p.to_mask(input_dims)
                     for p in dets.data['segmentations']]
    else:
        sseg_list = [None] * len(dets)

    kpts_mask = None
    if 'keypoints' in dets.data and 'keypoints' not in exclude:
        kp_classes = None
        if 'classes' in dets.data['keypoints'].meta:
            kp_classes = dets.data['keypoints'].meta['classes']
        else:
            for kp in dets.data['keypoints']:
                if kp is not None and 'classes' in kp.meta:
                    kp_classes = kp.meta['classes']
                    break

        if kp_classes is not None:
            num_kp_classes = len(kp_classes)
            kpts_mask = np.zeros((2, num_kp_classes) + tuple(input_dims),
                                 dtype=np.float32)

        pts_list = dets.data['keypoints'].data
        for pts in pts_list:
            if pts is not None:
                pass

        kpts_ignore_mask = np.ones((num_kp_classes,) + tuple(input_dims),
                                   dtype=np.float32)
    else:
        pts_list = [None] * len(dets)

    # Overlay smaller classes on top of larger ones
    if len(cxywh):
        area = cxywh[..., 2] * cxywh[..., 2]
    else:
        area = []
    sortx = np.argsort(area)[::-1]
    cxywh = cxywh[sortx]
    class_idxs = class_idxs[sortx]
    pts_list = list(ub.take(pts_list, sortx))
    sseg_list = list(ub.take(sseg_list, sortx))

    def iround(x):
        return int(round(x))

    H, W = input_dims
    xcoord, ycoord = np.meshgrid(np.arange(W), np.arange(H))

    for box, cidx, sseg_mask, pts in zip(cxywh, class_idxs, sseg_list, pts_list):
        (cx, cy, w, h) = box
        center = (iround(cx), iround(cy))
        # Adjust so smaller objects get more pixels
        wf = min(1, (w / 64))
        hf = min(1, (h / 64))
        # wf = min(1, (w / W))
        # hf = min(1, (h / H))
        wf = (1 - wf) * pmax + wf * pmin
        hf = (1 - hf) * pmax + hf * pmin
        half_w = iround(wf * w / 2 + 1)
        half_h = iround(hf * h / 2 + 1)
        axes = (half_w, half_h)

        if sseg_mask is None:
            mask = np.zeros_like(cidx_mask, dtype=np.uint8)
            mask = cv2.ellipse(mask, center, axes, angle=0.0,
                               startAngle=0.0, endAngle=360.0, color=1,
                               thickness=-1).astype(np.bool)
        else:
            mask = sseg_mask.to_c_mask().data.astype(np.bool)
        # class index
        cidx_mask[mask] = int(cidx)
        if 'class_probs' not in exclude:
            if soft:
                blip = kwimage.gaussian_patch((half_h * 2, half_w * 2))
                blip = blip / blip.max()
                subindex = (slice(cy - half_h, cy + half_h),
                            slice(cx - half_w, cx + half_w))
                kwimage.subpixel_maximum(cidx_probs[cidx], blip, subindex)

        # object size
        if 'diameter' not in exclude:
            size_mask[0][mask] = float(w)
            size_mask[1][mask] = float(h)

            assert np.all(size_mask[0][mask] == float(w))

        # object offset
        if 'offset' not in exclude:
            dx = cx - xcoord[mask]
            dy = cy - ycoord[mask]
            dxdy_mask[0][mask] = dx
            dxdy_mask[1][mask] = dy

        if kpts_mask is not None:
            if 'keypoints' not in exclude:
                if pts is not None:
                    # Keypoint offsets
                    _xys = pts.data['xy'].data
                    if len(_xys) > 0:
                        _cidxs = pts.data['class_idxs']
                        if _cidxs is None:
                            raise ValueError(
                                'cannot rasterize keypoints with undefined categories')
                        for xy, kp_cidx in zip(_xys, _cidxs):
                            if kp_cidx < 0:
                                import warnings
                                warnings.warn('Cannot rasterize keypoints with unknown classes')
                            else:
                                kp_x, kp_y = xy
                                kp_dx = kp_x - xcoord[mask]
                                kp_dy = kp_y - ycoord[mask]
                                kpts_mask[0, kp_cidx][mask] = kp_dx
                                kpts_mask[1, kp_cidx][mask] = kp_dy
                                kpts_ignore_mask[kp_cidx][mask] = 0

    fcn_target = {
        'cidx': cidx_mask,
    }
    if 'diameter' not in exclude:
        fcn_target['size'] = size_mask

    if 'offset' not in exclude:
        fcn_target['dxdy'] = dxdy_mask

    if 'class_probs' not in exclude:
        if soft:
            nonbg_idxs = sorted(set(range(num_obj_classes)) - {bg_idx})
            cidx_probs[bg_idx] = 1 - cidx_probs[nonbg_idxs].sum(axis=0)
            fcn_target['class_probs'] = cidx_probs

    if kpts_mask is not None:
        if 'keypoints' not in exclude:
            fcn_target['kpts'] = kpts_mask
            fcn_target['kpts_ignore'] = kpts_ignore_mask
    else:
        if 'keypoints' in dets.data:
            if any(kp is not None for kp in dets.data['keypoints']):
                raise AssertionError(
                    'dets had keypoints, but we didnt encode them, were the kp classes missing?')

    return fcn_target


class _UnitDoctTests:
    """
    Hacking in unit tests as doctests the file itself so it is easy to move to
    kwannot when I finally get around to that.
    """

    def _test_foreign_keys_compress():
        """
        A detections object should be able to maintain foreign keys through
        compress operations.

        Example:
            >>> from kwimage.structs.detections import _UnitDoctTests
            >>> from kwimage.structs.detections import _generic
            >>> _UnitDoctTests._test_foreign_keys_compress()
        """
        import kwimage
        n = 5
        dets = kwimage.Detections.random(num=n)
        flags = dets.scores > np.median(dets.scores)

        # Test normal compress
        reduced = dets.compress(flags)
        m = len(reduced)

        # Test case with None attribute
        dets2 = kwimage.Detections(**{
            'boxes': dets.data['boxes'],
            'custom': None,
            'datakeys': ['custom'],
        })
        reduced2 = dets2.compress(flags)
        assert dets2.data['custom'] is None, 'should be able to specify None value'
        assert reduced2.data['custom'] is None, 'should be able to specify None value'

        # Test case with _generic.ObjectList[None] attribute
        dets3 = kwimage.Detections(**{
            'boxes': dets.data['boxes'],
            'custom': _generic.ObjectList([None] * n),
            'datakeys': ['custom'],
        })
        reduced3 = dets3.compress(flags)
        assert dets3.data['custom'].data == [None] * n, 'should be able to specify ObjectList[None] value'
        assert reduced3.data['custom'].data == [None] * m, 'should be able to specify ObjectList[None] value'
        assert len(reduced3.data['custom']) == m, 'compress failed'

        # NOTE: We expect Lists to always be coreced to arrays
        # Test case with List[None] attribute
        dets4 = kwimage.Detections(**{
            'boxes': dets.data['boxes'],
            'custom': [None] * n,
            'datakeys': ['custom'],
        })
        reduced4 = dets4.compress(flags)
        assert dets4.data['custom'].dtype.kind == 'O', (
            'we currently expect list to be coerced (may change in the future)')
        assert reduced4.data['custom'].dtype.kind == 'O', (
            'we currently expect list to be coerced (may change in the future)')
        assert len(reduced4.data['custom']) == m, 'compress failed'


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.structs.detections
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
