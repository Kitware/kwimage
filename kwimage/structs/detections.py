# -*- coding: utf-8 -*-
"""
Structure for efficient access and modification of bounding boxes with
associated scores and class labels. Builds on top of the `kwimage.Boxes`
structure.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import torch
import numpy as np
import ubelt as ub
from kwimage.structs import boxes as _boxes
from kwimage.structs import _generic


class _DetDrawMixin:
    """
    Non critical methods for visualizing detections
    """
    def draw(self, color='blue', alpha=None, labels=True, centers=False,
             lw=2, fill=False, ax=None):
        """
        Draws boxes using matplotlib

        Example:
            >>> self = Detections.random(num=10, scale=512.0, rng=0, classes=['a', 'b', 'c'])
            >>> self.boxes.translate((-128, -128), inplace=True)
            >>> image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> fig = kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(image)
            >>> # xdoc: -REQUIRES(--show)
            >>> self.draw(color='blue', alpha=None)
            >>> # xdoc: +REQUIRES(--show)
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()
        """
        labels = self._make_labels(labels)
        alpha = self._make_alpha(alpha)
        self.boxes.draw(labels=labels, color=color, alpha=alpha, fill=fill,
                        centers=centers, ax=ax, lw=lw)

    def draw_on(self, image, color='blue', alpha=None, labels=True):
        """
        Draws boxes directly on the image using OpenCV

        Args:
            image (ndarray[uint8]): must be in uint8 format

        Returns:
            ndarray[uint8]: image with labeled boxes drawn on it

        Example:
            >>> import kwplot
            >>> self = Detections.random(num=10, scale=512, rng=0)
            >>> image = (np.random.rand(512, 512) * 255).astype(np.uint8)
            >>> image2 = self.draw_on(image, color='blue')
            >>> # xdoc: +REQUIRES(--show)
            >>> kwplot.figure(fnum=2000, doclf=True)
            >>> kwplot.autompl()
            >>> kwplot.imshow(image2)
            >>> kwplot.show_if_requested()
        """
        labels = self._make_labels(labels)
        alpha = self._make_alpha(alpha)
        image = self.boxes.draw_on(image, color=color, alpha=alpha,
                                   labels=labels)
        return image

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
                    labels = ['{:.4f}'.format(score) for score in self.scores]
                elif labels in ['class+score']:
                    labels = ['{} @ {:.4f}'.format(cid, score)
                              for cid, score in zip(identifers, self.scores)]
                else:
                    raise KeyError('unknown labels key {}'.format(labels))
        return labels


class _DetAlgoMixin:
    """
    Non critical methods for algorithmic manipulation of detections
    """

    def non_max_supression(self, thresh=0.0, perclass=False, impl='auto',
                           daq=False):
        """
        Find high scoring minimally overlapping detections

        Args:
            thresh (float): iou threshold
            perclass (bool): if True, works on a per-class basis
            impl (str): nms implementation to use
            daq (Bool | Dict): if False, uses reqgular nms, otherwise uses
                divide and conquor algorithm. If `daq` is a Dict, then
                it is used as the kwargs to `kwimage.daq_spatial_nms`

        Returns:
            ndarray[int]: indices of boxes to keep
        """
        import kwimage
        classes = self.class_idxs if perclass else None
        tlbr = self.boxes.to_tlbr().data
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
                daqkw['diameter'] = max(self.boxes.width.max(),
                                        self.boxes.height.max())

            keep = kwimage.daq_spatial_nms(tlbr, scores, **daqkw)
        else:
            keep = kwimage.non_max_supression(tlbr, scores, thresh=thresh,
                                              classes=classes, impl=impl)
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

            Additional custom keys may be specified as long as (a) the values
            are array-like and the first axis corresponds to the standard data
            values and (b) are custom keys are listed in the `datakeys` kwargs
            when constructing the Detections.

        meta (Dict):
            This contains contextual information about the detections.  This
            includes the class names, which can be indexed into via the class
            indexes.

    Example:
        >>> self = Detections.random(10)
        >>> other = Detections(self)
        >>> assert other.data == self.data
        >>> assert other.data is self.data, 'try not to copy unless necessary'
    """
    # __slots__ = ('data', 'meta',)

    # Valid keys for the data dictionary
    # NOTE: I'm not sure its productive to restrict to a set of specified
    # properties. It might be better to allow detections to have arbitrary data
    # properties like: velocity, as long as they are array-like. However, I'm
    # not sure how to best structure the code to allow this so it is both clear
    # and efficient. Currently I've allowed the user to specify custom datakeys
    # and metakeys as kwargs, but that design might change.
    __datakeys__ = ['boxes', 'scores', 'class_idxs', 'probs', 'weights']

    # Valid keys for the meta dictionary
    __metakeys__ = ['classes']

    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None,
                 checks=True, **kwargs):
        """
        Construct a Detections object by either explicitly specifying the
        internal data and meta dictionary structures or by passing expected
        attribute names as kwargs. Note that custom data and metadata can be
        specified as long as you pass the names of these keys in the `datakeys`
        and/or `metakeys` kwargs.

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

        Doctest:
            >>> # TODO: move to external unit test
            >>> # Coerce to numpy
            >>> import kwimage
            >>> dets = Detections(
            >>>     boxes=kwimage.Boxes.random(3).numpy(),
            >>>     class_idxs=[0, 1, 1],
            >>>     checks=True,
            >>> )
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
                for k, v in data.items():
                    if isinstance(v, _generic.ObjectList):
                        objlist.append(v)
                    elif isinstance(v, _boxes.Boxes):
                        if v.is_numpy():
                            ndarrays.append(k)
                        else:
                            tensors.append(k)
                    elif isinstance(v, np.ndarray):
                        ndarrays.append(k)
                    elif isinstance(v, torch.Tensor):
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
    def from_coco_annots(cls, anns, cats, classes=None, kpclasses=None,
                         shape=None):
        """
        Example:
            >>> from kwimage.structs.detections import *  # NOQA
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
            >>> cats = [
            >>>     {'id': 0, 'name': 'background'},
            >>>     {'id': 2, 'name': 'class1', 'keypoints': ['spot']}
            >>> ]
            >>> dets = Detections.from_coco_annots(anns, cats)

        Example:
            >>> import kwimage
            >>> # xdoctest: +REQUIRES(--module:ndsampler)
            >>> import ndsampler
            >>> sampler = ndsampler.CocoSampler.demo('photos')
            >>> iminfo, anns = sampler.load_image_with_annots(1)
            >>> shape = iminfo['imdata'].shape[0:2]
            >>> kpclasses = sampler.dset.keypoint_categories()
            >>> dets = kwimage.Detections.from_coco_annots(
            >>>     anns, sampler.dset.dataset['categories'], sampler.catgraph,
            >>>     kpclasses, shape=shape)

        Ignore:
            import skimage
            m = skimage.morphology.disk(4)
            mask = kwimage.Mask.from_mask(m, offset=(2, 3), shape=(20, 20))
            print(mask.to_bytes_rle().data)
        """
        import kwimage
        xywh = np.array([ann['bbox'] for ann in anns], dtype=np.float32)
        boxes = kwimage.Boxes(xywh, 'xywh')
        cids = [ann['category_id'] for ann in anns]
        cid_to_cat = {c['id']: c for c in cats}  # Hack
        cnames = [cid_to_cat[cid]['name'] for cid in cids]
        if classes is None:
            classes = list([cat['name'] for cat in cid_to_cat.values()])
        class_idxs = [classes.index(cname) for cname in cnames]
        dets = Detections(
            boxes=boxes,
            class_idxs=np.array(class_idxs),
            classes=classes,
        )
        if True:
            ss = [ann.get('segmentation', None) for ann in anns]
            masks = [
                None if s is None else kwimage.Mask.coerce(s, shape=shape)
                for s in ss
            ]
            dets.data['masks'] = kwimage.MaskList(masks)

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
                kpcidxs = [kpclasses.index(n) for n in kpnames]
                return kpcidxs
            kpts = []
            for ann in anns:
                k = ann.get('keypoints', None)
                if k is None:
                    kpts.append(k)
                else:
                    kpcidxs = None
                    if kpclasses is not None:
                        kpcidxs = _lookup_kp_class_idxs(ann['category_id'])
                    pts = kwimage.Points(
                        xy=np.array(k).reshape(-1, 3)[:, 0:2],
                        class_idxs=kpcidxs,
                    )
                    kpts.append(pts)
            dets.data['kpts'] = kwimage.PointsList(kpts)

            if kpclasses is not None:
                dets.data['kpts'].meta['classes'] = kpclasses
        return dets

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

    def warp(self, transform, inplace=False):
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
        if inplace:
            self.boxes.warp(transform, inplace=True)
            return self
        else:
            newdata = self.data.copy()
            newdata['boxes'] = self.boxes.warp(transform)
            return self.__class__(newdata, self.meta)

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
        """
        if len(dets) == 0:
            raise ValueError('need at least one detection to concatenate')
        newdata = {}
        for key in dets[0].data.keys():
            if dets[0].data[key] is None:
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
                    raise Exception('Error when trying to concat {}'.format(key))

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
            >>> import kwimage
            >>> dets = kwimage.Detections(boxes=kwimage.Boxes.random(10))
            >>> flags = np.random.rand(len(dets)) > 0.5
            >>> subset = dets.compress(flags)
            >>> assert len(subset) == flags.sum()
            >>> subset = dets.tensor().compress(flags)
            >>> assert len(subset) == flags.sum()
        """
        if len(flags) != len(self):
            raise IndexError('compress must get a flag for every item')

        if self.is_tensor():
            if isinstance(flags, np.ndarray):
                if flags.dtype.kind == 'b':
                    flags = flags.astype(np.uint8)
            flags = torch.ByteTensor(flags).to(self.device)
        newdata = {k: _safe_compress(v, flags, axis) for k, v in self.data.items()}
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
            >>> subset = dets.tensor().take([2, 3, 5, 7])
            >>> assert len(subset) == 4
        """
        if self.is_tensor():
            indices = torch.LongTensor(indices).to(self.device)
        newdata = {k: _safe_take(v, indices, axis) for k, v in self.data.items()}
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
            >>> self = Detections.random(3, tensor=True)
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
                try:
                    if isinstance(val, _boxes.Boxes):
                        newval = val.numpy()
                    else:
                        newval = val.data.cpu().numpy()
                # except AttributeError('memoryview.* no attribute .*cpu'):
                except AttributeError:
                    newval = val
            newdata[key] = newval
        newself = self.__class__(newdata, self.meta)
        return newself

    def tensor(self, device=ub.NoParam):
        """
        Converts numpy to tensors. Does not change memory if possible.

        Example:
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
                if torch.is_tensor(val):
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
    def random(cls, num=10, scale=1.0, rng=None, classes=3, tensor=False):
        """
        Creates dummy data, suitable for use in tests and benchmarks

        Args:
            num (int): number of boxes
            scale (float | tuple, default=1.0): bounding image size
            classes (int | Sequence): list of class labels or number of classes
            tensor (bool, default=False): determines backend
            rng (np.random.RandomState): random state
        """
        import kwimage
        import kwarray
        rng = kwarray.ensure_rng(rng)
        boxes = kwimage.Boxes.random(num=num, scale=scale, rng=rng, tensor=tensor)
        if isinstance(classes, int):
            num_classes = classes
            classes = ['class_{}'.format(c) for c in range(classes)]
        else:
            num_classes = len(classes)
        scores = rng.rand(len(boxes))
        class_idxs = rng.randint(0, num_classes, size=len(boxes))
        if tensor:
            class_idxs = torch.LongTensor(class_idxs)
            scores = torch.FloatTensor(scores)
        self = cls(boxes=boxes, scores=scores, class_idxs=class_idxs,
                   classes=classes)
        return self


def _safe_take(v, indices, axis):
    if v is None:
        return v
    try:
        return _boxes._take(v, indices, axis=axis)
    except TypeError:
        return v.take(indices, axis=axis)


def _safe_compress(v, flags, axis):
    if v is None:
        return v
    try:
        return _boxes._compress(v, flags, axis=axis)
    except TypeError:
        return v.compress(flags, axis=axis)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwimage.structs.detections
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
