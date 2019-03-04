import numpy as np
import ubelt as ub
import skimage


class Points(ub.NiceRepr):
    """
    Stores multiple keypoints for a single object.

    This stores both the geometry and the class metadata if available

    Ignore:
        meta = {
         "names" = ['head', 'nose', 'tail'],
         "skeleton" = [(0, 1), (0, 2)],
        }

    Example:
        >>> xy = np.random.rand(10, 2)
        >>> pts = Points(xy=xy)
        >>> print('pts = {!r}'.format(pts))
    """
    __slots__ = ('data', 'meta',)
    # Pre-registered keys for the data dictionary
    __datakeys__ = ['xy', 'class_idxs']
    # Pre-registered keys for the meta dictionary
    __metakeys__ = ['classes']

    def __init__(self, data=None, meta=None, datakeys=None, metakeys=None,
                 **kwargs):
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
        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def warp(self, transform, input_shape=None, inplace=False):
        """
        notes: input_shape is only used when transform is an imgaug augmenter
        """
        if inplace:
            newself = self
        else:
            newself = self.__class__(self.data.copy(), self.meta)

        if isinstance(transform, np.ndarray):
            matrix = transform
        elif isinstance(transform, skimage.transform._geometric.GeometricTransform):
            matrix = transform.params
        else:
            import imgaug
            if isinstance(transform, imgaug.augmenters.Augmenter):
                kpoi = self.to_imgaug(shape=input_shape)
                kpoi = transform.augment_keypoints(kpoi)
                xy = np.array([[kp.x, kp.y] for kp in kpoi.keypoints],
                              dtype=self.xy.dtype)
                newself.data['xy'] = xy
                return newself
            else:
                raise TypeError(type(transform))

        newself.data['xy'] = matrix.dot(newself.data['xy'])
        return newself

    def __nice__(self):
        return 'xy={!r}'.format(self.data['xy'])

    def __len__(self):
        return len(self.data['xy'])

    @property
    def xy(self):
        return self.data['xy']

    def to_imgaug(self, shape):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> xy = np.random.rand(10, 2)
            >>> pts = Points(xy=xy)
            >>> shape = (10, 10)
            >>> kpoi = pts.to_imgaug(shape)
            >>> import imgaug
            >>> augmenter = imgaug.augmenters.Fliplr(p=1)
            >>> aug_kpoi = augmenter.augment_keypoints(kpoi)
            >>> aug_pts = Points.from_imgaug(aug_kpoi)
            >>> aug_pts2 = pts.warp(augmenter, shape=shape)
        """
        import imgaug
        kps = [imgaug.Keypoint(x, y) for x, y in self.data['xy']]
        kpoi = imgaug.KeypointsOnImage(kps, shape=shape)
        return kpoi

    @classmethod
    def from_imgaug(cls, kpoi):
        import numpy as np
        xy = np.array([[kp.x, kp.y] for kp in kpoi.keypoints])
        self = cls(xy=xy)
        return self


class PointsList(ub.NiceRepr):
    """
    Stores a list of Points, each item usually corresponds to a different object.

    Notes:
        # TODO: when the data is homogenous we can use a more efficient
        # representation, otherwise we have to use heterogenous storage.
    """
    pass
