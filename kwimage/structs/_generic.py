import ubelt as ub
import kwarray
import numpy as np
# import abc

try:
    import torch
except Exception:
    torch = None
    ARRAY_TYPES = (np.ndarray,)
else:
    ARRAY_TYPES = (np.ndarray, torch.Tensor)


# class Spatial(ub.NiceRepr, abc.ABC):
class Spatial(ub.NiceRepr):
    """
    Abstract base class defining the spatial annotation API
    """

    # @abc.abstractmethod
    # def translate(self, offset, output_dims=None):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def scale(self, factor, output_dims=None):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
    #     raise NotImplementedError

    # def draw(self, image=None, **kwargs):
    #     # If draw doesnt exist use draw_on
    #     import numpy as np
    #     if image is None:
    #         dims = self.bounds
    #         shape = tuple(dims) + (4,)
    #         image = np.zeros(shape, dtype=np.float32)
    #     image = self.draw_on(image, **kwargs)
    #     import kwplot
    #     kwplot.imshow(image)

    # @abc.abstractmethod
    # def draw_on(self, image):
    #     """
    #     TODO -
    #         - [ ] Choose 1:
    #             Should accept either uint255 or float01, and should return the
    #             same kind
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def tensor(self, device=ub.NoParam):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def numpy(self):
    #     raise NotImplementedError

    # @classmethod
    # @abc.abstractmethod
    # def random(cls):
    #     raise NotImplementedError


class ObjectList(Spatial):
    """
    Stores a list of potentially heterogenous structures, each item usually
    corresponds to a different object.
    """

    # __slots__ = ('data', 'meta',)

    def __init__(self, data, meta=None):
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        try:
            return self.data.dtype
        except Exception:
            print('kwimage._generic: no dtype for ' + str(type(self.data)))
            raise

    def __nice__(self):
        return 'n={}'.format(len(self))

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def translate(self, offset, output_dims=None, inplace=False):
        newdata = [None if item is None else
                   item.translate(offset, output_dims=output_dims,
                                  inplace=inplace)
                   for item in self.data]
        return self.__class__(newdata, self.meta)

    def scale(self, factor, output_dims=None, inplace=False):
        newdata = [None if item is None else
                   item.scale(factor, output_dims=output_dims, inplace=inplace)
                   for item in self.data]
        return self.__class__(newdata, self.meta)

    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        if inplace:
            for item in self.data:
                if item is not None:
                    item.warp(transform, input_dims=input_dims,
                              output_dims=output_dims, inplace=inplace)
            return self
        else:
            newdata = [None if item is None else
                       item.warp(transform, input_dims=input_dims,
                                 output_dims=output_dims, inplace=inplace)
                       for item in self.data]
            return self.__class__(newdata, self.meta)

    def apply(self, func):
        newdata = [None if item is None else func(item) for item in self.data]
        return self.__class__(newdata, self.meta)

    def to_coco(self, style='orig'):
        for item in self.data:
            if item is None:
                yield None
            else:
                yield item.to_coco(style=style)

    def compress(self, flags, axis=0):
        assert axis == 0
        newdata = list(ub.compress(self.data, flags))
        return self.__class__(newdata, self.meta)

    def take(self, indices, axis=0):
        assert axis == 0
        newdata = list(ub.take(self.data, indices))
        return self.__class__(newdata, self.meta)

    def draw(self, **kwargs):
        patches = []
        for item in self.data:
            if item is not None:
                patch = item.draw(**kwargs)
                patches.append(patch)
        return patches

    def draw_on(self, image, **kwargs):
        for item in self.data:
            if item is not None:
                image = item.draw_on(image=image, **kwargs)
        return image

    def tensor(self, device=ub.NoParam):
        return self.apply(lambda item: item.tensor(device))

    def numpy(self):
        return self.apply(lambda item: item.numpy())

    @classmethod
    def concatenate(cls, items, axis=0):
        """
        Args:
            items (Sequence[ObjectList]): multiple object lists of the same type
            axis (int | None): unused, always implied to be axis 0

        Returns:
            ObjectList: combined object list

        Example:
            >>> import kwimage
            >>> cls = kwimage.MaskList
            >>> sub_cls = kwimage.Mask
            >>> item1 = cls([sub_cls.random(), sub_cls.random()])
            >>> item2 = cls([sub_cls.random()])
            >>> items = [item1, item2]
            >>> new = cls.concatenate(items)
            >>> assert len(new) == 3
        """
        if len(items) == 0:
            new = cls([])
        else:
            newdata = []
            for item in items:
                newdata.extend(item.data)

        newmeta = items[0].meta
        new = cls(newdata, newmeta)
        return new

    def is_tensor(cls):
        raise NotImplementedError

    def is_numpy(cls):
        raise NotImplementedError

    @classmethod
    def random(cls):
        raise NotImplementedError


def _consistent_dtype_fixer(data):
    """
    helper for ensuring out.dtype == in.dtype
    """
    import kwimage
    if data.dtype.kind == 'f':
        return kwimage.ensure_float01
    elif data.dtype.kind == 'u':
        return kwimage.ensure_uint255
    else:
        raise TypeError(data.dtype)


def _safe_take(data, indices, axis):
    if data is None:
        return data
    try:
        return data.take(indices, axis=axis)
    except (TypeError, AttributeError):
        return kwarray.ArrayAPI.take(data, indices, axis=axis)


def _safe_compress(data, flags, axis):
    if data is None:
        return data
    try:
        return data.compress(flags, axis=axis)
    except (TypeError, AttributeError):
        return kwarray.ArrayAPI.compress(data, flags, axis=axis)


def _issubclass2(child, parent):
    """
    Uses string comparisons to avoid ipython reload errors.
    Much less robust though.
    """
    # String comparison
    if child.__name__ == parent.__name__:
        if child.__module__ == parent.__module__:
            return True
    # Recurse through classes of obj
    return any(_issubclass2(base, parent) for base in child.__bases__)


def _isinstance2(obj, cls):
    """
    Uses string comparisons to avoid ipython reload errors.
    Much less robust though.

    Example:
        import kwimage
        from kwimage.structs import _generic
        cls = kwimage.structs._generic.ObjectList
        obj = kwimage.MaskList([])
        _generic._isinstance2(obj, cls)

        _generic._isinstance2(kwimage.MaskList([]), _generic.ObjectList)

        dets = kwimage.Detections(
            boxes=kwimage.Boxes.random(3).numpy(),
            class_idxs=[0, 1, 1],
            segmentations=kwimage.MaskList([None] * 3)
        )
    """
    if isinstance(obj, cls):
        return True
    try:
        return _issubclass2(obj.__class__, cls)
    except Exception:
        return False
    return False
