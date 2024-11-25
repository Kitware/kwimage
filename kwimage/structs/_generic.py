import ubelt as ub
import sys
import kwarray
import numbers
import numpy as np
from collections import abc
# import abc

# try:
#     import torch
# except Exception:
#     torch = None
#     ARRAY_TYPES = (np.ndarray,)
# else:
#     ARRAY_TYPES = (np.ndarray, torch.Tensor)


def isinstance_arraytypes(obj):
    """
    workaround so we dont need to import torch at the global level
    """
    torch = sys.modules.get('torch', None)
    if torch is None:
        return isinstance(obj, np.ndarray)
    else:
        return isinstance(obj, (np.ndarray, torch.Tensor))


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


# class ListProxy(abc.MutableSequence):
class _ExperimentalListProxy:
    """
    We may modify this implementation to be more generic in the future and
    directly inherit from :class:`abc.MutableSequence`. This intermediate form
    helps preserve backwards compatibility of ObjectList, but it is unclear if
    that is neeeded.

    Partially generated via ChatGPT.

    Requires that the inheriting class has a ``data`` attribute.
    """

    def __getitem__(self, index):
        """Retrieve an item by its index."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Update an item at the specified index."""
        self.data[index] = value

    def __delitem__(self, index):
        """Delete an item at the specified index."""
        del self.data[index]

    def __len__(self):
        """Return the length of the sequence."""
        return len(self.data)

    def insert(self, index, value):
        """Insert an item at a specific index."""
        self.data.insert(index, value)

    def append(self, value):
        """Add an item to the end of the sequence."""
        self.data.append(value)

    def clear(self):
        """Remove all items from the sequence."""
        self.data.clear()

    def reverse(self):
        """Reverse the sequence in place."""
        self.data.reverse()

    def extend(self, other):
        """Extend the sequence by appending elements from an iterable."""
        self.data.extend(other)

    def pop(self, index=-1):
        """Remove and return an item at the given index."""
        return self.data.pop(index)

    def remove(self, value):
        """Remove the first occurrence of a value."""
        self.data.remove(value)

    def __iadd__(self, other):
        """Support in-place addition (+=) to extend the sequence."""
        return self.data.__iadd__(other)

    def __contains__(self, item):
        """Check if the item is in the sequence."""
        return item in self.data

    def __iter__(self):
        """Return an iterator over the sequence."""
        return iter(self.data)

    def __reversed__(self):
        """Return a reverse iterator over the sequence."""
        return self.data.__reversed__()

    def index(self, value, start=0, stop=None):
        """
        Return the index of the first occurrence of a value.
        Raise ValueError if the value is not present.
        """
        return self.data.index(value, start, stop)

    def count(self, value):
        """Return the number of occurrences of a value."""
        return self.data.count(value)


class ObjectList(Spatial, _ExperimentalListProxy):
    """
    Stores a list of potentially heterogenous structures, each item usually
    corresponds to a different object.

    Example:
        >>> # TODO: better test that flexes the Spatial inheritence and
        >>> # method beyond those in _ExperimentalListProxy
        >>> from kwimage.structs._generic import *  # NOQA
        >>> class SpatialList(ObjectList):
        >>>    ...
        >>> self = SpatialList([])
        >>> self.append(1)
        >>> self.extend([1])
        >>> 1 in self
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
        """
        Generic draw method for list of spatial annotations
        """
        if 'setlim' in kwargs:
            import warnings
            warnings.warn('ObjectList currently need special handling for setlim kwarg')
        patches = []
        for item in self.data:
            if item is not None:
                patch = item.draw(**kwargs)
                patches.append(patch)
        return patches

    def draw_on(self, image, **kwargs):
        """
        TODO:
            document fastdraw - it flattens all subobjects into the same layer
            and then does any alpha blending. Is there a better name?
        """
        import kwimage

        # Take care of data prep before looping
        alpha = kwargs.get('alpha', None)
        copy = kwargs.pop('copy', False)

        has_float_alpha = alpha is not None and alpha < 1.0

        if has_float_alpha:
            image = kwimage.ensure_float01(image, copy=copy)
        elif copy:
            image = image.copy()

        # Check if we want to do the fastdraw hack
        fastdraw = kwargs.pop('fastdraw', 'auto')
        if fastdraw == 'auto':
            fastdraw = has_float_alpha

        if fastdraw:
            # Fast draw hack will remove the alpha from the subcalls to
            # draw_on, instead we will draw with full alpha on an empty canvas
            # and then blend together everything at the end.
            orig_canvas = image
            overlay_canvas = np.zeros_like(image, shape=(image.shape[0:2] + (4,)))
            image = overlay_canvas
            kwargs['alpha'] = None

        # Handle per-instance arguments
        # If color is given an it corresponds to each subitem
        # then pass the appropriate arg to each subitem
        instkw_list = [{} for _ in range(len(self.data))]
        selflen = len(self.data)
        _handle_perinstance_color_arg(selflen, instkw_list, kwargs, 'color')
        _handle_perinstance_color_arg(selflen, instkw_list, kwargs, 'edgecolor')
        _handle_perinstance_color_arg(selflen, instkw_list, kwargs, 'facecolor')

        for item, instkw in zip(self.data, instkw_list):
            if item is not None:
                image = item.draw_on(image=image, **kwargs, **instkw)

        if fastdraw:
            # Blend the empty canvas back onto
            overlay_canvas = image
            overlay_canvas[..., 3] *= alpha
            image = kwimage.overlay_alpha_images(overlay_canvas, orig_canvas)
            pass

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
            >>> # xdoctest: +REQUIRES(module:cv2)
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

    # @classmethod
    # def random(cls):
    #     raise NotImplementedError


def _handle_perinstance_color_arg(selflen, instkw_list, kwargs, argname):
    """
    helper to expand any color argument into multiple color arguments for each
    instance handled by the generic draw on method. This allows the user to
    specify a list of colors for each member, or a single color to be applied
    to everyone.
    """
    if argname in kwargs:
        color = kwargs.pop(argname, None)
        if (ub.iterable(color) and len(color) == selflen and
             len(color) > 0 and not isinstance(ub.peek(color), numbers.Number)):
            for d, c in zip(instkw_list, color):
                d[argname] = c
        else:
            kwargs[argname] = color


def _coerce_color_list_for(image, color, num):
    """
    Get a list of colors for each item
    """
    import kwimage
    if (ub.iterable(color) and len(color) == num and
         len(color) > 0 and not isinstance(ub.peek(color), numbers.Number)):
        color_list = [kwimage.Color(c).forimage(image) for c in color]
    else:
        color_list = [kwimage.Color(color).forimage(image)] * num
    return color_list


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


def _setlim(xmin, ymin, xmax, ymax, setlim=False, ax=None):
    """
    Helper for setlim argument for draw function
    """
    if ax is None:
        from matplotlib import pyplot as plt
        ax = plt.gca()

    if isinstance(setlim, str):
        if setlim == 'grow':
            # only allow growth
            x1_, x2_ = ax.get_xlim()
            xmin = min(x1_, xmin)
            xmax = max(x2_, xmax)

            y1_, y2_ = ax.get_ylim()
            ymin = min(y1_, ymin)
            ymax = max(y2_, ymax)
        else:
            raise KeyError(setlim)

    if isinstance(setlim, float):
        w = abs(xmax - xmin)
        h = abs(ymax - ymin)
        xpad = ((w * setlim) - w) / 2
        ypad = ((h * setlim) - h) / 2

        xmin = xmin - xpad
        ymin = ymin - ypad

        xmax = xmax + xpad
        ymax = ymax + ypad

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _handle_color_args_for(color, alpha, border, fill, edgecolor, facecolor, image):
    """
    Common logic for boxes and polygons
    """
    import kwimage
    if facecolor is None:
        if fill:
            facecolor = color
    elif facecolor is True:
        facecolor = color
    else:
        facecolor = kwimage.Color(facecolor, alpha=alpha).forimage(image)

    if edgecolor is None:
        if border:
            edgecolor = color
    elif edgecolor is True:
        edgecolor = color
    else:
        edgecolor = kwimage.Color(edgecolor, alpha=alpha).forimage(image)
