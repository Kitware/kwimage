import ubelt as ub
import xdev


class Spatial(ub.NiceRepr):
    """
    Abstract base class defining the spatial annotation API
    """
    def translate(self, offset, output_dims=None):
        raise NotImplementedError

    def scale(self, factor, output_dims=None):
        raise NotImplementedError

    def warp(self, transform, input_dims=None, output_dims=None, inplace=False):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError

    def draw_on(self, image):
        raise NotImplementedError

    def tensor(self, device=ub.NoParam):
        raise NotImplementedError

    def numpy(self):
        raise NotImplementedError

    @classmethod
    def random(cls):
        raise NotImplementedError


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

    def __nice__(self):
        return 'n={}'.format(len(self))

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @xdev.profile
    def translate(self, offset, output_dims=None):
        newdata = [None if item is None else
                   item.translate(offset, output_dims=output_dims)
                   for item in self.data]
        return self.__class__(newdata, self.meta)

    @xdev.profile
    def scale(self, factor, output_dims=None):
        newdata = [None if item is None else
                   item.scale(factor, output_dims=output_dims)
                   for item in self.data]
        return self.__class__(newdata, self.meta)

    @xdev.profile
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

    def compress(self, flags, axis=0):
        assert axis == 0
        newdata = list(ub.compress(self.data, flags))
        return self.__class__(newdata, self.meta)

    def take(self, indices, axis=0):
        assert axis == 0
        newdata = list(ub.take(self.data, indices))
        return self.__class__(newdata, self.meta)

    def draw(self, **kwargs):
        for item in self.data:
            if item is not None:
                item.draw(**kwargs)

    def draw_on(self, image, **kwargs):
        for item in self.data:
            if item is not None:
                image = item.draw_on(image=image, **kwargs)
        return image

    @xdev.profile
    def tensor(self, device=ub.NoParam):
        return self.apply(lambda item: item.tensor(device))

    @xdev.profile
    def numpy(self):
        return self.apply(lambda item: item.numpy())

    @classmethod
    def concatenate(cls, data):
        raise NotImplementedError

    def is_tensor(cls):
        raise NotImplementedError

    def is_numpy(cls):
        raise NotImplementedError

    @classmethod
    def random(cls):
        raise NotImplementedError
