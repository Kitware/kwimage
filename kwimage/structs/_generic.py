import ubelt as ub


class ObjectList(ub.NiceRepr):
    """
    Stores a list of potentially heterogenous structures, each item usually
    corresponds to a different object.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __nice__(self):
        return 'n={}'.format(len(self))

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def translate(self, offset, output_shape=None):
        newdata = [None if item is None else
                   item.translate(offset, output_shape=output_shape)
                   for item in self.data]
        return ObjectList(newdata)

    def scale(self, factor, output_shape=None):
        newdata = [None if item is None else
                   item.scale(factor, output_shape=output_shape)
                   for item in self.data]
        return ObjectList(newdata)

    def warp(self, transform, input_shape=None, output_shape=None):
        newdata = [None if item is None else
                   item.warp(transform, input_shape=input_shape,
                             output_shape=output_shape)
                   for item in self.data]
        return ObjectList(newdata)

    def apply(self, func):
        newdata = [None if item is None else func(item) for item in self.data]
        return ObjectList(newdata)

    def compress(self, flags, axis=0):
        assert axis == 0
        newdata = list(ub.compress(self.data, flags))
        return ObjectList(newdata)

    def take(self, indices, axis=0):
        assert axis == 0
        newdata = list(ub.take(self.data, indices))
        return ObjectList(newdata)

    def draw(self, **kwargs):
        for item in self.data:
            if item is not None:
                item.draw(**kwargs)

    def draw_on(self, image, **kwargs):
        for item in self.data:
            if item is not None:
                image = item.draw(image=image, **kwargs)
        return image

    def tensor(self, device=ub.NoParam):
        return self.apply(lambda item: item.tensor(device))

    def numpy(self):
        return self.apply(lambda item: item.numpy())
