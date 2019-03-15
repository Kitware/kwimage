import ubelt as ub
import cv2
import numpy as np
from . import _generic
import torch


class _GenMixin:
    def is_numpy(self):
        return self.data['xy'].is_numpy()

    def is_tensor(self):
        return self.data['xy'].is_tensor()

    @_generic.memoize_property
    def _impl(self):
        return self.data['xy']._impl

    def tensor(self, device=ub.NoParam):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10)
            >>> self.tensor()
        """
        impl = self._impl
        newdata = {k: v.tensor(device) if hasattr(v, 'tensor')
                   else impl.tensor(v, device)
                   for k, v in self.data.items()}
        new = self.__class__(newdata, self.meta)
        return new

    def numpy(self):
        """
        Example:
            >>> from kwimage.structs.points import *  # NOQA
            >>> self = Points.random(10)
            >>> self.tensor().numpy().tensor().numpy()
        """
        impl = self._impl
        newdata = {k: v.numpy() if hasattr(v, 'numpy') else impl.numpy(v)
                   for k, v in self.data.items()}
        new = self.__class__(newdata, self.meta)
        return new


class Polygon(object):
    """
    Represents a single polygon as set of exterior boundary points and a list
    of internal polygons representing holes.

    Example:
        >>> data = {
        >>>     'exterior': np.array([[13,  1], [13, 19], [25, 19], [25,  1]]),
        >>>     'interiors': [
        >>>         np.array([[13, 13], [14, 12], [24, 12], [25, 13], [25, 18], [24, 19], [14, 19], [13, 18]]),
        >>>         np.array([[13,  2], [14,  1], [24,  1], [25, 2], [25, 11], [24, 12], [14, 12], [13, 11]])]
        >>> }
        >>> self = Polygon(data)
    """
    __datakeys__ = ['exterior', 'interiors']
    __metakeys__ = []

    @classmethod
    def random(cls):
        """
        Example:
            >>> self = Polygon.random()
            >>> self.data
        """
        # TODO: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
        import kwimage
        mask = kwimage.Mask.random()
        exterior = mask.get_polygon()[0]
        self = cls(exterior=exterior)
        return self

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

            import kwimage
            if 'exterior' in data:
                if isinstance(data['exterior'], (np.ndarray, torch.Tensor)):
                    data['exterior'] = kwimage.Coords(data['exterior'])
            if 'interiors' in data:
                holes = []
                for hole in data['interiors']:
                    if isinstance(data['interiors'], (np.ndarray, torch.Tensor)):
                        hole = kwimage.Coords(hole)
                    holes.append(hole)
                data['interiors'] = holes

        elif isinstance(data, self.__class__):
            # Avoid runtime checks and assume the user is doing the right thing
            # if data and meta are explicitly specified
            meta = data.meta
            data = data.data
        if meta is None:
            meta = {}
        self.data = data
        self.meta = meta

    def draw_on(self, image, color='blue', fill=True):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random(10, rng=0).scale(128)
            >>> image = np.zeros((128, 128), dtype=np.float32)
            >>> image = self.draw_on(image)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(image, fnum=1)
        """
        import kwplot
        import kwimage
        # return shape of contours to openCV contours
        thickness = 1

        contour_idx = -1
        # line_type = cv2.LINE_AA
        line_type = cv2.LINE_8
        image = kwimage.ensure_uint255(image)
        image = kwimage.atleast_3channels(image)

        data = self.data
        contours = [data['exterior']] + data['interiors']
        contours = [np.expand_dims(c.astype(np.int), axis=1) for c in contours]

        rgb = kwplot.Color(color).as255()

        image = cv2.fillPoly(image, contours, rgb, line_type, shift=0)

        if 0:
            image = cv2.drawContours(image, contours, contour_idx, rgb, thickness,
                                     line_type)
        image = kwimage.ensure_float01(image)
        return image

    def draw(self, color='blue', ax=None, alpha=None, radius=1):
        """
        Example:
            >>> from kwimage.structs.polygon import *  # NOQA
            >>> self = Polygon.random()
            >>> self.draw()
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from matplotlib import pyplot as plt
            >>> kwplot.figure(fnum=2)
            >>> self.draw()
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 120)
            >>> ax.set_ylim(0, 120)
            >>> ax.invert_yaxis()
        """
        import matplotlib as mpl
        from matplotlib.patches import Path
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        data = self.data

        exterior = data['exterior'].data.tolist()
        exterior.append(exterior[0])
        n = len(exterior)
        verts = []
        verts.extend(exterior)
        codes = [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        interiors = data.get('interiors', [])
        for hole in interiors:
            hole = hole.data.tolist()
            hole.append(hole[0])
            n = len(hole)
            verts.extend(hole)
            codes += [Path.MOVETO] + ([Path.LINETO] * (n - 2)) + [Path.CLOSEPOLY]

        verts = np.array(verts)
        path = Path(verts, codes)
        patch = mpl.patches.PathPatch(path, alpha=.4)
        ax.add_patch(patch)

    def to_imgaug(self, shape):
        import imgaug
        ia_exterior = imgaug.Polygon(self.data['exterior'])
        ia_interiors = [imgaug.Polygon(p) for p in self.data.get('interiors', [])]
        iamp = imgaug.MultiPolygon([ia_exterior] + ia_interiors)
        return iamp


class PolygonList(_generic.ObjectList):
    pass
