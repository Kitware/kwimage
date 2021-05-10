"""
Generic structures for representing transforms

References:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py
"""
import numpy as np


class Transform(object):
    @classmethod
    def coerce(cls, data):
        pass


class Scale(Transform):
    def __init__(self, factor):
        self.factor = factor


class Translate(Transform):
    def __init__(self, offset):
        self.offset = offset


class Rotate(Transform):
    def __init__(self, theta):
        self.theta = theta


class Affine(Transform):
    def __init__(self, matrix):
        self.matrix = matrix

    def __json__(self):
        return {
            'type': self.__class__.__name__.lower(),
            'matrix': self.matrix
        }


class Projective(Transform):
    def __init__(self, matrix):
        self.matrix = matrix

    def invert(self):
        inv_mat = np.linalg.inv(self.matrix)
        return self.__class__(inv_mat)

    def __json__(self):
        return {
            'type': self.__class__.__name__.lower(),
            'matrix': self.matrix
        }


def _devcheck_inv_stability():
    import numpy as np

    rows = []

    for trial_idx in range(30):
        orig_12 = np.random.rand(3, 3).astype(np.float32)
        orig_21 = np.linalg.inv(orig_12)

        mat_12 = orig_12.copy()
        mat_21 = orig_21.copy()

        for idx in range(100):
            _mat_12 = np.linalg.inv(mat_21)
            _mat_21 = np.linalg.inv(mat_12)

            mat_12 = _mat_12
            mat_21 = _mat_21

            err_12 = np.abs(mat_12 - orig_12).sum()
            err_21 = np.abs(mat_21 - orig_21).sum()
            rows.append({
                'idx': idx,
                'error': err_12,
                'label': 'err_12'
            })
            rows.append({
                'idx': idx,
                'error': err_21,
                'label': 'err_21'
            })

    import kwplot
    import pandas as pd
    sns = kwplot.autosns()

    data = pd.DataFrame(rows)

    sns.lineplot(data=data, x='idx', y='error', hue='label')
