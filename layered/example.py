import numpy as np


class Example:
    """
    Immutable class representing one example in a dataset.
    """

    __slots__ = ('_data', '_target')

    def __init__(self, data, target):
        self._data = np.array(data, dtype=float)
        self._target = np.array(target, dtype=float)

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    def __getstate__(self):
        return {'data': self.data, 'target': self.target}

    def __setstate__(self, state):
        self._data = state['data']
        self._target = state['target']

    def __repr__(self):
        data = ' '.join(str(round(x, 2)) for x in self.data)
        target = ' '.join(str(round(x, 2)) for x in self.target)
        return '({})->({})'.format(data, target)
