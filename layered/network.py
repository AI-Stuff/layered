import operator
import numpy as np


class Layer:

    def __init__(self, size, activation):
        assert size and isinstance(size, int)
        self.size = size
        self.activation = activation()
        self.incoming = np.zeros(size)
        self.outgoing = np.zeros(size)
        assert len(self.incoming) == len(self.outgoing) == self.size

    def __len__(self):
        assert len(self.incoming) == len(self.outgoing)
        return len(self.incoming)

    def __repr__(self):
        return repr(self.outgoing)

    def __str__(self):
        table = zip(self.incoming, self.outgoing)
        rows = [' /'.join('{: >6.3f}'.format(x) for x in row) for row in table]
        return '\n'.join(rows)

    def apply(self, incoming):
        """
        Store the incoming activation, apply the activation function and store
        the result as outgoing activation.
        """
        assert len(incoming) == self.size
        self.incoming = incoming
        outgoing = self.activation(self.incoming)
        assert len(outgoing) == self.size
        self.outgoing = outgoing

    def delta(self, above):
        """
        The derivative of the activation function at the current state.
        """
        return self.activation.delta(self.incoming, self.outgoing, above)


class Matrices:

    def __init__(self, shapes, elements=None):
        self.shapes = shapes
        length = sum(x * y for x, y in shapes)
        if elements is not None:
            assert len(elements) == length
            elements = elements.copy()
        else:
            elements = np.zeros(length)
        self.flat = elements

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        if hasattr(index, '__len__'):
            assert isinstance(index[0], int)
            return self[index[0]][index[1:]]
        if isinstance(index, slice):
            return [self[i] for i in self._range_from_slice(index)]
        slice_ = self._locate(index)
        data = self.flat[slice_]
        data = data.reshape(self.shapes[index])
        return data

    def __setitem__(self, index, data):
        if hasattr(index, '__len__'):
            assert isinstance(index[0], int)
            self[index[0]][index[1:]] = data
            return
        if isinstance(index, slice):
            for i in self._range_from_slice(index):
                self[i] = data
            return
        slice_ = self._locate(index)
        data = data.reshape(slice_.stop - slice_.start)
        self.flat[slice_] = data

    def __getattr__(self, name):
        # Tunnel not found properties to the underlying array.
        flat = super().__getattribute__('flat')
        return getattr(flat, name)

    def __setattr_(self, name, value):
        # Ensure that the size of the underlying array doesn't change.
        if name == 'flat':
            assert value.shape == self.flat.shape
        super().__setattr__(name, value)

    def copy(self):
        return Matrices(self.shapes, self.flat.copy())

    def __add__(self, other):
        return self._operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._operation(other, lambda x, y: x / y)

    __rmul__ = __mul__

    __radd__ = __add__

    def _operation(self, other, operation):
        try:
            other = other.flat
        except AttributeError:
            pass
        return Matrices(self.shapes, operation(self.flat, other))

    def _locate(self, index):
        assert isinstance(index, int), (
            'Only single elemente can be indiced in the first dimension.')
        if index < 0:
            index = len(self.shapes) + index
        if not 0 <= index < len(self.shapes):
            raise IndexError
        offset = sum(x * y for x, y in self.shapes[:index])
        length = operator.mul(*self.shapes[index])
        return slice(offset, offset + length)

    def _range_from_slice(self, slice_):
        start = slice_.start if slice_.start else 0
        stop = slice_.stop if slice_.stop else len(self.shapes)
        step = slice_.step if slice_.step else 1
        return range(start, stop, step)

    def __str__(self):
        return str(len(self.flat)) + str(self.flat)


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.sizes = tuple(layer.size for layer in self.layers)
        # Weight matrices have the dimensions of the two layers around them.
        # Also, there is an additional bias input to each weight matrix.
        self.shapes = zip(self.sizes[:-1], self.sizes[1:])
        self.shapes = [(x + 1, y) for x, y in self.shapes]
        # Weight matrices are in between the layers.
        assert len(self.shapes) == len(self.layers) - 1

    def feed(self, weights, data):
        """
        Evaluate the network with alternative weights on the input data and
        return the output activation.
        """
        assert len(data) == self.layers[0].size
        self.layers[0].apply(data)
        # Propagate trough the remaining layers.
        connections = zip(self.layers[:-1], weights, self.layers[1:])
        for previous, weight, current in connections:
            incoming = self.forward(weight, previous.outgoing)
            current.apply(incoming)
        # Return the activations of the output layer.
        return self.layers[-1].outgoing

    @staticmethod
    def forward(weight, activations):
        # Add bias input of one.
        activations = np.insert(activations, 0, 1)
        assert activations[0] == 1
        right = activations.dot(weight)
        return right

    @staticmethod
    def backward(weight, activations):
        left = activations.dot(weight.transpose())
        # Don't expose the bias input of one.
        left = left[1:]
        return left
