import numpy as np
from layered.example import Example
from layered.utility import hstack_lines


class Layer:

    def __init__(self, size, activation):
        assert isinstance(size, int) and size
        self.size = size
        self.activation = activation
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
        outgoing = self.activation.apply(self.incoming)
        assert len(outgoing) == self.size
        self.outgoing = outgoing

    def delta(self):
        """
        The derivative of the activation function at the current state.
        """
        return self.activation.delta(self.incoming, self.outgoing)


class Weight(np.ndarray):

    def __new__(cls, from_size, to_size, init_scale=1e-2):
        # Add extra weights for the biases.
        shape = (from_size + 1, to_size)
        values = np.random.normal(0, init_scale, shape)
        return np.ndarray.__new__(cls, shape, float, values)

    def __str__(self):
        rows = [' '.join('{: >6.3f}'.format(x) for x in row) for row in self]
        return '\n'.join(rows)

    def forward(self, previous):
        # Add bias input of one.
        previous = np.insert(previous, 0, 1)
        assert previous[0] == 1
        current = previous.dot(self)
        return current

    def backward(self, next_):
        current = next_.dot(self.transpose())
        # Don't expose the bias input of one.
        current = current[1:]
        return current


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.sizes = tuple(layer.size for layer in self.layers)
        # Weights are stored as matrices of the shape length of the left layer
        # times length of the right layer.
        shapes = zip(self.sizes[:-1], self.sizes[1:])
        self.weights = list(Weight(*shape) for shape in shapes)
        # Weight matrices are in between the layers.
        assert len(self.weights) == len(self.layers) - 1

    def feed(self, data):
        assert len(data) == self.layers[0].size
        self.layers[0].apply(data)
        return self.forward(self.weights)

    def forward(self, weights):
        """
        Evaluate the network with alternative weights on its last input and
        return the output activation.
        """
        # Propagate trough the remaining layers.
        connections = zip(self.layers[:-1], weights, self.layers[1:])
        for previous, weight, current in connections:
            incoming = weight.forward(previous.outgoing)
            current.apply(incoming)
        # Return the activations of the output layer.
        return self.layers[-1].outgoing

    def visualize(self):
        print('Layers\n------')
        print(hstack_lines(map(str, self.layers), '  '))
        print('Weights\n-------')
        print(hstack_lines(map(str, self.weights), '  '))
