import collections

import numpy as np
import matplotlib.pyplot as plt


class Example:
    """
    Immutable class representing one example in a dataset.
    """
    __slots__ = ('data', 'target')

    def __init__(self, data, target):
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'target', target)

    def __setattr__(self, *args):
        raise TypeError

    def __delattr__(self, *args):
        raise TypeError

    def __repr__(self):
        data = ' '.join(str(round(x, 2)) for x in self.data)
        target = ' '.join(str(round(x, 2)) for x in self.target)
        return '({})->({})'.format(data, target)


class Activation:

    @staticmethod
    def apply(incoming):
        raise NotImplemented

    @staticmethod
    def delta(incoming, outgoing):
        raise NotImplemented


class Sigmoid(Activation):

    @staticmethod
    def apply(incoming):
        return 1 / (1 + np.exp(-incoming))

    @staticmethod
    def delta(incoming, outgoing):
        return outgoing * (1 - outgoing)


class Linear(Activation):

    @staticmethod
    def apply(incoming):
        return incoming

    @staticmethod
    def delta(incoming, outgoing):
        return np.ones(incoming.shape).astype(float)


class Relu(Activation):

    @staticmethod
    def apply(incoming):
        return np.maximum(incoming, 0)

    @staticmethod
    def delta(incoming, outgoing):
        return np.greater(incoming, 0).astype(float)


class Softmax(Activation):

    @staticmethod
    def apply(incoming):
        exp = np.exp(incoming)
        return exp / np.sum(exp)

    @staticmethod
    def delta(incoming, outgoing):
        others = np.dot(-outgoing, outgoing) - (-outgoing * outgoing)
        current = outgoing * (1 - outgoing)
        return others + current


class Cost:

    @staticmethod
    def apply(prediction, target):
        raise NotImplemented

    @staticmethod
    def delta(prediction, target):
        raise NotImplemented


class SquaredCost(Cost):

    @staticmethod
    def apply(prediction, target):
        return np.sum(np.square(prediction - target)) / 2

    @staticmethod
    def delta(prediction, target):
        return prediction - target


class CrossEntropyCost(Cost):

    @staticmethod
    def apply(prediction, target):
        return np.sum(-target * np.log(prediction))

    @staticmethod
    def delta(prediction, target):
        return target-prediction


class Layer:

    def __init__(self, size, activation):
        assert isinstance(size, int) and size
        self.size = size
        self.activation = activation
        self.incoming = np.zeros(size)
        self.outgoing = np.zeros(size)
        assert len(self.incoming) == len(self.outgoing) == self.size

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

    def __new__(cls, from_size, to_size, init_scale=1e-4):
        # Add extra weights for the biases.
        shape = (from_size + 1, to_size)
        values = np.random.normal(0, init_scale, shape)
        return np.ndarray.__new__(cls, shape, float, values)

    def forward(self, previous):
        # Add bias input of one.
        previous = np.insert(previous, 1, 0)
        current = previous.dot(self)
        return current

    def backward(self, next_):
        current = next_.dot(self.transpose())
        # Remove bias input of one.
        current = current[1:]
        return current


class Network:

    def __init__(self, layers, cost, checked=False):
        self.layers = layers
        self.cost = cost
        self.checked = checked
        self.sizes = tuple(layer.size for layer in self.layers)
        # Weights are stored as matrices of the shape length of the left layer
        # times length of the right layer.
        shapes = zip(self.sizes[:-1], self.sizes[1:])
        self.weights = list(Weight(*shape) for shape in shapes)
        # Weight matrices are in between the layers.
        assert len(self.weights) == len(self.layers) - 1

    def train(self, examples, learning_rate):
        """
        Perform a forward and backward pass for each example and average the
        gradients. Update the weights in the opposite direction scaled by the
        learning rate. Return the average cost over the examples.
        """
        # In batch gradient decent, we average the gradients over the training
        # examples.
        cost = 0
        gradient = list(np.zeros(weight.shape) for weight in self.weights)
        for example in examples:
            prediction = self.feed(example.data)
            # print('output:', prediction, 'target:', example.target)
            for i, delta in enumerate(self._backpropagation(example.target)):
                gradient[i] += delta
            cost += self.cost.apply(prediction, example.target)
        # Normalize by the number of examples.
        cost /= len(examples)
        gradient = list(x / len(examples) for x in gradient)
        # The minimum and maximum gradient values are useful to validate
        # the gradient calculation and understand what the network is doing
        # internally.
        # flat = np.hstack(np.array(list(x.flatten() for x in gradient)))
        # print('gradient min:', flat.min(), 'max:', flat.max())
        # Clip extreme gradient values.
        # for i in range(len(gradient)):
        #    gradient[i] = np.clip(gradient[i], -100, 100)
        # Check the gradient.
        if self.checked:
            numerical = self._numerical_gradient(examples)
            difference = list(np.absolute(x - y) for x, y in
                zip(gradient, numerical))
            worst = max(x.max() for x in difference)
            if worst > 1e-4:
                print(worst)
        # Update weights in opposide gradient-direction multiplied by the
        # learning rate.
        for index, derivative in enumerate(gradient):
            self.weights[index] -= learning_rate * derivative
        # Return the cost on the provided training examples.
        return cost

    def feed(self, data):
        return self._feed(data, self.weights)

    def evaluate(self, examples):
        return self._evaluate(examples, self.weights)

    def _feed(self, data, weights):
        assert len(data) == self.layers[0].size
        self.layers[0].apply(data)
        # Propagate trough the remaining layers.
        connections = zip(self.layers[:-1], weights, self.layers[1:])
        for previous, weight, current in connections:
            incoming = weight.forward(previous.outgoing)
            current.apply(incoming)
        # Return the activations of the output layer.
        return self.layers[-1].outgoing

    def _evaluate(self, examples, weights):
        """
        Return the average cost over the examples.
        """
        cost = 0
        for example in examples:
            prediction = self._feed(example.data, weights)
            cost += self.cost.apply(prediction, example.target)
        return cost / len(examples)

    def _backpropagation(self, target):
        delta_layers = self._delta_layers(target)
        delta_weights = self._delta_weights(delta_layers)
        return delta_weights

    def _delta_layers(self, target):
        assert len(target) == self.layers[-1].size
        # We start with the gradient at the output layer. It's computed as the
        # product of error derivative and local derivative at the last layer.
        prediction = self.layers[-1].outgoing
        cost = self.cost.delta(prediction, target)
        local = self.layers[-1].delta()
        gradient = [cost * local]
        # Propagate backwards trough the hidden layers but not the input layer.
        hidden = list(zip(self.weights[1:], self.layers[1:-1]))
        for weight, layer in reversed(hidden):
            # The gradient at a layer is computed as the derivative of both the
            # local activation and the weighted sum of the derivatives in the
            # deeper layer.
            deeper = weight.backward(gradient[-1])
            local = layer.delta()
            gradient.append(deeper * local)
        gradient = list(reversed(gradient))
        # We computed the gradient at the hidden layers and the output layer.
        assert len(gradient) == len(self.layers) - 1
        assert all(len(x) == y.size for x, y in zip(gradient, self.layers[1:]))
        return gradient

    def _delta_weights(self, delta_layers):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for previous, delta in zip(self.layers[:-1], delta_layers):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            bias_and_activation = np.insert(previous.outgoing, 1, 0)
            gradient.append(np.outer(bias_and_activation, delta))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(self.weights)
        assert all(len(x) == len(y) for x, y in zip(gradient, self.weights))
        return gradient

    def _numerical_gradient(self, examples, distance=1e-5):
        """
        Modify each weight individually in both directions to calculate a
        numerical gradient of the weights.
        """
        # We need a copy of the weights that we can modify to evaluate the cost
        # function on.
        weights = self.weights.copy()
        gradient = list(np.zeros(weight.shape) for weight in self.weights)
        for i, connection in enumerate(self.weights):
            for j, original in enumerate(connection):
                # Sample above and below and compute costs.
                weights[i][j] = original + distance
                above = self._evaluate(examples, weights)
                weights[i][j] = original - distance
                below = self._evaluate(examples, weights)
                # Compute numerical gradient.
                gradient[i][j] = (above - below) / (2 * distance)
                # Restore original value for the next coordinates in the loop.
                weights[i][j] = original
        return gradient


class Trainer:

    def __init__(self, network, batch_size=10, plot_freq=5e2,
            plot_smoothing=100):
        self.network = network
        self.batch_size = batch_size
        self.plot_freq = plot_freq
        self.plot_smoothing = plot_smoothing
        self.costs = []
        self.smoothed = []
        self._init_plot()
        self._init_metadata()

    def _init_plot(self):
        self.plot, = plt.plot([], [])
        plt.xlabel('Batch')
        plt.ylabel('Cost')
        plt.ion()
        plt.show()

    def _init_metadata(self):
        self.metadata = collections.OrderedDict()
        self.metadata['cost'] = str(self.network.cost.__name__)
        self.metadata['layers'] = []
        for index, layer in enumerate(self.network.layers):
            self.metadata['layers'].append('{} ({})'.format(
                layer.activation.__name__, layer.size))
        self.metadata['learning rate'] = []
        self.metadata['batch size'] = self.batch_size

    def __call__(self, examples, learning_rate=1e-3):
        """
        Split examples into batches and train on them. Return a list of the
        costs on each batch.
        """
        self.metadata['learning rate'].append(learning_rate)
        self._print_metadata()
        plot_batches = max(1, int(self.plot_freq // self.batch_size))
        for example in self._batched(examples, self.batch_size):
            cost = self.network.train(example, learning_rate)
            self.costs.append(cost)
            self.smoothed.append(np.mean(self.costs[-self.plot_smoothing:]))
            if len(self.costs) % plot_batches == 0:
                self._plot_cost()

    def _plot_cost(self):
        range_ = range(len(self.smoothed))
        plt.xlim(xmin=0, xmax=range_[-1])
        plt.ylim(ymin=0, ymax=1.1*max(self.smoothed))
        self.plot.set_data(range_, self.smoothed)
        plt.draw()
        plt.pause(0.001)

    def _print_metadata(self):
        print('')
        indent = max(len(key) for key in self.metadata) + 1
        width = indent + 1 + max(len(str(value)) for value in
                self.metadata.values())
        print('Meta parameters', '-' * 32,  sep='\n')
        for key, value in self.metadata.items():
            print(('{: <' + str(indent) + '}').format(key + ':'), end=' ')
            if isinstance(value, list):
                print(value[0])
                list(print(' ' * indent, x) for x in value[1:])
            else:
                print(value)
        print('')

    @staticmethod
    def _batched(examples, size):
        for i in range(0, len(examples), size):
            yield examples[i:i+size]


def examples_regression(amount, inputs=10):
    data = np.random.rand(amount, inputs)
    products = np.prod(data, axis=1)
    products = products / np.max(products)
    sums = np.sum(data, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])
    return [Example(x, y) for x, y in zip(data, targets)]


def examples_classification(amount, classes=3, inputs=10):
    data = np.random.randint(0, 1000, (amount, inputs))
    mods = np.mod(np.sum(data, axis=1), classes)
    data = data.astype(float) / data.max()
    targets = np.zeros((amount, classes))
    for index, mod in enumerate(mods):
        targets[index][mod] = 1
    return [Example(x, y) for x, y in zip(data, targets)]


if __name__ == '__main__':
    # Generate and split dataset.
    examples = examples_regression(100000)
    split = int(0.8 * len(examples))
    training, testing = examples[:split], examples[split:]
    # Create a network. The input and output layer sizes are deriven from the
    # input and target data.
    network = Network([
        Layer(len(training[0].data), Linear),
        Layer(10, Sigmoid),
        Layer(10, Sigmoid),
        Layer(len(training[0].target), Sigmoid)
    ], SquaredCost, checked=False)
    # Training.
    trainer = Trainer(network, batch_size=10, plot_smoothing=100)
    for _ in range(1):
        trainer(training, 1e-2)
    # Evaluation.
    cost = network.evaluate(testing)
    print('Test set cost:', cost)

