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


class SquaredErrors(Cost):

    @staticmethod
    def apply(prediction, target):
        return np.sum(np.square(prediction - target)) / 2

    @staticmethod
    def delta(prediction, target):
        return prediction - target


class CrossEntropy(Cost):

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


class Gradient:

    def __init__(self, *args, **kwargs):
        assert len(args) == 2 and not kwargs
        assert isinstance(args[0], Network) and issubclass(args[1], Cost)
        self.network = args[0]
        self.cost = args[1]()

    def apply(self, target):
        raise NotImplemented


class Backpropagation(Gradient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, target):
        return self._weights(self._layers(target))

    def _layers(self, target):
        assert len(target) == self.network.layers[-1].size
        # We start with the gradient at the output layer. It's computed as the
        # product of error derivative and local derivative at the last layer.
        prediction = self.network.layers[-1].outgoing
        cost = self.cost.delta(prediction, target)
        local = self.network.layers[-1].delta()
        gradient = [cost * local]
        # Propagate backwards trough the hidden layers but not the input layer.
        hidden = list(zip(network.weights[1:], self.network.layers[1:-1]))
        for weight, layer in reversed(hidden):
            # The gradient at a layer is computed as the derivative of both the
            # local activation and the weighted sum of the derivatives in the
            # deeper layer.
            deeper = weight.backward(gradient[-1])
            local = layer.delta()
            gradient.append(deeper * local)
        gradient = list(reversed(gradient))
        # We computed the gradient at the hidden layers and the output layer.
        assert len(gradient) == len(network.layers) - 1
        assert all(len(x) == y.size for x, y in
            zip(gradient, self.network.layers[1:]))
        return gradient

    def _weights(self, delta_layers):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for previous, delta in zip(network.layers[:-1], delta_layers):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            bias_and_activation = np.insert(previous.outgoing, 1, 0)
            gradient.append(np.outer(bias_and_activation, delta))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(network.weights)
        assert all(len(x) == len(y) for x, y in
            zip(gradient, self.network.weights))
        return gradient


class NumericalGradient(Gradient):

    def __init__(self, *args, **kwargs):
        self.distance = kwargs.pop('distance', 1e-5)
        super().__init__(*args, **kwargs)

    def apply(self, target):
        """
        Modify each weight individually in both directions to calculate a
        numerical gradient of the weights.
        """
        # We need a copy of the weights that we can modify to evaluate the cost
        # function on.
        weights = self.network.weights.copy()
        gradient = list(np.zeros(weight.shape) for weight
            in self.network.weights)
        for i, connection in enumerate(network.weights):
            for j, original in np.ndenumerate(connection):
                # Sample above and below and compute costs.
                weights[i][j] = original + self.distance
                above = self._evaluate(weights, target)
                weights[i][j] = original - self.distance
                below = self._evaluate(weights, target)
                # Restore the original value so we can reuse the weight matrix
                # for the next iteration.
                weights[i][j] = original
                # Compute the numerical gradient.
                gradient[i][j] = (above - below) / (2 * self.distance)
        return gradient

    def _evaluate(self, weights, target):
        prediction = self.network.forward(weights)
        cost = self.cost.apply(prediction, target)
        return cost


class CheckedGradient(Gradient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args[:-1], **kwargs)
        GradientClass = args[-1]
        assert issubclass(GradientClass, Gradient)
        self.gradient = GradientClass(self.network, args[1])
        self.numerical = NumericalGradient(self.network, args[1])

    def apply(self, target):
        gradient = self.gradient.apply(target)
        computed = self._flatten(gradient)
        numerical = self._flatten(self.numerical.apply(target))
        distance = np.absolute(computed - numerical)

        print('\n      Numerical    Computed     Difference')
        print('Min {:12.8f} {:12.8f} {:12.8f}'.format(numerical.min(),
            computed.min(), distance.min()))
        print('Max {:12.8f} {:12.8f} {:12.8f}'.format(numerical.max(),
            computed.max(), distance.max()))

        if distance.max() > 1e-3 * np.absolute(numerical).max():
            print('Numerical gradient check failed')
        else:
            print('Gradient looks good')
        return gradient

    def _flatten(self, gradient):
        return np.hstack(np.array(list(x.flatten() for x in gradient)))


class Optimization:

    def __init__(self, *args, **kwargs):
        assert len(args) == 2 and not kwargs
        assert isinstance(args[0], Network)
        self.network = args[0]
        self.cost = args[1]()

    def apply(self, examples):
        """
        Expected to return a list or generator of cost values.
        """
        raise NotImplemented


class GradientDecent(Optimization):

    def __init__(self, *args, **kwargs):
        assert isinstance(args[-1], Gradient)
        self.gradient = args[-1]
        self.learning_rate = kwargs.pop('learning_rate', 1e-6)
        super().__init__(*args[:-1], **kwargs)

    def apply(self, examples):
        """
        Perform a forward and backward pass for each example. For each
        example, update the weights in the opposite direction scaled by the
        learning rate. Return the cost for each sample.
        """
        for example in examples:
            gradient, cost = self._compute(example)
            self._update_weights(gradient)
            yield cost

    def _update_weights(self, gradient):
        for index, derivative in enumerate(gradient):
            self.network.weights[index] -= self.learning_rate * derivative

    def _compute(self, example):
        """
        Compute and average gradients and costs over all the examples.
        """
        prediction = self.network.feed(example.data)
        gradient = self.gradient.apply(example.target)
        cost = self.cost.apply(prediction, example.target)
        return gradient, cost


class BatchGradientDecent(GradientDecent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, examples):
        """
        Perform a forward and backward pass for each example and average the
        gradients. Update the weights in the opposite direction scaled by the
        learning rate. Return the average cost over the examples.
        """
        gradient, cost = self._compute(examples)
        self._update_weights(gradient)
        return [cost]

    def _compute(self, examples):
        """
        Compute and average gradients and costs over all the examples.
        """
        avg_cost = 0
        avg_gradient = list(np.zeros(weight.shape) for weight in
            self.network.weights)
        for example in examples:
            gradient, cost = super()._compute(example)
            assert len(gradient) == len(avg_gradient)
            avg_cost += cost / len(examples)
            for index, values in enumerate(gradient):
                avg_gradient[index] += values
        # Normalize by the number of examples.
        cost /= len(examples)
        gradient = list(x / len(examples) for x in avg_gradient)
        return gradient, cost

    def _print_min_max(self):
        # The minimum and maximum gradient values are useful to validate
        # the gradient calculation and understand what the network is doing
        # internally.
        flat = np.hstack(np.array(list(x.flatten() for x in gradient)))
        print('gradient min:', flat.min(), 'max:', flat.max())


class MiniBatchGradientDecent(BatchGradientDecent):

    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 10)
        super().__init__(*args, **kwargs)

    def apply(self, examples):
        for batch in self._batched(examples, self.batch_size):
            yield super().apply(batch)

    def _batched(self, examples, size):
        for i in range(0, len(examples), size):
            yield examples[i:i+size]


class Plot:

    def __init__(self, optimization, refresh=100, smoothing=1):
        self.optimization = optimization
        self.refresh = refresh
        self.smoothing = smoothing
        self.costs = []
        self.smoothed = []
        self._init_plot()

    def apply(self, cost):
        self.costs.append(cost)
        self.smoothed.append(np.mean(self.costs[-self.smoothing:]))
        if len(self.costs) % self.refresh == 0:
            self._refresh()

    def _init_plot(self):
        self.plot, = plt.plot([], [])
        plt.xlabel('Batch')
        plt.ylabel('Cost')
        plt.ion()
        plt.show()

    def _refresh(self):
        range_ = range(len(self.smoothed))
        plt.xlim(xmin=0, xmax=range_[-1])
        plt.ylim(ymin=0, ymax=1.1*max(self.smoothed))
        self.plot.set_data(range_, self.smoothed)
        plt.draw()
        plt.pause(0.001)


def test(network, examples):
    """
    Return the average cost over the examples.
    """
    cost = 0
    for example in examples:
        prediction = network.feed(example.data)
        cost += network.cost.apply(prediction, example.target)
    return cost / len(examples)


def examples_regression(amount, inputs=10):
    data = np.random.rand(amount, inputs)
    products = np.prod(data, axis=1)
    products = products / np.max(products)
    sums = np.sum(data, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])
    return [Example(x, y) for x, y in zip(data, targets)]


def examples_classification(amount, inputs=10, classes=3):
    data = np.random.randint(0, 1000, (amount, inputs))
    mods = np.mod(np.sum(data, axis=1), classes)
    data = data.astype(float) / data.max()
    targets = np.zeros((amount, classes))
    for index, mod in enumerate(mods):
        targets[index][mod] = 1
    return [Example(x, y) for x, y in zip(data, targets)]


if __name__ == '__main__':
    # Generate and split dataset.
    examples = examples_classification(50000)
    split = int(0.8 * len(examples))
    training, testing = examples[:split], examples[split:]
    # Create a network. The input and output layer sizes are derived from the
    # input and target data.
    network = Network([
        Layer(len(training[0].data), Linear),
        Layer(10, Sigmoid),
        Layer(len(training[0].target), Sigmoid)
    ])
    # Training.
    gradient = Backpropagation(network, CrossEntropy)
    # gradient = NumericalGradient(network, CrossEntropy)
    # gradient = CheckedGradient(network, CrossEntropy, Backpropagation)
    optimization = MiniBatchGradientDecent(network, CrossEntropy, gradient,
        learning_rate=1e-2)
    plot = Plot(optimization)
    for cost in optimization.apply(examples):
        plot.apply(cost)
    # Evaluation.
    cost = test(network, testing)
    print('Test set cost:', cost)

