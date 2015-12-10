import math
import functools
import multiprocessing
import numpy as np
from layered.network import Matrices
from layered.utility import batched


class Gradient:

    def __init__(self, network, cost):
        self.network = network
        self.cost = cost

    def __call__(self, weights, example):
        raise NotImplementedError


class Backprop(Gradient):
    """
    Use the backpropagation algorithm to efficiently determine the gradient of
    the cost function with respect to each individual weight.
    """

    def __call__(self, weights, example):
        prediction = self.network.feed(weights, example.data)
        delta_output = self._delta_output(prediction, example.target)
        delta_layers = self._delta_layers(weights, delta_output)
        delta_weights = self._delta_weights(delta_layers)
        return delta_weights

    def _delta_output(self, prediction, target):
        assert len(target) == self.network.layers[-1].size
        # The derivative with respect to the output layer is computed as the
        # product of error derivative and local derivative at the layer.
        delta_cost = self.cost.delta(prediction, target)
        delta_output = self.network.layers[-1].delta(delta_cost)
        assert len(delta_cost) == len(delta_output) == len(target)
        return delta_output

    def _delta_layers(self, weights, delta_output):
        # Propagate backwards through the hidden layers but not the input
        # layer. The current weight matrix is the one to the right of the
        # current layer.
        gradient = [delta_output]
        hidden = list(zip(weights[1:], self.network.layers[1:-1]))
        assert all(x.shape[0] - 1 == len(y) for x, y in hidden)
        for weight, layer in reversed(hidden):
            delta = self._delta_layer(layer, weight, gradient[-1])
            gradient.append(delta)
        return reversed(gradient)

    def _delta_layer(self, layer, weight, above):
        # The gradient at a layer is computed as the derivative of both the
        # local activation and the weighted sum of the derivatives in the
        # deeper layer.
        backward = self.network.backward(weight, above)
        delta = layer.delta(backward)
        assert len(layer) == len(backward) == len(delta)
        return delta

    def _delta_weights(self, delta_layers):
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        gradient = Matrices(self.network.shapes)
        prev_and_delta = zip(self.network.layers[:-1], delta_layers)
        for index, (previous, delta) in enumerate(prev_and_delta):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            activations = np.insert(previous.outgoing, 0, 1)
            assert activations[0] == 1
            gradient[index] = np.outer(activations, delta)
        return gradient


class NumericalGradient(Gradient):
    """
    Approximate the gradient for each weight individually by sampling the error
    function slightly above and below the current value of the weight.
    """

    def __init__(self, network, cost, distance=1e-5):
        super().__init__(network, cost)
        self.distance = distance

    def __call__(self, weights, example):
        """
        Modify each weight individually in both directions to calculate a
        numeric gradient of the weights.
        """
        # We need a copy of the weights that we can modify to evaluate the cost
        # function on.
        modified = Matrices(weights.shapes, weights.flat.copy())
        gradient = Matrices(weights.shapes)
        for i, connection in enumerate(weights):
            for j, original in np.ndenumerate(connection):
                # Sample above and below and compute costs.
                modified[i][j] = original + self.distance
                above = self._evaluate(modified, example)
                modified[i][j] = original - self.distance
                below = self._evaluate(modified, example)
                # Restore the original value so we can reuse the weight matrix
                # for the next iteration.
                modified[i][j] = original
                # Compute the numeric gradient.
                sample = (above - below) / (2 * self.distance)
                gradient[i][j] = sample
        return gradient

    def _evaluate(self, weights, example):
        prediction = self.network.feed(weights, example.data)
        cost = self.cost(prediction, example.target)
        assert cost.shape == prediction.shape
        return cost.sum()


class CheckedBackprop(Gradient):
    """
    Computes the gradient both analytically trough backpropagation and
    numerically to validate the backpropagation implementation and derivatives
    of activation functions and cost functions. This is slow by its nature and
    it's recommended to validate derivatives on small networks.
    """

    def __init__(self, network, cost, distance=1e-5, tolerance=1e-8):
        self.tolerance = tolerance
        super().__init__(network, cost)
        self.analytic = Backprop(network, cost)
        self.numeric = NumericalGradient(network, cost, distance)

    def __call__(self, weights, example):
        analytic = self.analytic(weights, example)
        numeric = self.numeric(weights, example)
        distances = np.absolute(analytic.flat - numeric.flat)
        worst = distances.max()
        if worst > self.tolerance:
            print('Gradient differs by {:.2f}%'.format(100 * worst))
        else:
            print('Gradient looks good')
        return analytic


class BatchBackprop:
    """
    Calculate the average gradient over a batch of examples.
    """

    def __init__(self, network, cost):
        self.backprop = Backprop(network, cost)

    def __call__(self, weights, examples):
        gradient = Matrices(weights.shapes)
        for example in examples:
            gradient += self.backprop(weights, example)
        return gradient / len(examples)


class ParallelBackprop:
    """
    Alternative to BatchBackprop that yields the same results but utilizes
    multiprocessing to make use of more than one processor core.
    """

    def __init__(self, network, cost, workers=4):
        self.backprop = BatchBackprop(network, cost)
        self.workers = workers
        self.pool = multiprocessing.Pool(self.workers)

    def __call__(self, weights, examples):
        batch_size = int(math.ceil(len(examples) / self.workers))
        batches = list(batched(examples, batch_size))
        sizes = [len(x) / batch_size for x in batches]
        sizes = [x / sum(sizes) for x in sizes]
        assert len(batches) <= self.workers
        assert sum(sizes) == 1
        compute = functools.partial(self.backprop, weights)
        gradients = self.pool.map(compute, batches)
        return sum(x * y for x, y in zip(gradients, sizes))
