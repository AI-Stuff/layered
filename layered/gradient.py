import numpy as np
from layered.network import Network, Matrices
from layered.cost import Cost


class Gradient:

    def __init__(self, network, cost):
        self.network = network
        self.cost = cost

    def __call__(self, weights, example):
        raise NotImplemented


class Backpropagation(Gradient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, weights, example):
        prediction = self.network.feed(weights, example.data)
        delta_layers = self._delta_layers(weights, prediction, example.target)
        delta_weights = self._delta_weights(delta_layers)
        return delta_weights

    def _delta_layers(self, weights, prediction, target):
        assert len(target) == self.network.layers[-1].size
        # We start with the gradient at the output layer.
        gradient = [self._delta_output(prediction, target)]
        # Propagate backwards through the hidden layers but not the input
        # layer. The current weight matrix is the one to the right of the
        # current layer.
        hidden = list(zip(weights[1:], self.network.layers[1:-1]))
        assert all(x.shape[0] - 1 == len(y) for x, y in hidden)
        for weight, layer in reversed(hidden):
            gradient.append(self._delta_hidden(layer, weight, gradient[-1]))
        return reversed(gradient)

    def _delta_output(self, prediction, target):
        # The derivative with respect to the output layer is computed as the
        # product of error derivative and local derivative at the layer.
        cost = self.cost.delta(prediction, target)
        local = self.network.layers[-1].delta()
        assert len(cost) == len(local)
        return cost * local

    def _delta_hidden(self, layer, weight, delta_right):
        # The gradient at a layer is computed as the derivative of both the
        # local activation and the weighted sum of the derivatives in the
        # deeper layer.
        backward = self.network.backward(weight, delta_right)
        local = layer.delta()
        assert len(layer) == len(backward) == len(local)
        return backward * local

    def _delta_weights(self, delta_layers):
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        gradient = Matrices(self.network.shapes)
        prev_and_delta = zip(self.network.layers[:-1], delta_layers)
        for index, (previous, delta) in enumerate(prev_and_delta):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            bias_and_activation = np.insert(previous.outgoing, 0, 1)
            assert bias_and_activation[0] == 1
            gradient[index] = np.outer(bias_and_activation, delta)
        return gradient


class NumericalGradient(Gradient):

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


class CheckedBackpropagation(Gradient):

    def __init__(self, network, cost, distance=1e-5, tolerance=1e-8):
        self.tolerance = tolerance
        super().__init__(network, cost)
        self.analytic = Backpropagation(network, cost)
        self.numeric = NumericalGradient(network, cost, distance)

    def __call__(self, weights, example):
        analytic = self.analytic(weights, example)
        numeric = self.numeric(weights, example)
        # Flatten the gradients so that we can compare their elements.
        analytic_flat = self._flatten(analytic)
        numeric_flat = self._flatten(numeric)
        distances = np.absolute(analytic_flat - numeric_flat)
        worst = distances.max() / np.absolute(numeric_flat).max()
        if worst > self.tolerance:
            print('Gradient differs by {:.2f}%'.format(100 * worst))
        else:
            print('Gradient looks good')
        return analytic

    def _flatten(self, gradient):
        return np.hstack(np.array(list(x.flatten() for x in gradient)))
