import numpy as np
from layered.network import Network
from layered.cost import Cost


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
        layers = self._layers(target)
        weights = self._weights(layers)
        return weights

    def _layers(self, target):
        assert len(target) == self.network.layers[-1].size
        # We start with the gradient at the output layer. It's computed as the
        # product of error derivative and local derivative at the last layer.
        prediction = self.network.layers[-1].outgoing
        cost = self.cost.delta(prediction, target)
        local = self.network.layers[-1].delta()
        assert len(cost) == len(local)
        gradient = [cost * local]
        # Propagate backwards through the hidden layers but not the input
        # layer.  The current weight matrix is the one to the right of the
        # current layer.
        hidden = list(zip(self.network.weights[1:], self.network.layers[1:-1]))
        assert all(x.shape[0] - 1 == len(y) for x, y in hidden)
        for weight, layer in reversed(hidden):
            # The gradient at a layer is computed as the derivative of both the
            # local activation and the weighted sum of the derivatives in the
            # deeper layer.
            backward = weight.backward(gradient[-1])
            local = layer.delta()
            assert len(layer) == len(backward) == len(local)
            gradient.append(backward * local)
        gradient = list(reversed(gradient))
        assert len(gradient) == len(self.network.layers) - 1
        # We computed the gradient at the hidden layers and the output layer.
        assert len(gradient) == len(self.network.layers) - 1
        assert all(len(x) == y.size for x, y in
            zip(gradient, self.network.layers[1:]))
        return gradient

    def _weights(self, delta_layers):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for previous, delta in zip(self.network.layers[:-1], delta_layers):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            bias_and_activation = np.insert(previous.outgoing, 0, 1)
            assert bias_and_activation[0] == 1
            gradient.append(np.outer(bias_and_activation, delta))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(self.network.weights)
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
        for i, connection in enumerate(self.network.weights):
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
                sample = (above - below) / (2 * self.distance)
                gradient[i][j] = sample
        return gradient

    def _evaluate(self, weights, target):
        prediction = self.network.forward(weights)
        cost = self.cost.apply(prediction, target)
        assert cost.shape == prediction.shape == target.shape
        return cost.sum()


class CheckedGradient(Gradient):

    def __init__(self, *args, **kwargs):
        self.tolerance = kwargs.pop('tolerance', 1e-8)
        distance = kwargs.pop('distance', 1e-5)
        super().__init__(*args[:-1], **kwargs)
        GradientClass = args[-1]
        assert issubclass(GradientClass, Gradient)
        self.gradient = GradientClass(self.network, args[1])
        self.numerical = NumericalGradient(self.network, args[1],
            distance=distance)

    def apply(self, target):
        gradient = self.gradient.apply(target)
        computed = self._flatten(gradient)
        numerical = self._flatten(self.numerical.apply(target))
        distances = np.absolute(computed - numerical)
        worst = distances.max() / np.absolute(numerical).max()
        if worst > self.tolerance:
            print('Gradient differs by {:.2f}%'.format(100 * worst))
        else:
            print('Gradient looks good')
        return gradient

    def _flatten(self, gradient):
        return np.hstack(np.array(list(x.flatten() for x in gradient)))
