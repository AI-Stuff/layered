import numpy as np
from layered.network import Network
from layered.gradient import Gradient


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
