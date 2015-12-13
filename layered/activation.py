import numpy as np


class Activation:

    def __call__(self, incoming):
        raise NotImplementedError

    def delta(self, incoming, outgoing, above):
        """
        Compute the derivative of the cost with respect to the input of this
        activation function. Outgoing is what this function returned in the
        forward pass and above is the derivative of the cost with respect to
        the outgoing activation.
        """
        raise NotImplementedError


class Identity(Activation):

    def __call__(self, incoming):
        return incoming

    def delta(self, incoming, outgoing, above):
        delta = np.ones(incoming.shape).astype(float)
        return delta * above


class Sigmoid(Activation):

    def __call__(self, incoming):
        return 1 / (1 + np.exp(-incoming))

    def delta(self, incoming, outgoing, above):
        delta = outgoing * (1 - outgoing)
        return delta * above


class Relu(Activation):

    def __call__(self, incoming):
        return np.maximum(incoming, 0)

    def delta(self, incoming, outgoing, above):
        delta = np.greater(incoming, 0).astype(float)
        return delta * above


class Softmax(Activation):

    def __call__(self, incoming):
        # The constant doesn't change the expression but prevents overflows.
        constant = np.max(incoming)
        exps = np.exp(incoming - constant)
        return exps / exps.sum()

    def delta(self, incoming, outgoing, above):
        delta = outgoing * above
        sum_ = delta.sum(axis=delta.ndim - 1, keepdims=True)
        delta -= outgoing * sum_
        return delta
