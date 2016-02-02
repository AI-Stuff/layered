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


class SparseField(Activation):

    def __init__(self, inhibition=0.05, leaking=0.0):
        self.inhibition = inhibition
        self.leaking = leaking

    def __call__(self, incoming):
        count = len(incoming)
        length = int(np.sqrt(count))
        assert length ** 2 == count, 'layer size must be a square'
        field = incoming.copy().reshape((length, length))
        radius = int(np.sqrt(self.inhibition * count)) // 2
        assert radius, 'no inhibition due to small factor'
        outgoing = np.zeros(field.shape)
        while True:
            x, y = np.unravel_index(field.argmax(), field.shape)
            if field[x, y] <= 0:
                break
            outgoing[x, y] = 1
            surrounding = np.s_[
                max(x - radius, 0):min(x + radius + 1, length),
                max(y - radius, 0):min(y + radius + 1, length)]
            field[surrounding] = 0
            assert field[x, y] == 0
        outgoing = outgoing.reshape(count)
        outgoing = np.maximum(outgoing, self.leaking * incoming)
        return outgoing

    def delta(self, incoming, outgoing, above):
        delta = np.greater(outgoing, 0).astype(float)
        return delta * above


class SparseRange(Activation):
    """
    E%-Max Winner-Take-All.

    Binary activation. First, the activation function is applied. Then all
    neurons within the specified range below the strongest neuron are set to
    one. All others are set to zero. The gradient is the one of the activation
    function for active neurons and zero otherwise.

    See: A Second Function of Gamma Frequency Oscillations: An E%-Max
    Winner-Take-All Mechanism Selects Which Cells Fire. (2009)
    """

    def __init__(self, range_=0.3, function=Sigmoid()):
        assert 0 < range_ < 1
        self._range = range_
        self._function = function

    def __call__(self, incoming):
        incoming = self._function(incoming)
        threshold = self._threshold(incoming)
        active = (incoming >= threshold)
        outgoing = np.zeros(incoming.shape)
        outgoing[active] = 1
        # width = active.sum() * 80 / 1000
        # print('|', '#' * width, ' ' * (80 - width), '|')
        return outgoing

    def delta(self, incoming, outgoing, above):
        # return self._function.delta(incoming, outgoing, outgoing * above)
        return outgoing * self._function.delta(incoming, outgoing, above)

    def _threshold(self, incoming):
        min_, max_ = incoming.min(), incoming.max()
        threshold = min_ + (max_ - min_) * (1 - self._range)
        return threshold
