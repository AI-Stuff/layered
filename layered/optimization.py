class GradientDecent:
    """
    Adapt the weights in the opposite direction of the gradient to reduce the
    error.
    """

    def __call__(self, weights, gradient, learning_rate=0.1):
        return weights - learning_rate * gradient


class Momentum:
    """
    Slow down changes of direction in the gradient by aggregating previous
    values of the gradient and multiplying them in.
    """

    def __init__(self):
        self.previous = None

    def __call__(self, gradient, rate=0.9):
        if self.previous is None:
            self.previous = gradient.copy()
        gradient = rate * self.previous + gradient
        self.previous = gradient
        return gradient.copy()


class WeightDecay:
    """
    Slowly moves each weight closer to zero for regularization. This can help
    the model to find simpler solutions.
    """

    def __call__(self, weights, rate=1e-4):
        return (1 - rate) * weights
