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
        gradient = gradient.copy()
        if self.previous is None:
            self.previous = gradient.copy()
        else:
            assert self.previous.shape == gradient.shape
            gradient += rate * self.previous
            self.previous = gradient.copy()
        return gradient


class WeightDecay:
    """
    Slowly moves each weight closer to zero for regularization. This can help
    the model to find simpler solutions.
    """

    def __call__(self, weights, rate=1e-4):
        return (1 - rate) * weights


class WeightTying:
    """
    Constraint groups of slices of the gradient to have the same value by
    averaging them. Should be applied to the initial weights and each gradient.
    """

    def __init__(self, *groups):
        for group in groups:
            assert group and hasattr(group, '__len__')
            assert all([isinstance(x[0], int) for x in group])
            assert all([isinstance(y, (slice, int)) for x in group for y in x])
        self.groups = groups

    def __call__(self, matrices):
        matrices = matrices.copy()
        for group in self.groups:
            slices = [matrices[slice_] for slice_ in group]
            assert all([x.shape == slices[0].shape for x in slices]), (
                'All slices within a group must have the same shape. '
                'Shapes are ' + ', '.join(str(x.shape) for x in slices) + '.')
            average = sum(slices) / len(slices)
            assert average.shape == slices[0].shape
            for slice_ in group:
                matrices[slice_] = average
        return matrices
