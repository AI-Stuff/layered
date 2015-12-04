import numpy as np


class Distribution:

    def __call__(self, shape, variance):
        raise NotImplementedError


class Normal(Distribution):

    def __call__(self, shape, variance):
        std = np.sqrt(variance)
        return np.random.normal(0, std, shape)


class InitWeights:

    def __call__(self, weights):
        raise NotImplementedError


class Random(InitWeights):
    """
    Initialize the weights with random values, all with the same variance.
    """

    def __init__(self, distribution=Normal(), variance=0.01):
        self.distribution = distribution
        self.variance = variance

    def __call__(self, weights):
        weights.flat = self.distribution(len(weights.flat), self.variance)


class Xavier(InitWeights):
    """
    Xavier initialization draws from a distribution with zero mean
    and variance = 2 / (|in| + |out|) so that the variance of the
    signal passed trough the layers stays roughly the same in both
    forward and backward passes.
    """

    def __init__(self, distribution=Normal()):
        self.distribution = distribution

    def __call__(self, weights):
        for index, matrix in enumerate(weights):
            variance = len(matrix.shape) / sum(matrix.shape)
            weights[index] = self.distribution(matrix.shape, np.sqrt(variance))
