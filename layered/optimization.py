import numpy as np
from layered.network import Network, Matrices
from layered.gradient import Gradient


class GradientDecent:

    def __call__(self, weights, gradient, learning_rate=0.1):
        return weights - learning_rate * gradient
