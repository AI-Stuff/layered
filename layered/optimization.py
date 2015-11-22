import numpy as np
from layered.network import Matrices
from layered.gradient import Gradient


class GradientDecent:

    def __call__(self, weights, gradient, learning_rate=0.1):
        return weights - learning_rate * gradient


class Momentum:

    def __init__(self):
        self.previous = None

    def __call__(self, gradient, rate=0.9):
        if self.previous is None:
            self.previous = gradient.copy()
        gradient = rate * self.previous + gradient
        self.previous = gradient
        return gradient.copy()


class WeightDecay:

    def __call__(self, weights, rate=1e-4):
        return (1 - rate) * weights
