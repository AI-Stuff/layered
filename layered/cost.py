import numpy as np


class Cost:

    def __call__(self, prediction, target):
        raise NotImplemented

    def delta(self, prediction, target):
        raise NotImplemented


class Squared(Cost):

    def __call__(self, prediction, target):
        return (prediction - target) ** 2 / 2

    def delta(self, prediction, target):
        return prediction - target


class CrossEntropy(Cost):

    def __call__(self, prediction, target):
        epsilon = 1e-11
        clipped = np.clip(prediction, epsilon, 1 - epsilon)
        cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
        return -cost

    def delta(self, prediction, target):
        delta = (prediction - target) / (prediction - prediction ** 2)
        assert delta.shape == target.shape == prediction.shape
        return delta
