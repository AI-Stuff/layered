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

    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def __call__(self, prediction, target):
        clipped = np.clip(prediction, self.epsilon, 1 - self.epsilon)
        cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
        return -cost

    def delta(self, prediction, target):
        denominator = np.maximum(prediction - prediction ** 2, self.epsilon)
        delta = (prediction - target) / denominator
        assert delta.shape == target.shape == prediction.shape
        return delta
