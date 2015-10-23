import numpy as np


class Cost:

    @staticmethod
    def apply(prediction, target):
        raise NotImplemented

    @staticmethod
    def delta(prediction, target):
        raise NotImplemented


class Squared(Cost):

    @staticmethod
    def apply(prediction, target):
        return (prediction - target) ** 2 / 2

    @staticmethod
    def delta(prediction, target):
        return prediction - target


class CrossEntropy(Cost):

    @staticmethod
    def apply(prediction, target):
        epsilon = 1e-11
        clipped = np.clip(prediction, epsilon, 1 - epsilon)
        cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
        return -cost

    @staticmethod
    def delta(prediction, target):
        delta = (prediction - target) / (prediction - prediction ** 2)
        assert delta.shape == target.shape == prediction.shape
        return delta
