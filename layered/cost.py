import numpy as np


class Cost:

    @staticmethod
    def apply(prediction, target):
        raise NotImplemented

    @staticmethod
    def delta(prediction, target):
        raise NotImplemented


class SquaredErrors(Cost):

    @staticmethod
    def apply(prediction, target):
        return np.sum(np.square(prediction - target)) / 2

    @staticmethod
    def delta(prediction, target):
        return prediction - target


class CrossEntropy(Cost):

    @staticmethod
    def apply(prediction, target):
        return np.sum(-target * np.log(prediction))

    @staticmethod
    def delta(prediction, target):
        return target-prediction
