import numpy as np


class Cost:

    def __call__(self, prediction, target):
        raise NotImplementedError

    def delta(self, prediction, target):
        raise NotImplementedError


class SquaredError(Cost):
    """
    Fast and simple cost function.
    """

    def __call__(self, prediction, target):
        return (prediction - target) ** 2 / 2

    def delta(self, prediction, target):
        return prediction - target


class CrossEntropy(Cost):
    """
    Logistic cost function used for classification tasks. Learns faster in the
    beginning than SquaredError because large errors are penalized
    exponentially. This makes sense in classification since only the best class
    will be the predicted one.
    """

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
