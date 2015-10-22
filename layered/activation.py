import numpy as np


class Activation:

    @staticmethod
    def apply(incoming):
        raise NotImplemented

    @staticmethod
    def delta(incoming, outgoing):
        raise NotImplemented


class Sigmoid(Activation):

    @staticmethod
    def apply(incoming):
        return 1 / (1 + np.exp(-incoming))

    @staticmethod
    def delta(incoming, outgoing):
        return outgoing * (1 - outgoing)


class Linear(Activation):

    @staticmethod
    def apply(incoming):
        return incoming

    @staticmethod
    def delta(incoming, outgoing):
        return np.ones(incoming.shape).astype(float)


class Relu(Activation):

    @staticmethod
    def apply(incoming):
        return np.maximum(incoming, 0)

    @staticmethod
    def delta(incoming, outgoing):
        return np.greater(incoming, 0).astype(float)


class Softmax(Activation):

    @staticmethod
    def apply(incoming):
        exp = np.exp(incoming)
        return exp / np.sum(exp)

    @staticmethod
    def delta(incoming, outgoing):
        others = np.dot(-outgoing, outgoing) - (-outgoing * outgoing)
        current = outgoing * (1 - outgoing)
        return others + current
