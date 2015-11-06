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
        exps = np.exp(incoming)
        return exps / exps.sum()

    @staticmethod
    def delta_1(incoming, outgoing):
        others = np.dot(-outgoing, outgoing) - (-outgoing * outgoing)
        current = outgoing * (1 - outgoing)
        return others + current

    @staticmethod
    def delta_2(incoming, outgoing):
        exps = np.exp(incoming)
        others = exps.sum() - exps
        return 1 / (2 + exps / others + others / exps)
