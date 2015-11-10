import numpy as np
from layered.utility import averaged


class Evaluator:

    def __init__(self, network, examples, every=1000):
        self.network = network
        self.examples = examples
        self.every = every
        self.last_index = 0

    def __call__(self, index, weights):
        if index < self.last_index + self.every:
            return
        self.last_index = index
        error = 100 - 100 * self._accuracy(weights)
        print('Batch {} test error {:.2f}%'.format(index + 1, error))

    def _accuracy(self, weights):
        return averaged(self.examples, lambda x: self._predicts(weights, x))

    def _predicts(self, weights, example):
        target = np.argmax(example.target)
        prediction = np.argmax(self.network.feed(weights, example.data))
        return target == prediction
