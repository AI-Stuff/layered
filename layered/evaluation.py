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
        error = 1 - self._accuracy(weights)
        print('Batch {} test error {:.2f}%'.format(index, 100 * error))

    def _accuracy(self, weights):
        predicts = lambda x: float(self._predicts(weights, x))
        return averaged(predicts, self.examples)

    def _predicts(self, weights, example):
        prediction = self.network.feed(weights, example.data)
        return np.argmax(prediction) == np.argmax(example.target)
