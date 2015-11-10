import numpy as np
from layered.utility import averaged


class Evaluator:

    def __init__(self, network, examples, every=1000):
        self.network = network
        self.examples = examples
        self.every = every

    def __call__(self, index, weights):
        if (index + 1) % self.every > 0:
            return
        error = 100 * self._error(weights)
        print('Batch {} test error {:.2f}%'.format(index + 1, error))

    def _error(self, weights):
        return averaged(self.examples, lambda x:
            float(np.argmax(x.target) !=
            np.argmax(self.network.feed(weights, x.data))))
