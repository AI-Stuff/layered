import numpy as np


def compute_costs(network, weights, cost, examples):
    prediction = [network.feed(weights, x.data) for x in examples]
    costs = [cost(x, y.target).mean() for x, y in zip(prediction, examples)]
    return costs


def compute_error(network, weights, examples):
    prediction = [network.feed(weights, x.data) for x in examples]
    error = sum(bool(np.argmax(x) != np.argmax(y.target)) for x, y in
                zip(prediction, examples)) / len(examples)
    return error
