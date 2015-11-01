import numpy as np
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.example import Example
from layered.gradient import Backpropagation, CheckedBackpropagation
from layered.network import Network, Layer, Matrices
from layered.optimization import GradientDecent
from layered.plot import Plot
from layered.utility import repeat, batched, average, listify
from layered.dataset import Regression, Classification, Mnist


class Problem:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def compute_error(network, weights, examples):
    return average(examples, lambda x:
        float(np.argmax(x.target) !=
        np.argmax(network.feed(weights, x.data))))


def compute_costs(network, weights, cost, examples):
    prediction = [network.feed(weights, x.data) for x in examples]
    costs = [cost(x, y.target).mean() for x, y in zip(prediction, examples)]
    return list(costs)


def evaluate_every(index, problem, network, weights, testing):
    if (index + 1) % (problem.evaluate_every // problem.batch_size) != 0:
        return
    error = 100 * compute_error(network, weights, testing)
    print('Batch {} test error {:.2f}%'.format(index + 1, error))


if __name__ == '__main__':
    # Define the problem
    problem = Problem(training_rounds=5, batch_size=2, learning_rate=0.05,
        evaluate_every=5000, weight_scale=0.1, dataset=Mnist(),
        cost=Squared())

    # Define model and initialize weights
    network = Network([
        Layer(len(problem.dataset.training[0].data), Linear),
        Layer(700, Relu),
        Layer(300, Relu),
        Layer(len(problem.dataset.training[0].target), Sigmoid)
    ])
    weights = Matrices(network.shapes)
    weights.flat = np.random.normal(0, problem.weight_scale, len(weights.flat))

    # Classes needed during training
    backprop = Backpropagation(network, problem.cost)
    decent = GradientDecent()
    plot = Plot()

    # Train the model
    examples = repeat(problem.dataset.training, problem.training_rounds)
    batches = batched(examples, problem.batch_size)
    testing = problem.dataset.testing
    for index, batch in enumerate(batches):
        gradient = average(batch, lambda x: backprop(weights, x))
        weights = decent(weights, gradient, problem.learning_rate)
        plot(compute_costs(network, weights, problem.cost, batch))
        evaluate_every(index, problem, network, weights, testing)
