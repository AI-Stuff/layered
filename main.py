import numpy as np
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.example import Example
from layered.gradient import BatchBackprop, ParallelBackprop
from layered.network import Network, Layer, Matrices
from layered.optimization import GradientDecent, Momentum, WeightDecay
from layered.plot import Plot
from layered.utility import repeated, batched, averaged, listify
from layered.dataset import Regression, Classification, Mnist


class Problem:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def compute_error(network, weights, examples):
    return averaged(examples, lambda x:
        float(np.argmax(x.target) !=
        np.argmax(network.feed(weights, x.data))))


def compute_costs(network, weights, cost, examples):
    prediction = [network.feed(weights, x.data) for x in examples]
    costs = [cost(x, y.target).mean() for x, y in zip(prediction, examples)]
    return list(costs)


def every(every, index, callable_, *args, **kwargs):
    if (index + 1) % every == 0:
        callable_(*args, **kwargs)


def evaluate(index, network, weights, testing):
    error = 100 * compute_error(network, weights, testing)
    print('Batch {} test error {:.2f}%'.format(index + 1, error))


if __name__ == '__main__':
    # Define the problem
    problem = Problem(
        training_rounds=20,
        batch_size=100,
        learning_rate=1.2,
        momentum=0.3,
        weight_scale=0.01,
        weight_decay=1e-3,
        evaluate_every=5000,
        dataset=Mnist(),
        cost=Squared())

    # Define model and initialize weights
    network = Network([
        Layer(len(problem.dataset.training[0].data), Linear),
        Layer(700, Relu),
        Layer(500, Relu),
        Layer(300, Relu),
        Layer(len(problem.dataset.training[0].target), Sigmoid)
    ])
    weights = Matrices(network.shapes)
    weights.flat = np.random.normal(0, problem.weight_scale, len(weights.flat))

    # Classes needed during training
    backprop = ParallelBackprop(network, problem.cost)
    momentum = Momentum()
    decent = GradientDecent()
    decay = WeightDecay()
    plot = Plot()

    # Train the model
    repeats = repeated(problem.dataset.training, problem.training_rounds)
    batches = batched(repeats, problem.batch_size)
    for index, batch in enumerate(batches):
        gradient = backprop(weights, batch)
        gradient = momentum(gradient, problem.momentum)
        weights = decent(weights, gradient, problem.learning_rate)
        weights = decay(weights, problem.weight_decay)
        # Show progress
        plot(compute_costs(network, weights, problem.cost, batch))
        every(problem.evaluate_every // problem.batch_size, index, evaluate,
            index, network, weights, problem.dataset.testing)
