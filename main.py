import argparse
import numpy as np
from layered.problem import Problem
from layered.gradient import BatchBackprop, ParallelBackprop, CheckedBackprop
from layered.network import Network, Matrices
from layered.optimization import GradientDecent, Momentum, WeightDecay
from layered.plot import Plot
from layered.utility import repeated, batched, averaged, listify
from layered.dataset import Regression, Classification, Mnist


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
    # The problem defines dataset, network and learning parameters
    parser = argparse.ArgumentParser('layered')
    parser.add_argument('problem',
        nargs='?', default='problem/example.yaml',
        help='path to the YAML problem definition')
    parser.add_argument('-n', '--no-visual',
        dest='visual', action='store_false',
        help='show a diagram of training costs')
    args = parser.parse_args()
    problem = Problem(args.problem)

    # Define model and initialize weights
    network = Network(problem.layers)
    weights = Matrices(network.shapes)
    weights.flat = np.random.normal(0, problem.weight_scale, len(weights.flat))

    # Classes needed during training
    backprop = ParallelBackprop(network, problem.cost)
    momentum = Momentum()
    decent = GradientDecent()
    decay = WeightDecay()
    if args.visual:
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
        if args.visual:
            plot(compute_costs(network, weights, problem.cost, batch))
        every(problem.evaluate_every // problem.batch_size, index, evaluate,
            index, network, weights, problem.dataset.testing)
