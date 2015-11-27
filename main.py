import os
import argparse
import numpy as np
from layered.problem import Problem
from layered.gradient import BatchBackprop
from layered.network import Network, Matrices
from layered.optimization import GradientDecent, Momentum, WeightDecay
from layered.utility import repeated, batched
from layered.evaluation import compute_costs, compute_error


def every(times, step_size, index):
    """
    Given a loop over batches of an iterable and an operation that should be
    performed every few elements. Determine whether the operation should be
    called for the current index.
    """
    current = index * step_size
    step = current // times * times
    reached = current >= step
    overshot = current >= step + step_size
    return current and reached and not overshot


if __name__ == '__main__':
    # The problem defines dataset, network and learning parameters
    parser = argparse.ArgumentParser('layered')
    parser.add_argument(
        'problem', nargs='?',
        help='path to the YAML problem definition')
    parser.add_argument(
        '-v', '--visual', action='store_true',
        help='show a diagram of training costs')
    args = parser.parse_args()
    print('Problem', os.path.split(args.problem)[1])
    problem = Problem(args.problem)

    # Define model and initialize weights
    network = Network(problem.layers)
    weights = Matrices(network.shapes)
    weights.flat = np.random.normal(0, problem.weight_scale, len(weights.flat))

    # Classes needed during training
    backprop = BatchBackprop(network, problem.cost)
    momentum = Momentum()
    decent = GradientDecent()
    decay = WeightDecay()
    if args.visual:
        from layered.plot import RunningPlot
        plot_training = RunningPlot()

    # Train the model
    repeats = repeated(problem.dataset.training, problem.epochs)
    batches = batched(repeats, problem.batch_size)
    for index, batch in enumerate(batches):
        gradient = backprop(weights, batch)
        gradient = momentum(gradient, problem.momentum)
        weights = decent(weights, gradient, problem.learning_rate)
        weights = decay(weights, problem.weight_decay)
        # Show progress
        if args.visual:
            costs = compute_costs(network, weights, problem.cost, batch)
            plot_training(costs)
        if every(problem.evaluate_every, problem.batch_size, index):
            error = compute_error(
                    network, weights, problem.cost, problem.dataset.testing)
            print('Batch {} test error {:.2f}%'.format(index, 100 * error))
    print('Done')
