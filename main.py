import numpy as np
from layered.core import Example
from layered.network import Network, Layer
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import SquaredErrors, CrossEntropy
from layered.optimization import MiniBatchGradientDecent
from layered.gradient import Backpropagation, CheckedGradient
from layered.utility import examples_regression, examples_classification
from layered.plot import Plot


if __name__ == '__main__':
    # Generate and split dataset.
    examples = examples_regression(50000)
    split = int(0.8 * len(examples))
    training, testing = examples[:split], examples[split:]

    # Create a network. The input and output layer sizes are derived from the
    # input and target data.
    network = Network([
        Layer(len(training[0].data), Linear),
        Layer(10, Sigmoid),
        Layer(10, Sigmoid),
        Layer(len(training[0].target), Sigmoid)
    ])

    # Training.
    cost = SquaredErrors
    gradient = Backpropagation(network, cost)
    # gradient = CheckedGradient(network, cost, Backpropagation)
    optimization = MiniBatchGradientDecent(network, cost, gradient,
        learning_rate=1e-2)
    plot = Plot(optimization)
    for error in optimization.apply(examples):
        plot.apply(error)

    # Evaluation.
    predictions = [network.feed(x.data) for x in testing]
    errors = [cost().apply(x, y.target) for x, y in zip(predictions, testing)]
    print('Test set performance:', sum(errors) / len(testing))

