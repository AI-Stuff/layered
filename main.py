import numpy as np
from layered.example import Example
from layered.network import Network, Layer
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.optimization import MiniBatchGradientDecent
from layered.gradient import Backpropagation, CheckedGradient
from layered.plot import Plot
from layered.dataset import Regression, Classification, Mnist


def evaluation(network, testing, cost):
    predictions = [network.feed(x.data) for x in testing]
    pairs = [(x, y.target) for x, y in zip(predictions, testing)]
    error = sum(np.argmax(x) != np.argmax(y) for x, y in pairs) / len(testing)
    print('Test error {:.2f}%'.format(100 * error))


if __name__ == '__main__':
    print('Loading dataset')
    dataset = Mnist()

    # Create a network. The input and output layer sizes are derived from the
    # input and target data.
    network = Network([
        Layer(len(dataset.training[0].data), Linear),
        Layer(700, Relu),
        Layer(300, Relu),
        Layer(len(dataset.training[0].target), Sigmoid)
    ])

    cost = Squared
    gradient = Backpropagation(network, cost)
    # gradient = CheckedGradient(network, cost, Backpropagation)
    optimization = MiniBatchGradientDecent(network, cost, gradient,
        learning_rate=5e-2, batch_size=2)
    plot = Plot(optimization)

    print('Start training')
    for round_ in range(10):
        print('Round', round_)
        for error in optimization.apply(dataset.training):
            plot.apply(error)
        evaluation(network, dataset.testing, cost())
