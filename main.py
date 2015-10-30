import numpy as np
from layered.example import Example
from layered.network import Network, Layer, Matrices
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.optimization import GradientDecent
from layered.gradient import Backpropagation, CheckedBackpropagation
from layered.plot import Plot
from layered.dataset import Regression, Classification, Mnist


def evaluation(network, weights, testing):
    predictions = [network.feed(weights, x.data) for x in testing]
    pairs = [(x, y.target) for x, y in zip(predictions, testing)]
    error = sum(np.argmax(x) != np.argmax(y) for x, y in pairs) / len(testing)
    print('Test error {:.2f}%'.format(100 * error))


def batched(iterable, size):
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) == size:
            yield batch
            batch = []
    yield batch


if __name__ == '__main__':
    print('Loading dataset')
    dataset = Mnist()

    network = Network([
        Layer(len(dataset.training[0].data), Linear),
        Layer(700, Relu),
        Layer(300, Relu),
        Layer(len(dataset.training[0].target), Sigmoid)
    ])
    weights = Matrices(network.shapes)

    cost = Squared()
    backprop = Backpropagation(network, cost)
    decent = GradientDecent()
    plot = Plot()

    print('Start training')
    for round_ in range(10):
        print('Round', round_)
        for example in dataset.training:
            gradient = backprop(weights, example)
            weights = decent(weights, gradient, learning_rate=0.1)
            plot(cost(network.feed(weights, example.data), example.target))
        evaluation(network, weights, dataset.testing)
