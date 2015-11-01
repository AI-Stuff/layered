import numpy as np
from layered.example import Example
from layered.network import Network, Layer, Matrices
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.optimization import GradientDecent
from layered.gradient import Backpropagation, CheckedBackpropagation
from layered.plot import Plot
from layered.dataset import Regression, Classification, Mnist
from layered.utility import repeat, batched, average


def compute_error(network, weights, examples):
    return average(examples, lambda x:
        float(np.argmax(x.target) !=
        np.argmax(network.feed(weights, x.data))))


def compute_cost(network, weights, examples):
    return average(batch, lambda x:
        cost(network.feed(weights, x.data), x.target).mean())


if __name__ == '__main__':
    print('Load dataset')
    dataset = Mnist()

    network = Network([
        Layer(len(dataset.training[0].data), Linear),
        Layer(700, Relu),
        Layer(300, Relu),
        Layer(len(dataset.training[0].target), Sigmoid)
    ])

    num_weights = sum(x * y for x, y in network.shapes)
    weights = Matrices(network.shapes, np.random.normal(0, 0.1, num_weights))

    cost = Squared()
    backprop = Backpropagation(network, cost)
    decent = GradientDecent()
    plot = Plot()

    print('Start training')
    examples = repeat(dataset.training, 5)
    for index, batch in enumerate(batched(examples, 2)):
        gradient = average(batch, lambda x: backprop(weights, x))
        weights = decent(weights, gradient, learning_rate=0.05)
        for example in batch:
            plot(cost(network.feed(weights, example.data), example.target))
        # plot(compute_cost(network, weights, batch))
        if (index + 1) % 1000 == 0:
            print('Batch {} test error {:.2f}%'.format(index + 1, 100 *
                compute_error(network, weights, dataset.testing)))
