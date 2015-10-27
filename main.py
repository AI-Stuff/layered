import numpy as np
from layered.example import Example
from layered.network import Network, Layer
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.optimization import MiniBatchGradientDecent
from layered.gradient import Backpropagation, CheckedGradient
from layered.plot import Plot
from layered.dataset import Regression, Classification, Mnist


if __name__ == '__main__':
    # Generate and split dataset.
    # examples = regression(50000)
    print('Loading dataset')
    dataset = Mnist()

    # Create a network. The input and output layer sizes are derived from the
    # input and target data.
    network = Network([
        Layer(len(dataset.training[0].data), Linear),
        Layer(500, Relu),
        Layer(len(dataset.training[0].target), Sigmoid)
    ])

    # Training.
    cost = Squared
    gradient = Backpropagation(network, cost)
    # gradient = CheckedGradient(network, cost, Backpropagation)
    optimization = MiniBatchGradientDecent(network, cost, gradient,
        learning_rate=1e-1, batch_size=10)
    plot = Plot(optimization)
    print('Start training')
    try:
        for error in optimization.apply(dataset.training):
            plot.apply(error)
    except KeyboardInterrupt:
        print('\nAborted')

    # Evaluation.
    print('Evaluation')
    predictions = [network.feed(x.data) for x in dataset.testing]
    pairs = [(x, y.target) for x, y in zip(predictions, dataset.testing)]
    for pair in pairs[:20]:
        print(pair)
    error = sum(cost().apply(x, y).sum() / len(x) for x, y in pairs)
    error /= len(dataset.testing)
    accuracy = sum(np.argmax(x) == np.argmax(y) for x, y in pairs)
    accuracy /= len(dataset.testing)
    print('Test set cost {:.6f} and classification error {:.2f}%'.format(
        error, 100 * (1 - accuracy)))

