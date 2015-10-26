import numpy as np
from layered.example import Example
from layered.network import Network, Layer
from layered.activation import Linear, Sigmoid, Relu, Softmax
from layered.cost import Squared, CrossEntropy
from layered.optimization import MiniBatchGradientDecent
from layered.gradient import Backpropagation, CheckedGradient
from layered.plot import Plot
from layered.dataset import regression, classification, mnist


if __name__ == '__main__':
    # Generate and split dataset.
    # examples = regression(50000)
    # split = int(0.8 * len(examples))
    # training, testing = examples[:split], examples[split:]
    print('Loading dataset')
    training, testing = mnist()

    # Create a network. The input and output layer sizes are derived from the
    # input and target data.
    network = Network([
        Layer(len(training[0].data), Linear),
        Layer(1000, Relu),
        Layer(1000, Relu),
        Layer(len(training[0].target), Sigmoid)
    ])

    # Training.
    cost = Squared
    gradient = Backpropagation(network, cost)
    # gradient = CheckedGradient(network, cost, Backpropagation)
    optimization = MiniBatchGradientDecent(network, cost, gradient,
        learning_rate=1e-2, batch_size=10)
    plot = Plot(optimization)
    print('Start training')
    for error in optimization.apply(training):
        plot.apply(error)

    # Evaluation.
    print('Evaluation')
    predictions = [network.feed(x.data) for x in testing]
    pairs = [(x, y.target) for x, y in zip(predictions, testing)]
    for pair in pairs[:20]:
        print(pair)
    error = sum(cost().apply(x, y).sum() / len(x) for x, y in pairs)
    error /= len(testing)
    accuracy = sum(np.argmax(x) == np.argmax(y) for x, y in pairs)
    accuracy /= len(testing)
    print('Test set cost {:.6f} and classification error {:.2f}%'.format(
        error, 100 * (1 - accuracy)))

