import numpy as np
import matplotlib.pyplot as plt


class Example:
    """
    Immutable class representing one example in a dataset.
    """
    __slots__ = ('data', 'target')

    def __init__(self, data, target):
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'target', target)

    def __setattr__(self, *args):
        raise TypeError

    def __delattr__(self, *args):
        raise TypeError


class Function:

    @staticmethod
    def apply(incoming):
        raise NotImplemented

    @staticmethod
    def delta(incoming, outgoing):
        raise NotImplemented


class Sigmoid(Function):

    @staticmethod
    def apply(incoming):
        return 1 / (1 + np.exp(-incoming))

    @staticmethod
    def delta(incoming, outgoing):
        return outgoing * (1 - outgoing)


class Linear(Function):

    @staticmethod
    def apply(incoming):
        return incoming

    @staticmethod
    def delta(incoming, outgoing):
        return np.ones(incoming.shape).astype(float)


class Relu(Function):

    @staticmethod
    def apply(incoming):
        return np.maximum(incoming, 0)

    @staticmethod
    def delta(incoming, outgoing):
        return np.greater(incoming, 0).astype(float)


class Loss:

    @staticmethod
    def apply(prediction, target):
        raise NotImplemented

    @staticmethod
    def delta(prediction, target):
        raise NotImplemented


class SquaredLoss(Loss):

    @staticmethod
    def apply(prediction, target):
        return np.sum(np.square(prediction - target)) / 2

    @staticmethod
    def delta(prediction, target):
        return prediction - target


class Layer:

    def __init__(self, size, function):
        assert isinstance(size, int) and size
        self.size = size
        self.function = function
        self.incoming = np.zeros(size)
        self.outgoing = np.zeros(size)
        assert len(self.incoming) == len(self.outgoing) == self.size

    def apply(self, incoming):
        """
        Store the incoming activation, apply the activation function and store
        the result as outgoing activation.
        """
        assert len(incoming) == self.size
        self.incoming = incoming
        outgoing = self.function.apply(self.incoming)
        assert len(outgoing) == self.size
        self.outgoing = outgoing

    def delta(self):
        """
        The derivative of the activation function at the current state.
        """
        return self.function.delta(self.incoming, self.outgoing)


class Weight:

    def __init__(self, from_, to, init_scale=1e-4):
        assert isinstance(from_, int) and isinstance(to, int)
        # Add extra weights for the biases.
        self.shape = (from_ + 1, to)
        self.weight = np.random.normal(0, init_scale, self.shape)

    def __isub__(self, other):
        assert other.shape == self.weight.shape
        self.weight -= other

    def forward(self, previous):
        # Add bias input of one.
        previous = np.insert(previous, 1, 0)
        current = previous.dot(self.weight)
        return current

    def backward(self, next_):
        current = next_.dot(self.weight.transpose())
        # Remove bias input of one.
        current = current[1:]
        return current


class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.sizes = tuple(layer.size for layer in self.layers)
        # Weights are stored as matrices of the shape length of the left layer
        # times length of the right layer.
        shapes = zip(self.sizes[:-1], self.sizes[1:])
        self.weights = list(Weight(*shape) for shape in shapes)
        # Weight matrices are in between the layers.
        assert len(self.weights) == len(self.layers) - 1

    def train(self, examples, learning_rate):
        """
        Perform a forward and backward pass for each example and average the
        gradients. Update the weights in the opposite direction scaled by the
        learning rate. Return the average loss over the examples.
        """
        # In batch gradient decent, we average the gradients over the training
        # examples.
        loss = 0
        gradient = list(np.zeros(weight.shape) for weight in self.weights)
        for example in examples:
            prediction = self.feed(example.data)
            for i, delta in enumerate(self._backpropagation(example.target)):
                gradient[i] += delta
            loss += self.loss.apply(prediction, example.target)
        # Normalize by the number of examples.
        loss /= len(examples)
        gradient = list(x / len(examples) for x in gradient)
        # The minimum and maximum gradient values are useful to validate
        # the gradient calculation and understand what the network is doing
        # internally.
        # flat = np.hstack(np.array(list(x.flatten() for x in gradient)))
        # print('gradient min:', flat.min(), 'max:', flat.max())
        # Clip extreme gradient values.
        # for i in range(len(gradient)):
        #    gradient[i] = np.clip(gradient[i], -100, 100)
        # Update weights in opposide gradient-direction multiplied by the
        # learning rate.
        for index, derivative in enumerate(gradient):
            self.weights[index].weight -= learning_rate * derivative
        # Return the loss on the provided training examples.
        return loss

    def feed(self, data):
        assert len(data) == self.layers[0].size
        self.layers[0].apply(data)
        # Propagate trough the remaining layers.
        connections = zip(self.layers[:-1], self.weights, self.layers[1:])
        for previous, weight, current in connections:
            incoming = weight.forward(previous.outgoing)
            current.apply(incoming)
        # Return the activations of the output layer.
        return self.layers[-1].outgoing

    def evaluate(self, examples):
        """
        Return the average loss over the examples.
        """
        loss = 0
        for example in examples:
            prediction = self.feed(example.data)
            loss += self._loss(prediction, example.target)
        return loss / len(examples)

    def _backpropagation(self, target):
        delta_layers = self._delta_layers(target)
        delta_weights = self._delta_weights(delta_layers)
        return delta_weights

    def _delta_layers(self, target):
        assert len(target) == self.layers[-1].size
        # We start with the gradient at the output layer. It's computed as the
        # product of error derivative and local derivative at the last layer.
        prediction = self.layers[-1].outgoing
        loss = self.loss.delta(prediction, target)
        local = self.layers[-1].delta()
        gradient = [loss * local]
        # Propagate backwards trough the hidden layers but not the input layer.
        hidden = list(zip(self.weights[1:], self.layers[1:-1]))
        for weight, layer in reversed(hidden):
            # The gradient at a layer is computed as the derivative of both the
            # local activation and the weighted sum of the derivatives in the
            # deeper layer.
            deeper = weight.backward(gradient[-1])
            local = layer.delta()
            gradient.append(deeper * local)
        gradient = list(reversed(gradient))
        # We computed the gradient at the hidden layers and the output layer.
        assert len(gradient) == len(self.layers) - 1
        assert all(len(x) == y.size for x, y in zip(gradient, self.layers[1:]))
        return gradient

    def _delta_weights(self, delta_layers):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for previous, delta in zip(self.layers[:-1], delta_layers):
            # We want to tweak the bias weights so we need them in the
            # gradient.
            bias_and_activation = np.insert(previous.outgoing, 1, 0)
            gradient.append(np.outer(bias_and_activation, delta))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(self.weights)
        assert all(len(x) == len(y.weight) for x, y
            in zip(gradient, self.weights))
        return gradient


class Trainer:

    def __init__(self, network, batch_size=10, plot_freq=5e2,
            plot_smoothing=100):
        self.network = network
        self.batch_size = batch_size
        self.plot_freq = plot_freq
        self.plot_smoothing = plot_smoothing
        self.losses = []
        self.smoothed = []
        self._init_plot()

    def _init_plot(self):
        self.plot, = plt.plot([], [])
        plt.xlabel('training batches')
        plt.ylabel('squared errors of current batch')
        plt.ion()
        plt.show()

    def __call__(self, examples, learning_rate=1e-3):
        """
        Split examples into batches and train on them. Return a list of the
        losses on each batch.
        """
        plot_batches = max(1, int(self.plot_freq // self.batch_size))
        for example in self._batched(examples, self.batch_size):
            loss = self.network.train(example, learning_rate)
            self.losses.append(loss)
            self.smoothed.append(np.mean(self.losses[-self.plot_smoothing:]))
            if len(self.losses) % plot_batches == 0:
                self._plot_loss()

    def _plot_loss(self):
        range_ = range(len(self.smoothed))
        plt.xlim(xmin=0, xmax=range_[-1])
        plt.ylim(ymin=0, ymax=max(self.smoothed))
        self.plot.set_data(range_, self.smoothed)
        plt.draw()
        plt.pause(0.001)

    def _batched(self, examples, size):
        for i in range(0, len(examples), size):
            yield examples[i:i+size]


def generate_mock_examples(amount):
    data = np.random.rand(amount, 10)
    products = np.prod(data, axis=1)
    products = products / np.max(products)
    sums = np.sum(data, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])
    return [Example(x, y) for x, y in zip(data, targets)]


if __name__ == '__main__':
    # Generate and split dataset.
    examples = generate_mock_examples(100000)
    split = int(0.8 * len(examples))
    training, testing = examples[:split], examples[split:]
    # Create a network. The input and output layer sizes are deriven from the
    # input and target data.
    network = Network([
        Layer(len(training[0].data), Linear),
        Layer(10, Sigmoid),
        Layer(10, Sigmoid),
        Layer(len(training[0].target), Sigmoid)  # TODO: Softmax
    ], SquaredLoss)
    # Training.
    trainer = Trainer(network, batch_size=2, plot_smoothing=5)
    for _ in range(2):
        trainer(training)
    # Evaluation.
    loss = network.evaluate(testing)
    print('Test set loss:', loss)

