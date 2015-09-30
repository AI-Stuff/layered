import numpy as np
import matplotlib.pyplot as plt
from drawille import Canvas


class Network:

    def __init__(self, layer_sizes):
        # Add extra weights for the biases.
        self._init_neurons(layer_sizes)
        self._init_weights(layer_sizes)
        # The weight matrices are between the layers.
        assert len(self.layers) - 1 == len(self.weights)

    def _init_neurons(self, layer_sizes):
        assert all(layer_sizes)
        self.layers = [np.zeros(size) for size in layer_sizes]

    def _init_weights(self, layer_sizes, scale=1e-3):
        self.weights = []
        for from_, to in zip(layer_sizes, layer_sizes[1:]):
            # An additional input to the next layer is the bias value of one.
            shape = from_ + 1, to
            weights = np.random.normal(0, scale, shape)
            self.weights.append(weights)

    def train(self, inputs, targets, learning_rate):
        """
        Perform a forward and backward pass for each example and average the
        gradients. Update the weights in the opposite direction scaled by the
        learning rate. Return the average loss over the examples.
        """
        assert len(inputs) == len(targets)
        # In batch gradient decent, we average the gradients over the training
        # examples.
        gradient = [np.zeros(x.shape) for x in self.weights]
        loss = 0
        for input_, target in zip(inputs, targets):
            output = self.feed(input_)
            gradient += self._back_propagation(target)
            loss += self._loss(output, target)
        # Normalize by the number of examples.
        loss /= len(inputs)
        gradient /= len(inputs)
        # Update weights in opposide gradient-direction multiplied by the
        # learning rate.
        self.weights -= learning_rate * gradient
        # Return the loss on the provided training examples.
        return loss

    def feed(self, inputs):
        assert len(inputs) == len(self.layers[0])
        # Set inputs.
        self.layers[0] = inputs
        # Propagate layer by layer. Input for the next layer is the current
        # layer and a bias value of one.
        for i in range(len(self.layers) - 1):
            bias_and_incoming = np.insert(self.layers[i], 1, 0)
            activation = bias_and_incoming.dot(self.weights[i])
            activation = self._activation(activation)
            assert len(activation) == len(self.layers[i + 1])
            self.layers[i + 1] = activation
        # Return the activations of the output layer.
        return self.layers[-1]

    def evaluate(self, inputs, targets):
        loss = 0
        for input_, target in zip(inputs, targets):
            output = self.feed(input_)
            loss += self._loss(output, target)
        return loss / len(inputs)

    def _back_propagation(self, target):
        gradient_neurons = self._gradient_neurons(target)
        gradient_weights = self._gradient_weights(gradient_neurons)
        return np.array(gradient_weights)

    def _gradient_neurons(self, target):
        assert len(target) == len(self.layers[-1])
        # The gradient at the output neurons is given by the derivative of the
        # loss function.
        gradient = [self._loss_derivative(self.layers[-1], target)]
        # Propagate layer by layer backwards. The gradient at each layer is
        # computed as the weighted sum of the activations of the next deeper
        # layer with the derivative of the activation function applied to all
        # value.
        layers_weights = list(zip(self.layers[:-1], self.weights))
        for layer, weights in reversed(layers_weights):
            # The first element is the derivative at the bias value of one. We
            # don't need that.
            outgoings = gradient[-1].dot(weights.transpose())[1:]
            gradient.append(self._activation_derivative(layer, outgoings))
        gradient = list(reversed(gradient))
        # The gradient of the neurons has the same size as the neurons.
        assert len(gradient) == len(self.layers)
        assert all(len(x) == len(y) for x, y in zip(gradient, self.layers))
        return gradient

    def _gradient_weights(self, gradient_neurons):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for activation, derivatives in zip(self.layers[:-1],
                gradient_neurons[1:]):
            bias_activation = np.insert(activation, 1, 0)
            gradient.append(np.outer(bias_activation, derivatives))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(self.weights)
        assert all(len(x) == len(y) for x, y in zip(gradient, self.weights))
        return gradient

    def _activation(self, incomings):
        # return np.maximum(incomings, 0)
        activation = 1 / (1 + np.exp(-incomings))
        assert len(activation) == len(incomings)
        return activation

    def _activation_derivative(self, activations, outgoings):
        # return np.greater(activations, 0).astype(float)
        sigmoid = self._activation(outgoings)
        return sigmoid * (1 - sigmoid)

    def _loss(self, outputs, targets):
        errors = np.square(outputs - targets)
        loss = np.sum(errors) / 2
        return loss

    def _loss_derivative(self, outputs, targets):
        return outputs - targets

    def _print_neurons(self):
        for layer in self.layers:
            print(np.round(layer, 2))


class TerminalChart:

    def __init__(self, width=100, height=100):
        self.height = height
        self.width = width
        self.data = [0] * self.width
        self.canvas = Canvas()
        self.offset = 0
        self.max = 0

    def __iadd__(self, values):
        if not hasattr(values, '__len__'):
            values = [values]
        for value in values:
            self.data[self.offset % self.width] = value
            self.offset += 1
            self.max = max(self.max, value)
        return self

    def __str__(self):
        data = np.array(self.data)
        scaled = data * self.height / (self.max or 1)
        self.clear()
        for i in range(self.offset, self.offset + self.width):
            i %= self.width
            self.canvas.set(i + 1, self.height + 1 - int(scaled[i]))
        frame = self.canvas.frame(0, 0, self.width + 2, self.height + 1)
        return frame

    def clear(self):
        self.canvas.clear()
        for x in range(self.width + 1):
            self.canvas.set(x, 0)
            self.canvas.set(x, self.height + 1)
        for y in range(self.height + 1):
            self.canvas.set(0, y)
            self.canvas.set(self.width, y)


class Training:

    def __init__(self, network, learning_rate=1e-3, batch_size=100,
            plot_freq=5e2):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.plot_freq = plot_freq
        self.chart = TerminalChart()
        print(self.chart)

    def __call__(self, inputs, targets):
        """
        Split examples into batches and train on them. Return a list of the
        losses on each batch.
        """
        assert len(inputs) == len(targets)
        plot_batches = int(self.plot_freq // self.batch_size)
        losses = []
        for xs, ys in self._batched(inputs, targets, self.batch_size):
            loss = self.network.train(xs, ys, self.learning_rate)
            losses.append(loss)
            # Plot the average loss over the last few batches as soon as they
            # are available.
            if len(losses) % plot_batches == 0 and len(losses) >= plot_batches:
                average_loss = sum(losses[-plot_batches:]) / plot_batches
                self._plot_loss(average_loss)
        return losses

    def _plot_loss(self, loss):
        self.chart += loss
        print(self.chart)

    def _batched(self, inputs, targets, size):
        assert len(inputs) == len(targets)
        for i in range(0, len(inputs), size):
            yield inputs[i:i+size], targets[i:i+size]


def create_train_test(inputs, targets):
    assert len(inputs) == len(targets)
    assert len(inputs[0]) and len(targets[0])
    # Split the dataset into many training examples and some test examples.
    split = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:split], inputs[split:]
    train_targets, test_targets = targets[:split], targets[split:]
    # Create a network. The input and output layer sizes are deriven from the
    # input and target data.
    network = Network([len(inputs[0])] + [15] * 2 + [len(targets[0])])
    # Train the network on the training examples.
    training = Training(network, 1e-4, 100)
    losses = training(train_inputs, train_targets)
    plt.plot(losses)
    plt.xlabel('training batches')
    plt.ylabel('squared errors of current batch')
    plt.ylim(ymin=0)
    plt.xlim(xmax=len(losses))
    plt.show()
    # Evaluate the trained network on the test examples.
    loss = network.evaluate(test_inputs, test_targets)
    print('Test set loss:', loss)

if __name__ == '__main__':
    inputs = np.random.rand(100000, 10)
    targets = []
    # targets.append(np.prod(inputs, axis=1))
    targets.append(np.sum(inputs, axis=1))
    targets = np.column_stack(targets)
    create_train_test(inputs, targets)

