import numpy as np
import matplotlib.pyplot as plt
from drawille import Canvas


class Network:

    def __init__(self, layer_sizes):
        # Add extra weights for the biases.
        self._init_neurons(layer_sizes)
        self._init_weights(layer_sizes)
        # The incoming and outgoing activations for each neuron.
        assert len(self.incoming) == len(self.outgoing)
        # Weights between the layers. Weight matrices are pointing outwards
        # from the layer with the same index.
        assert len(self.incoming) == len(self.weights) + 1

    def _init_neurons(self, layer_sizes):
        assert all(layer_sizes)
        # Incoming and outgoing activation for each neuron in each layer. We
        # don't need to store incoming values for the input layer since no
        # activation function is applied there.
        self.incoming = [None] + [np.zeros(size) for size in layer_sizes[1:]]
        self.outgoing = [np.zeros(size) for size in layer_sizes]

    def _init_weights(self, layer_sizes, scale=1e-4):
        self.weights = []
        for shape in zip(layer_sizes, layer_sizes[1:]):
            # An additional input to the next layer is the bias value of one.
            shape = shape[0] + 1, shape[1]
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
        # flat = np.hstack(np.array(list(x.flatten() for x in gradient)))
        # print(flat.min(), flat.max())
        # for i in range(len(gradient)):
        #    gradient[i] = np.clip(gradient[i], -100, 100)
        # Update weights in opposide gradient-direction multiplied by the
        # learning rate.
        self.weights -= learning_rate * gradient
        # Return the loss on the provided training examples.
        return loss

    def feed(self, inputs):
        assert len(inputs) == len(self.outgoing[0])
        self.outgoing[0] = inputs
        # Propagate trough the hidden layers.
        for i in range(1, len(self.outgoing)):
            # Input for the current layer is the previous layer and a bias
            # value of one.
            previous = np.insert(self.outgoing[i-1], 1, 0)
            self.incoming[i] = previous.dot(self.weights[i-1])
            self.outgoing[i] = self._activation(self.incoming[i])
        # Return the activations of the output layer.
        return self.outgoing[-1]

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
        assert len(target) == len(self.outgoing[-1])
        # We start with the gradient at the output neurons. It's affected by
        # the loss and activation.
        delta_loss = self._delta_loss(self.outgoing[-1], target)
        delta_activation = self._delta_activation(self.incoming[-1],
                self.outgoing[-1])
        gradient = [delta_loss * delta_activation]
        # Propagate backwards trough the hidden layers.
        for i in reversed(range(1, len(self.incoming) - 1)):
            # The gradient at a layer is computed as the derivative of both the
            # activation and the weighted sum of the derivatives of the deeper
            # layer. The first element of the outgoing sums of derivatives
            # corresponds to the bias value of one that cannot be changed
            # anyway.
            outgoing_sum = gradient[-1].dot(self.weights[i].transpose())[1:]
            delta_activation = self._delta_activation(self.incoming[i],
                    self.outgoing[i])
            gradient.append(outgoing_sum * delta_activation)
        gradient = list(reversed(gradient))
        # The gradient of the neurons has the same size as the neurons except
        # that we don't need have gradients for the input layer.
        assert len(gradient) == len(self.incoming) - 1
        assert all(len(x) == len(y) for x, y in
                zip(gradient, self.incoming[1:]))
        return gradient

    def _gradient_weights(self, gradient_neurons):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for outgoing, delta_incoming in zip(self.outgoing[:-1],
                gradient_neurons):
            bias_and_outgoing = np.insert(outgoing, 1, 0)
            gradient.append(np.outer(bias_and_outgoing, delta_incoming))
        # The gradient of the weights has the same size as the weights.
        assert len(gradient) == len(self.weights)
        assert all(len(x) == len(y) for x, y in zip(gradient, self.weights))
        return gradient

    def _activation(self, incoming):
        assert hasattr(incoming, '__len__')
        # return incoming
        # return np.maximum(incoming, 0)
        return 1 / (1 + np.exp(-incoming))

    def _delta_activation(self, incoming, outgoing):
        assert len(incoming) == len(outgoing)
        # return np.ones(incoming.shape).astype(float)
        # return np.greater(incoming, 0).astype(float)
        return outgoing * (1 - outgoing)

    def _loss(self, outputs, targets):
        errors = np.square(outputs - targets)
        loss = np.sum(errors) / 2
        return loss

    def _delta_loss(self, outputs, targets):
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

    def __init__(self, network, learning_rate=1e-3, batch_size=10,
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
        plot_batches = max(1, int(self.plot_freq // self.batch_size))
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
        try:
            self.chart += loss
            print(self.chart)
        except:
            print('Failed to draw chart')

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
    network = Network([len(inputs[0])] + [9] * 2 + [len(targets[0])])
    # Train the network on the training examples.
    training = Training(network, learning_rate=1e-2, batch_size=2)
    losses = []
    for _ in range(2):
        losses += training(train_inputs, train_targets)
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

    products = np.prod(inputs, axis=1)
    products = products / np.max(products)
    sums = np.sum(inputs, axis=1)
    sums = sums / np.max(sums)
    targets = np.column_stack([sums, products])

    create_train_test(inputs, targets)

