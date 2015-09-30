import numpy as np
from drawille import Canvas


class Chart:

    def __init__(self, width=100, height=100):
        self.height = height
        self.width = width
        self.data = [0] * self.width
        self.canvas = Canvas()
        self.offset = 0
        self.max = 0

    def add(self, value):
        self.data[self.offset % self.width] = value
        self.offset += 1
        self.max = max(self.max, value)

    def __str__(self):
        data = np.array(self.data)
        scaled = data * self.height / self.max
        self.clear()
        for i in range(self.offset, self.offset + self.width):
            i %= self.width
            self.canvas.set(i + 1, self.height - int(scaled[i]))
        frame = self.canvas.frame(0, 0, self.width + 2, self.height + 2)
        return frame

    def clear(self):
        self.canvas.clear()
        for x in range(self.width + 1):
            self.canvas.set(x, 0)
            self.canvas.set(x, self.height)
        for y in range(self.height + 1):
            self.canvas.set(0, y)
            self.canvas.set(self.width, y)


class Network:

    def __init__(self, layer_sizes):
        # Add extra weights for the biases.
        layer_sizes = [x + 1 for x in layer_sizes]
        self._init_neurons(layer_sizes)
        self._init_weights(layer_sizes)
        # The weight matrices are between the layers.
        assert len(self.layers) - 1 == len(self.weights)
        # Chart to visualize the current loss during training.
        self.chart = Chart()

    def _init_neurons(self, layer_sizes):
        assert all(size for size in layer_sizes)
        # Layers have one more neuron than specified that is always set to one
        # and used to set feed the bias.
        self.layers = [np.zeros(size) for size in layer_sizes]

    def _init_weights(self, layer_sizes, scale=1e-3):
        self.weights = []
        for shape in zip(layer_sizes, layer_sizes[1:]):
            weights = np.random.normal(0, scale, shape)
            self.weights.append(weights)

    def train_batched(self, inputs, targets, batch_size=100,
            learning_rate=1e-3, plot_size=10000):
        assert len(inputs) == len(targets)
        losses = []
        for i in range(0, len(inputs), batch_size):
            input_batch = inputs[i:i+batch_size]
            target_batch = targets[i:i+batch_size]
            loss = self.train(input_batch, target_batch, learning_rate)
            losses.append(loss)
            if i % (plot_size // batch_size) == 0 and i > 2 * batch_size:
               self._plot_loss(loss)
        return losses

    def train(self, inputs, targets, learning_rate):
        assert len(inputs) == len(targets)
        # In batch gradient decent, we average the gradients over the training
        # examples.
        combined_gradient = [np.zeros(x.shape) for x in self.weights]
        combined_loss = 0
        for input_, target in zip(inputs, targets):
            output = self.feed(input_)
            current_gradient = self._back_propagation(target)
            combined_gradient += current_gradient / len(inputs)
            combined_loss += self._loss(output, target) / len(inputs)
        # Update weights in opposide gradient-direction multiplied by the
        # learning rate.
        self.weights -= learning_rate * combined_gradient
        # Return the loss on the provided training examples.
        return combined_loss

    def feed(self, inputs):
        assert len(inputs) == len(self.layers[0]) - 1
        # Set inputs.
        self.layers[0][0] = 1
        self.layers[0][1:] = inputs
        # Propagate layer by layer.
        for i in range(len(self.layers) - 1):
            activation = self._activation(self.layers[i].dot(self.weights[i]))
            # activation = self.layers[i].dot(self.weights[i])
            activation[0] = 1
            assert len(activation) == len(self.layers[i + 1])
            self.layers[i + 1] = activation
        # Return the activations of the output layer excluding the bias helper
        # neuron.
        return self.layers[-1][1:]

    def evaluate(self, input_, target):
        output = self.feed(input_)
        return self._loss(output, target)

    def _back_propagation(self, target):
        gradient_neurons = self._gradient_neurons(target)
        gradient_weights = self._gradient_weights(gradient_neurons)
        # print(np.sum(np.sum(np.abs(x)) for x in gradient_weights))
        return np.array(gradient_weights)

    def _gradient_neurons(self, target):
        assert len(target) == len(self.layers[-1]) - 1
        # The gradient at the output neurons is given by the derivative of the
        # loss function. The bias helper neuron is not adjusted so its
        # derivative is zero.
        target = np.array([0] + target.tolist())
        gradient = [self._loss_derivative(self.layers[-1], target)]
        # Propagate layer by layer backwards. The gradient at each layer is
        # computed as the weighted sum of the activations of the next deeper
        # layer with the derivative of the activation function applied to all
        # value.
        layers_and_outgoing_weights = zip(self.layers[1:-1], self.weights[1:])
        for layer, weights in reversed(list(layers_and_outgoing_weights)):
            outgoing = gradient[-1].dot(weights.transpose())
            gradient.append([0] + self._activation_derivative(layer, outgoing))
        gradient = list(reversed(gradient))
        # The bias helper neurons should not be modified and their gradient
        # should be zero.
        for layer in gradient:
            layer[0] = 0
        # The gradient of the neurons has the same size as the neurons except
        # that we don't need it for the input layer.
        assert len(gradient) == len(self.layers) - 1
        assert all(len(x) == len(y) for x, y in zip(gradient, self.layers[1:]))
        return gradient

    def _gradient_weights(self, gradient_neurons):
        gradient = []
        # The gradient with respect to the weights is computed as the gradient
        # at the target neuron multiplied by the activation of the source
        # neuron.
        for layer, next_gradient in zip(self.layers[:-1], gradient_neurons):
            gradient.append(np.outer(layer, next_gradient))
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

    def _plot_loss(self, loss):
        self.chart.add(loss)
        print(self.chart)

def create_train_test(inputs, targets):
    assert len(inputs) == len(targets)
    assert len(inputs[0]) and len(targets[0])
    # Split the dataset into many training examples and some test examples.
    split = int(0.5 * len(inputs))
    train_inputs, test_inputs = inputs[:split], inputs[split:]
    train_targets, test_targets = targets[:split], targets[split:]
    # Create a network. The input and output layer sizes are deriven from the
    # input and target data.
    network = Network([len(inputs[0])] + [15] * 2 + [len(targets[0])])
    # Train the network on the training examples.
    losses = []
    for i in range(5):
        losses += network.train_batched(train_inputs, train_targets,
                learning_rate=1e-3)
    # print(losses)
    # Evaluate the trained network.
    loss = 0
    for input_, target in zip(test_inputs, test_targets):
        loss += network.evaluate(input_, target) / len(test_inputs)
    return loss

if __name__ == '__main__':
    inputs = np.random.rand(100000, 10)
    targets = []
    # targets.append(np.prod(inputs, axis=1))
    targets.append(np.sum(inputs, axis=1))
    targets = np.column_stack(targets)
    create_train_test(inputs, targets)

