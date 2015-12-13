# pylint: disable=no-self-use, wildcard-import, unused-wildcard-import
from layered.activation import Identity, Relu
from layered.cost import CrossEntropy
from layered.gradient import (
    NumericalGradient, Backprop, BatchBackprop, ParallelBackprop)
from test.fixtures import *


class TestBackprop:

    def test_against_numerical(self, network_and_weights, cost, example):
        network, weights = network_and_weights
        if isinstance(cost, CrossEntropy) and isinstance(
                network.layers[1].activation, (Identity, Relu)):
            pytest.xfail(
                'Cross entropy doesn\'t work with linear activations for some '
                'reason.')
        backprop = Backprop(network, cost)
        numerical = NumericalGradient(network, cost)
        gradient = backprop(weights, example)
        reference = numerical(weights, example)
        assert np.allclose(gradient, reference)


class TestBatchBackprop:

    def test_calculation(self, network_and_weights, cost, examples):
        network, weights = network_and_weights
        batched = BatchBackprop(network, cost)
        backprop = Backprop(network, cost)
        gradient = batched(weights, examples)
        reference = sum(backprop(weights, x) for x in examples) / len(examples)
        assert np.allclose(gradient, reference)


class TestParallelBachprop:

    def test_against_batch_backprop(self, network_and_weights, cost, examples):
        network, weights = network_and_weights
        parallel = ParallelBackprop(network, cost)
        batched = BatchBackprop(network, cost)
        gradient = parallel(weights, examples)
        reference = batched(weights, examples)
        assert np.allclose(gradient, reference)
