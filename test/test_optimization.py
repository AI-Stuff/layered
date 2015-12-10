# pylint: disable=wildcard-import, unused-wildcard-import, no-self-use
import numpy as np
from layered.optimization import GradientDecent, Momentum, WeightDecay
from test.fixtures import *


class TestGradientDecent:

    def test_calculation(self, weights_and_gradient):
        weights, gradient = weights_and_gradient
        decent = GradientDecent()
        updated = decent(weights, gradient, 0.1)
        reference = weights - 0.1 * gradient
        assert np.allclose(updated, reference)


class TestMomentum:

    def test_zero_rate(self, weights_and_gradient):
        _, gradient = weights_and_gradient
        original = gradient
        momentum = Momentum()
        for _ in range(5):
            gradient = momentum(gradient, rate=0)
        assert np.allclose(gradient, original)


class TestWeightDecay:

    def test_calculation(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        decay = WeightDecay()
        updated = decay(weights, 0.1)
        reference = 0.9 * weights
        assert np.allclose(updated, reference)
