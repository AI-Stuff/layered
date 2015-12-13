# pylint: disable=wildcard-import, unused-wildcard-import, no-self-use
import pytest
import numpy as np
from layered import optimization
from test.fixtures import *


@pytest.fixture(params=[(1, 1), (1, 2), (2, 1), (2, 2), (4, 5)])
def weights_and_gradient_and_groups(request):
    size, layers = request.param
    shapes = [(size, size)] * layers
    weights = random_matrices(shapes)
    gradient = random_matrices(shapes)
    slices = [np.s_[i, :, :] for i, x in enumerate(weights)]
    groups = (slices,)
    return weights, gradient, groups


class TestGradientDecent:

    def test_calculation(self, weights_and_gradient):
        weights, gradient = weights_and_gradient
        decent = optimization.GradientDecent()
        updated = decent(weights, gradient, 0.1)
        reference = weights - 0.1 * gradient
        assert np.allclose(updated, reference)

    def test_shapes_match(self, weights_and_gradient):
        weights, gradient = weights_and_gradient
        decent = optimization.GradientDecent()
        updated = decent(weights, gradient, 0.1)
        assert weights.shapes == updated.shapes

    def test_copy_data(self, weights_and_gradient):
        weights, gradient = weights_and_gradient
        decent = optimization.GradientDecent()
        before = weights.copy()
        updated = decent(weights, gradient, 0.1)
        assert (before.flat == weights.flat).all()
        assert updated.flat[0] != 42
        weights.flat[0] = 42
        assert updated.flat[0] != 42


class TestMomentum:

    def test_zero_rate(self, weights_and_gradient):
        _, gradient = weights_and_gradient
        original = gradient
        momentum = optimization.Momentum()
        for _ in range(5):
            gradient = momentum(gradient, rate=0)
        assert np.allclose(gradient, original)

    def test_shapes_match(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        momentum = optimization.Momentum()
        updated = momentum(weights, 0.1)
        assert weights.shapes == updated.shapes

    def test_copy_data(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        momentum = optimization.Momentum()
        before = weights.copy()
        updated = momentum(weights, 0.1)
        assert (before.flat == weights.flat).all()
        assert updated.flat[0] != 42
        weights.flat[0] = 42
        assert updated.flat[0] != 42


class TestWeightDecay:

    def test_calculation(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        decay = optimization.WeightDecay()
        updated = decay(weights, 0.1)
        reference = 0.9 * weights
        assert np.allclose(updated, reference)

    def test_shapes_match(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        decay = optimization.WeightDecay()
        updated = decay(weights, 0.1)
        assert weights.shapes == updated.shapes

    def test_copy_data(self, weights_and_gradient):
        weights, _ = weights_and_gradient
        decay = optimization.WeightDecay()
        before = weights.copy()
        updated = decay(weights, 0.1)
        assert (before.flat == weights.flat).all()
        assert updated.flat[0] != 42
        weights.flat[0] = 42
        assert updated.flat[0] != 42


class TestWeightTying:

    def test_calculation(self, weights_and_gradient_and_groups):
        weights, _, groups = weights_and_gradient_and_groups
        tying = optimization.WeightTying(*groups)
        updated = tying(weights)
        self._is_tied(updated, groups)

    def test_shapes_match(self, weights_and_gradient_and_groups):
        weights, _, groups = weights_and_gradient_and_groups
        tying = optimization.WeightTying(*groups)
        updated = tying(weights)
        assert weights.shapes == updated.shapes

    def test_dont_affect_others(self, weights_and_gradient_and_groups):
        weights, _, _ = weights_and_gradient_and_groups
        if len(weights.shapes) < 2:
            pytest.skip()
        group = (np.s_[0, :, :], np.s_[1, :, :])
        tying = optimization.WeightTying(group)
        updated = tying(weights)
        assert (updated[0] == updated[1]).all()
        for before, after in zip(weights[2:], updated[2:]):
            assert (before == after).all()

    def test_weights_stay_tied(self, weights_and_gradient_and_groups):
        weights, gradient, groups = weights_and_gradient_and_groups
        tying = optimization.WeightTying(*groups)
        decent = optimization.GradientDecent()
        weights = tying(weights)
        weights = decent(weights, gradient, 0.1)
        self._is_tied(weights, groups)

    def test_copy_data(self, weights_and_gradient_and_groups):
        weights, _, groups = weights_and_gradient_and_groups
        tying = optimization.WeightTying(*groups)
        before = weights.copy()
        updated = tying(weights)
        assert (before.flat == weights.flat).all()
        assert updated.flat[0] != 42
        weights.flat[0] = 42
        assert updated.flat[0] != 42

    def _is_tied(self, matrices, groups):
        for group in groups:
            slices = [matrices[x] for x in group]
            assert [np.allclose(x, slices[0]) for x in slices]
