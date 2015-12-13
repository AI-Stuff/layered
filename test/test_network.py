# pylint: disable=no-self-use
import pytest
import numpy as np
from layered.network import Matrices


@pytest.fixture
def matrices():
    return Matrices([(5, 8), (4, 2)])


class TestMatrices:

    def test_initialization(self, matrices):
        assert np.array_equal(matrices[0], np.zeros((5, 8)))
        assert np.array_equal(matrices[1], np.zeros((4, 2)))

    def test_slice_indices(self, matrices):
        for index, matrix in enumerate(matrices):
            for (x, y), _ in np.ndenumerate(matrix):
                slice_ = np.s_[index, x, y]
                assert matrices[index][x, y] == matrices[slice_]

    def test_negative_indices(self, matrices):
        assert matrices[-1].shape == matrices[len(matrices) - 1].shape

    def test_number_assignment(self, matrices):
        matrices[0, 4, 5] = 42
        assert matrices[0, 4, 5] == 42

    def test_matrix_assignment(self, matrices):
        matrix = np.random.rand(5, 8)
        matrices[0] = matrix
        assert (matrices[0] == matrix).all()

    def test_invalid_matrix_assignment(self, matrices):
        with pytest.raises(ValueError):
            matrices[0] = np.random.rand(5, 9)
