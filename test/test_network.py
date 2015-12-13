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

    def test_indexing(self, matrices):
        for index, matrix in enumerate(matrices):
            for (x, y), _ in np.ndenumerate(matrix):
                assert matrices[index][x, y] == matrices[index, x, y]

    def test_slicing(self, matrices):
        for index, matrix in enumerate(matrices):
            assert (matrices[index][:, :] == matrices[index, :, :]).all()
            assert (matrices[index][:, :] == matrix[:, :]).all()

    def test_negative_indices(self, matrices):
        for i in range(len(matrices)):
            positive = matrices[len(matrices) - i - 1]
            negative = matrices[i - 1]
            assert negative.shape == positive.shape
            assert (negative == positive).all()

    def test_assignment(self, matrices):
        matrices[0, 4, 5] = 42
        assert matrices[0, 4, 5] == 42

    def test_matrix_assignment(self, matrices):
        np.random.seed(0)
        matrix = np.random.rand(*matrices.shapes[0])
        matrices[0] = matrix
        assert (matrices[0] == matrix).all()

    def test_sliced_matrix_assignment(self, matrices):
        np.random.seed(0)
        matrix = np.random.rand(*matrices.shapes[0])
        matrices[0][:, :] = matrix
        assert (matrices[0] == matrix).all()
        matrices[0, :, :] = matrix
        assert (matrices[0] == matrix).all()

    def test_invalid_matrix_assignment(self, matrices):
        np.random.seed(0)
        shape = matrices.shapes[0]
        matrix = np.random.rand(shape[0] + 1, shape[1])
        with pytest.raises(ValueError):
            matrices[0] = matrix
