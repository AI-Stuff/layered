import pytest
import numpy as np
from layered.network import Matrices


class TestMatrices:

    def test_initialization(self):
        matrices = Matrices([(5, 8), (4, 2)])
        assert np.array_equal(matrices[0], np.zeros((5, 8)))
        assert np.array_equal(matrices[1], np.zeros((4, 2)))

    def test_get_item(self):
        matrices = Matrices([(5, 8), (4, 2)])
        assert matrices[0][4, 2] == 0

    def test_negative_indices(self):
        matrices = Matrices([(5, 8), (4, 2)])
        print(matrices[1])
        print(matrices[-1])
        assert matrices[-1].shape == (4, 2)

    def test_number_assignment(self):
        matrices = Matrices([(5, 8), (4, 2)])
        matrices[0][4, 5] = 42
        assert matrices[0][4, 5] == 42

    def test_matrix_assignment(self):
        matrices = Matrices([(5, 8), (4, 2)])
        matrix = np.random.rand(5, 8)
        matrices[0] = matrix
        assert (matrices[0] == matrix).all()

    def test_invalid_matrix_assignment(self):
        matrices = Matrices([(5, 8), (4, 2)])
        with pytest.raises(ValueError):
            matrices[0] = np.random.rand(5, 9)
