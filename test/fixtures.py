import pytest
import numpy as np
from layered.network import Matrices
from layered.utility import pairwise


def random_matrices(shapes):
    matrix = Matrices(shapes)
    print(matrix.shape)
    matrix.flat = np.random.normal(0, 0.1, len(matrix.flat))
    print(matrix.shape)
    return matrix


@pytest.fixture(params=[(5, 5, 6, 3)])
def weights_and_gradient(request):
    shapes = list(pairwise(request.param))
    weights = random_matrices(shapes)
    gradient = random_matrices(shapes)
    return weights, gradient
