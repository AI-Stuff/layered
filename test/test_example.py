# pylint: disable=no-self-use
import numpy as np
from layered.example import Example


class TestExample:

    def test_representation(self):
        data = np.array([1, 2, 3])
        target = np.array([1, 2, 3])
        example = Example(data, target)
        repr(example)
