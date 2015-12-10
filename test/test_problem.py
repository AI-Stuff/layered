# pylint: disable=no-self-use
import pytest
from layered.problem import Problem


class TestProblem:

    def test_unknown_property(self):
        with pytest.raises(Exception):
            Problem('foo: 42')

    def test_incompatible_type(self):
        with pytest.raises(Exception):
            Problem('learning_rate: foo')

    def test_read_value(self):
        problem = Problem('learning_rate: 0.4')
        assert problem.learning_rate == 0.4

    def test_default_value(self):
        problem = Problem(' ')
        print(problem)
        assert problem.learning_rate == 0.1
