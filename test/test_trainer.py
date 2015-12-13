# pylint: disable=no-self-use
import pytest
from layered.trainer import Trainer
from layered.problem import Problem


@pytest.fixture
def problem():
    return Problem(
        """
        dataset: Test
        layers:
        - activation: Identity
          size: 3
        """)


class TestTrainer:

    def test_no_crash(self, problem):
        trainer = Trainer(problem)
        trainer()
