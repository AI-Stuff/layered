import pytest
from layered.utility import repeat, batched, average


class MockGenerator:

    def __init__(self, data):
        self.data = data
        self.evaluated = 0

    def __iter__(self):
        for element in self.data:
            self.evaluated += 1
            yield element


class MockCustomOperators:

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return MockCustomOperators(self.value + other.value)

    __radd__ = __add__

    def __truediv__(self, other):
        return MockCustomOperators(self.value / other)


class TestRepeat:

    def test_result(self):
        iterable = range(14)
        repeated = repeat(iterable, 3)
        assert list(repeated) == list(iterable) * 3

    def test_generator(self):
        iterable = MockGenerator([1, 2, 3])
        repeated = repeat(iterable, 3)
        assert iterable.evaluated == 0
        list(repeated)
        assert iterable.evaluated == 3 * 3


class TestBatched:

    def test_result(self):
        iterable = range(14)
        batches = batched(iterable, 3)
        batches = list(batches)
        assert len(batches) == 5
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 2

    def test_generator(self):
        iterable = MockGenerator([1, 2, 3])
        batches = batched(iterable, 3)
        assert iterable.evaluated == 0
        list(batches)
        assert iterable.evaluated == 3


class TestAverage:

    def test_result(self):
        assert average([1, 2, 3, 4], lambda x: x) == 2.5
        assert average([1, 2, 3, 4], lambda x: x ** 2) == 7.5

    def test_custom_operators(self):
        iterable = [MockCustomOperators(i) for i in range(1, 5)]
        assert average(iterable, lambda x: x).value == 2.5
