import pytest


class TestPlot:

    def test_interactive_backend(self):
        import matplotlib
        matplotlib.use('TkAgg')
