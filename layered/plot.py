import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook


warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


class Plot:

    def __init__(self, refresh=5000, smoothing=1):
        self.refresh = refresh
        self.costs = []
        self.offset = 0
        self.overall = 0
        self._init_plot()

    def __call__(self, cost):
        self.costs.append(sum(cost))
        self.overall += 1
        if self.overall % self.refresh == 0:
            self._plot(self.offset, self.costs)
            self.offset += len(self.costs)
            self.costs = []

    def _init_plot(self):
        plt.xlabel('Example')
        plt.ylabel('Cost')
        plt.ion()
        plt.show()

    def _plot(self, offset, costs):
        x = np.arange(offset, offset + len(costs))
        y = np.array(costs)
        plt.plot(x, y, linestyle='', color='blue', marker=',', alpha=0.1)
        plt.ylim(ymin=0, ymax=1.1 * max(y))
        plt.draw()
        plt.pause(0.001)
