import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook


warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


class Plot:

    def __init__(self, optimization, refresh=100, smoothing=1):
        self.optimization = optimization
        self.refresh = refresh
        self.smoothing = smoothing
        self.costs = []
        self.smoothed = []
        self._init_plot()

    def apply(self, cost):
        self.costs.append(cost)
        self.smoothed.append(np.mean(self.costs[-self.smoothing:]))
        if len(self.costs) % self.refresh == 0:
            self._refresh()

    def _init_plot(self):
        self.plot, = plt.plot([], [])
        plt.xlabel('Batch')
        plt.ylabel('Cost')
        plt.ion()
        plt.show()

    def _refresh(self):
        range_ = range(len(self.smoothed))
        plt.xlim(xmin=0, xmax=range_[-1])
        plt.ylim(ymin=0, ymax=1.1*max(self.smoothed))
        self.plot.set_data(range_, self.smoothed)
        plt.draw()
        plt.pause(0.001)
