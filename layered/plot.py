import warnings
import threading
import time
import numpy as np
import pylab as pl
import matplotlib.cbook


warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


class Plot:

    def __init__(self, refresh=1):
        self.refresh = refresh
        self.count = 0
        self.max_ = 0
        self._init_style()
        self._init_update()

    def __call__(self, cost):
        cost = cost.sum()
        self.max_ = max(self.max_, cost)
        with self.lock:
            pl.plot(self.count, cost, **self.style)
        self.count += 1

    def _init_style(self):
        self.style = {
            'color': 'blue',
            'marker': '.',
            'markersize': 2
        }

    def _init_update(self):
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update)
        self.thread.start()

    def _init_plot(self):
        with self.lock:
            pl.xlabel('Example')
            pl.ylabel('Cost')
            pl.show(block=False)

    def _update(self):
        self._init_plot()
        while True:
            before = time.time()
            with self.lock:
                pl.ylim(0, 1.05 * self.max_)
                pl.xlim(0, self.count)
                pl.draw()
            duration = time.time() - before
            pl.pause(max(0, self.refresh - duration))
