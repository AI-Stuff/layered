import collections
import warnings
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook


warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


class Plot:

    def __init__(self, refresh=0.5, width=1000, **kwargs):
        self.refresh = refresh
        self.width = width
        self.data = collections.deque([None] * self.width, maxlen=width)
        self.max = 0
        self._init_style(**kwargs)
        self._init_update()

    def __call__(self, costs):
        with self.lock:
            if max(costs) > self.max:
                self.max = max(costs)
                self.ax.set_ylim(0, self.max)
            self.data += costs
            self.li.set_ydata(self.data)

    def _init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(
            111, xlabel='Training example', ylabel='Cost')
        self.li, = self.ax.plot(np.arange(self.width), self.data, **self.style)
        self.ax.set_xlim(0, self.width)
        self.ax.get_xaxis().set_ticks([])
        self.fig.canvas.draw()
        plt.show(block=False)

    def _init_update(self):
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update)
        self.thread.start()

    def _init_style(self, **kwargs):
        self.style = {
            'linestyle': '',
            'color': 'blue',
            'marker': '.',
            'markersize': 5
        }
        self.style.update(kwargs)

    def _update(self):
        with self.lock:
            self._init_plot()
        while True:
            before = time.time()
            with self.lock:
                self.fig.canvas.draw()
            duration = time.time() - before
            plt.pause(max(0.001, self.refresh - duration))
