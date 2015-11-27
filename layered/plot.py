import collections
import warnings
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook


warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


class Plot:

    def __init__(self, figure=None, refresh=0.5):
        self.figure = figure
        self.refresh = refresh
        self.max = 0
        self._init_data()
        self._init_worker()

    def __call__(self, values):
        with self.lock:
            if max(values) > self.max:
                self.max = max(values)
                self.ax.set_ylim(0, self.max)
            self.data += values
            self.li.set_ydata(self.data)

    def _axis(self):
        return 'X', 'Y'

    def _init_line(self):
        return self.ax.plot(self.data)[0]

    def _init_data(self):
        self.data = []

    def _init_plot(self):
        self.figure = self.figure or plt.figure()
        self.ax = self.figure.add_subplot(
            111, xlabel=self._axis()[0], ylabel=self._axis()[1])
        self.li = self._init_line()
        self.figure.canvas.draw()
        plt.show(block=False)

    def _init_worker(self):
        self.lock = threading.Lock()
        self.lock.acquire()
        self.thread = threading.Thread(target=self._work)
        self.thread.start()

    def _work(self):
        self._init_plot()
        self.lock.release()
        while True:
            before = time.time()
            with self.lock:
                self.figure.canvas.draw()
            duration = time.time() - before
            plt.pause(max(0.001, self.refresh - duration))


class RunningPlot(Plot):

    def __init__(self, figure=None, refresh=0.5, width=1000):
        self.width = width
        super().__init__(figure, refresh)

    def _axis(self):
        return 'Training example', 'Cost'

    def _init_line(self):
        style = {
            'linestyle': '',
            'color': 'blue',
            'marker': '.',
            'markersize': 5
        }
        return self.ax.plot(np.arange(self.width), self.data, **style)[0]

    def _init_data(self):
        self.data = collections.deque([None] * self.width, maxlen=self.width)

    def _init_plot(self):
        super()._init_plot()
        self.ax.set_xlim(0, self.width)
        self.ax.get_xaxis().set_ticks([])
