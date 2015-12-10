import collections
import inspect
import threading
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt


def init_matplotlib():
    # Hide matplotlib deprecation message.
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
    # Ensure available interactive backend.
    if matplotlib.get_backend() not in matplotlib.rcsetup.interactive_bk:
        print('No visual backend available. Maybe you are inside a virtualenv '
              'that was created without --system-site-packages.')


# Don't call the code if Sphinx inspects the file mocking external imports.
if inspect.ismodule(matplotlib):
    init_matplotlib()


class Window:

    def __init__(self, refresh=0.5):
        self.refresh = refresh
        self._init_worker()

    def plot(self, *args, **kwargs):
        return Plot(self.figure, self.lock, *args, **kwargs)

    def _init_worker(self):
        self.lock = threading.Lock()
        self.lock.acquire()
        self.thread = threading.Thread(target=self._work)
        self.thread.start()
        with self.lock:
            return

    def _init_figure(self):
        self.figure = plt.figure()
        plt.show(block=False)

    def _work(self):
        self._init_figure()
        self.lock.release()
        while True:
            before = time.time()
            with self.lock:
                self.figure.canvas.draw()
            duration = time.time() - before
            plt.pause(max(0.001, self.refresh - duration))


class Plot:

    STYLES = {
        'dot': {
            'linestyle': '',
            'color': 'blue',
            'marker': '.',
            'markersize': 5,
        },
        'line': {},
    }

    def __init__(self, figure, lock, tile=111, style='line',
                 title='', xlabel='', ylabel='', fixed=None):
        assert style in type(self).STYLES
        self.figure = figure
        self.lock = lock
        self.height = 0
        self.fixed = fixed
        if fixed:
            self.xdata = list(range(fixed))
            self.ydata = collections.deque([None] * fixed, maxlen=fixed)
            self.width = fixed
        else:
            self.xdata = []
            self.ydata = []
            self.width = 0
        styles = type(self).STYLES[style]
        with self.lock:
            self.ax = self.figure.add_subplot(
                tile, title=title, xlabel=xlabel, ylabel=ylabel)
            self.ax.get_xaxis().set_ticks([])
            self.line, = self.ax.plot(self.xdata, self.ydata, **styles)

    def __call__(self, values):
        self.ydata += values
        self.height = max(self.height, *values)
        if not self.fixed:
            self.xdata += [x + len(self.xdata) for x in range(len(values))]
            self.width += len(values)
        assert len(self.xdata) == len(self.ydata) == self.width
        with self.lock:
            self.line.set_xdata(self.xdata)
            self.line.set_ydata(self.ydata)
            self.ax.set_xlim(0, max(1, self.width - 1), emit=False)
            self.ax.set_ylim(0, 1.05 * self.height, emit=False)
