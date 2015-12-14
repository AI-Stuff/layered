# pylint: disable=wrong-import-position
import collections
import time
import warnings
import inspect
import threading
import matplotlib

# Don't call the code if Sphinx inspects the file mocking external imports.
if inspect.ismodule(matplotlib):  # noqa
    # On Mac force backend that works with threading.
    if matplotlib.get_backend() == 'MacOSX':
        matplotlib.use('TkAgg')
    # Hide matplotlib deprecation message.
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
    # Ensure available interactive backend.
    if matplotlib.get_backend() not in matplotlib.rcsetup.interactive_bk:
        print('No visual backend available. Maybe you are inside a virtualenv '
              'that was created without --system-site-packages.')

import matplotlib.pyplot as plt


class Interface:

    def __init__(self, title='', xlabel='', ylabel='', style=None):
        self._style = style or {}
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self.xdata = []
        self.ydata = []
        self.width = 0
        self.height = 0

    @property
    def style(self):
        return self._style

    @property
    def title(self):
        return self._title

    @property
    def xlabel(self):
        return self._xlabel

    @property
    def ylabel(self):
        return self._ylabel


class State:

    def __init__(self):
        self.running = False


class Window:

    def __init__(self, refresh=0.5):
        self.refresh = refresh
        self.thread = None
        self.state = State()
        self.figure = plt.figure()
        self.interfaces = []
        plt.ion()
        plt.show()

    def register(self, position, interface):
        axis = self.figure.add_subplot(
            position, title=interface.title,
            xlabel=interface.xlabel, ylabel=interface.ylabel)
        axis.get_xaxis().set_ticks([])
        line, = axis.plot(interface.xdata, interface.ydata, **interface.style)
        self.interfaces.append((axis, line, interface))

    def start(self, work):
        """
        Hand the main thread to the window and continue work in the provided
        function. A state is passed as the first argument that contains a
        `running` flag. The function is expected to exit if the flag becomes
        false. The flag can also be set to false to stop the window event loop
        and continue in the main thread after the `start()` call.
        """
        assert threading.current_thread() == threading.main_thread()
        assert not self.state.running
        self.state.running = True
        self.thread = threading.Thread(target=work, args=(self.state,))
        self.thread.start()
        while self.state.running:
            try:
                before = time.time()
                self.update()
                duration = time.time() - before
                plt.pause(max(0.001, self.refresh - duration))
            except KeyboardInterrupt:
                self.state.running = False
                self.thread.join()
                return

    def stop(self):
        """
        Close the window and stops the worker thread. The main thread will
        resume with the next command after the `start()` call.
        """
        assert threading.current_thread() == self.thread
        assert self.state.running
        self.state.running = False

    def update(self):
        """
        Redraw the figure to show changed data. This is automatically called
        after `start()` was run.
        """
        assert threading.current_thread() == threading.main_thread()
        for axis, line, interface in self.interfaces:
            line.set_xdata(interface.xdata)
            line.set_ydata(interface.ydata)
            axis.set_xlim(0, interface.width or 1, emit=False)
            axis.set_ylim(0, interface.height or 1, emit=False)
        self.figure.canvas.draw()


class Plot(Interface):

    def __init__(self, title, xlabel, ylabel, style=None, fixed=None):
        # pylint: disable=too-many-arguments, redefined-variable-type
        super().__init__(title, xlabel, ylabel, style or {})
        self.max_ = 0
        if not fixed:
            self.xdata = []
            self.ydata = []
        else:
            self.xdata = list(range(fixed))
            self.ydata = collections.deque([None] * fixed, maxlen=fixed)
            self.width = fixed

    def __call__(self, values):
        self.ydata += values
        self.max_ = max(self.max_, *values)
        self.height = 1.05 * self.max_
        while len(self.xdata) < len(self.ydata):
            self.xdata.append(len(self.xdata))
        self.width = len(self.xdata) - 1
        assert len(self.xdata) == len(self.ydata)
