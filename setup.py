import os
import sys
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test


class TestCommand(test):

    description = 'run tests, linters and create a coverage report'
    user_options = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returncode = 0

    def finalize_options(self):
        super().finalize_options()
        # New setuptools don't need this anymore, thus the try block.
        try:
            # pylint: disable=attribute-defined-outside-init
            self.test_args = []
            self.test_suite = 'True'
        except AttributeError:
            pass

    def run_tests(self):
        self._call('python -m pytest --cov=layered test')
        self._call('python -m pylint layered')
        self._call('python -m pylint test')
        self._call('python -m pylint setup.py')
        self._check()

    def _call(self, command):
        env = os.environ.copy()
        env['PYTHONPATH'] = ''.join(':' + x for x in sys.path)
        print('Run command', command)
        try:
            subprocess.check_call(command.split(), env=env)
        except subprocess.CalledProcessError as error:
            print('Command failed with exit code', error.returncode)
            self.returncode = 1

    def _check(self):
        if self.returncode:
            sys.exit(self.returncode)


class BuildExtCommand(build_ext):
    """
    Fix Numpy build error when bundled as a dependency.
    From http://stackoverflow.com/a/21621689/1079110
    """

    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


DESCRIPTION = 'Clean reference implementation of feed forward neural networks'

SETUP_REQUIRES = [
    'numpy',
    'sphinx',
]

INSTALL_REQUIRES = [
    'PyYAML',
    'numpy',
    'matplotlib',
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pylint',
]


if __name__ == '__main__':
    setuptools.setup(
        name='layered',
        version='0.1.7',
        description=DESCRIPTION,
        url='http://github.com/danijar/layered',
        author='Danijar Hafner',
        author_email='mail@danijar.com',
        license='MIT',
        packages=['layered'],
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
        cmdclass={'test': TestCommand, 'build_ext': BuildExtCommand},
        entry_points={'console_scripts': ['layered=layered.__main__:main']},
    )
