import os
import sys
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test


class TestCommand(test):

    description = 'run tests, linters and create a coverage report'
    user_options = []

    def finalize_options(self):
        super().finalize_options()
        # New setuptools don't need this anymore, thus the try block.
        try:
            self.test_args = []
            self.test_suite = 'True'
        except AttributeError:
            pass

    def run_tests(self):
        self._call('python -m pytest --cov=layered test')
        self._call('python -m pep8 layered test setup.py')

    def _call(self, command):
        env = os.environ.copy()
        env['PYTHONPATH'] = ''.join(':' + x for x in sys.path)
        try:
            subprocess.check_call(command.split(), env=env)
        except subprocess.CalledProcessError as error:
            print('Command failed with exit code', error.returncode)
            sys.exit(error.returncode)


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


def parse_requirements(filename):
    with open(filename) as file_:
        lines = map(lambda x: x.strip('\n'), file_.readlines())
    lines = filter(lambda x: x and not x.startswith('#'), lines)
    return list(lines)


DESCRIPTION = 'Clean reference implementation of feed forward neural networks'

SETUP_REQUIRES = [
    'numpy',
]

INSTALL_REQUIRES = [
    'PyYAML',
    'numpy',
    'matplotlib',
]

TESTS_REQUIRE = [
    'pep8',
    'pytest',
    'pytest-cov',
]


if __name__ == '__main__':
    setuptools.setup(
        name='layered',
        version='0.1.5',
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
