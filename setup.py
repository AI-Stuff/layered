import sys
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test


class TestCommand(test):

    description = 'run tests, linters and create a coverage report'
    user_options = []

    def run(self):
        super().run()
        self._call(['pep8', 'layered', 'test', 'setup.py'])

    def run_tests(self):
        import pytest
        errno = pytest.main(['--cov=layered', 'test'])
        sys.exit(errno)

    def _call(self, command):
        try:
            subprocess.check_call(command)
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


if __name__ == '__main__':
    setuptools.setup(
        name='layered',
        version='0.1.1',
        description=DESCRIPTION,
        url='http://github.com/danijar/layered',
        author='Danijar Hafner',
        author_email='mail@danijar.com',
        license='MIT',
        packages=['layered'],
        setup_requires=parse_requirements('requirement/core.txt'),
        install_requires=parse_requirements('requirement/user.txt'),
        tests_require=parse_requirements('requirement/test.txt'),
        entry_points={'console_scripts': ['layered=layered.__main__:main']},
        cmdclass={'test': TestCommand, 'build_ext': BuildExtCommand},
    )
