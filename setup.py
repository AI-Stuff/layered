import os
import sys
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext as _build_ext


class TestCommand(setuptools.Command):

    description = 'run linters, tests and create a coverage report'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self._run(['pep8', 'layered', 'test', 'setup.py'])
        self._run(['py.test', '--cov=layered', 'test'])

    def _run(self, command):
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as error:
            print('Command failed with exit code', error.returncode)
            sys.exit(error.returncode)


class BuildExtCommand(_build_ext):

    def finalize_options(self):
        # Fix Numpy build error when bundled as a dependency.
        # From http://stackoverflow.com/a/21621689/1079110
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def requirements(filename):
    with open(filename) as file_:
        lines = map(lambda x: x.strip('\n'), file_.readlines())
    lines = filter(lambda x: x and not x.startswith('#'), lines)
    return list(lines)


DESCRIPTION = 'Clean reference implementation of feed forward neural networks'


if __name__ == '__main__':
    setuptools.setup(
        name='layered',
        version='0.1.0',
        description=DESCRIPTION,
        url='http://github.com/danijar/layered',
        author='Danijar Hafner',
        author_email='mail@danijar.com',
        license='MIT',
        packages=['layered'],
        setup_requies=requirements('requirement/core.txt'),
        install_requires=requirements('requirement/user.txt'),
        tests_require=requirements('requirement/test.txt'),
        cmdclass={'test': TestCommand, 'build_ext': BuildExtCommand},
        entry_points={'console_scripts': ['layered=layered.__main__:main']},
    )
