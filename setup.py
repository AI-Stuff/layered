import sys
import subprocess
import setuptools


class TestCommand(setuptools.Command):

    description = 'run linters, tests and create a coverage report'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self._run(['pep8', 'layered', 'test'])
        self._run(['py.test', '--cov=layered', 'test'])

    def _run(self, command):
        try:
            subprocess.call(command)
        except subprocess.CalledProcessError as error:
            print('Command failed with exit code', error.returncode)
            sys.exit(error.returncode)


install_requires = [
    'PyYAML',
    'numpy',
    'matplotlib',
]


tests_require = [
    'pep8',
    'pytest',
    'pytest-cov',
    'coveralls',
]


cmdclass = {
    'test': TestCommand,
}


setuptools.setup(
    name='layered',
    version='0.1.0',
    description='Clean reference implementation of feed forward neural networks',
    url='http://github.com/danijar/layered',
    author='Danijar Hafner',
    author_email='mail@danijar.com',
    license='MIT',
    packages=['layered'],
    install_requires=install_requires + tests_require,
    cmdclass=cmdclass,
)
