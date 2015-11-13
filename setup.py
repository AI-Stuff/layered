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


def parse_requirements(filename):
    with open(filename) as file_:
        lines = map(lambda x: x.strip('\n'), file_.readlines())
    lines = filter(lambda x: x and not x.startswith('#'), lines)
    return list(lines)


setuptools.setup(
    name='layered',
    version='0.1.0',
    description='Clean reference implementation of feed forward neural networks',
    url='http://github.com/danijar/layered',
    author='Danijar Hafner',
    author_email='mail@danijar.com',
    license='MIT',
    packages=['layered'],
    install_requires=parse_requirements('requirement/core.txt'),
    tests_require=parse_requirements('requirement/test.txt'),
    cmdclass={'test': TestCommand},
)
