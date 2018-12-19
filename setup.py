import subprocess
import sys

from setuptools import setup, Distribution, find_packages
from setuptools.command.install import install


class Install(install):
    def run(self):
        install.run(self)
        python_executable = sys.executable
        protoc_command_clean = ["make -C " + self.install_lib + "baloo/weld/convertors clean"]
        if subprocess.call(protoc_command_clean, shell=True) != 0:
            sys.exit(-1)

        protoc_command_make = ["make -C " + self.install_lib + "baloo/weld/convertors/ EXEC=" + python_executable]
        if subprocess.call(protoc_command_make, shell=True) != 0:
            sys.exit(-1)


class BinaryDistribution(Distribution):
    @staticmethod
    def has_ext_modules():
        return True


DESCRIPTION = 'Implementing the bare necessities of Pandas with the lazy evaluating and optimizing Weld framemework.'

setup(name='baloo',
      description=DESCRIPTION,
      version='0.0.1',
      packages=find_packages(),
      package_data={'': ['numpy.cpp', 'common.h', 'Makefile']},
      include_package_data=True,
      cmdclass={"install": Install},
      distclass=BinaryDistribution,
      url='https://github.com/radujica/baloo',
      author='Radu Jica',
      author_email='radu.jica+code@gmail.com',
      install_requires=['pandas', 'numpy', 'tabulate'],
      platforms='any',
      python_requires='>=3.0')
