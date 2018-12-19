import os

from setuptools import setup, find_packages


def read(name):
    return open(os.path.join(os.path.dirname(__file__), name)).read()


setup(
    name='baloo',
    description='Implementing the bare necessities of Pandas with the lazy evaluating and optimizing Weld framework.',
    long_description=read('README.md'),
    version='0.0.2',
    license='BSD 3-Clause',
    packages=find_packages(),
    package_data={'baloo.weld.libs': ['libweld.so', 'numpy_weld_convertor.so']},
    include_package_data=True,
    url='https://github.com/radujica/baloo',
    author='Radu Jica',
    author_email='radu.jica+code@gmail.com',
    install_requires=['pandas', 'numpy', 'tabulate'],
    platforms='linux',
    python_requires='>=3.0'
)
