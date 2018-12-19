from setuptools import setup, find_packages

DESCRIPTION = 'Implementing the bare necessities of Pandas with the lazy evaluating and optimizing Weld framemework.'

setup(
    name='baloo',
    description=DESCRIPTION,
    version='0.0.1',
    packages=find_packages(),
    package_data={'baloo.weld.libs': ['libweld.so', 'numpy_weld_convertor.so']},
    include_package_data=True,
    url='https://github.com/radujica/baloo',
    author='Radu Jica',
    author_email='radu.jica+code@gmail.com',
    install_requires=['pandas', 'numpy', 'tabulate'],
    platforms='any',
    python_requires='>=3.0'
)
