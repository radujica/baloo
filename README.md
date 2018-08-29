# Baloo

Implementing the *bare necessities* of [Pandas](https://pandas.pydata.org/) with the *lazy* evaluating
and optimizing [Weld](https://github.com/weld-project/weld) framework.

## Install
    python setup.py install

## Develop
    // first update path to pyweld in Pipfile
    pipenv install --dev
    pipenv run pip install -e <path-to-baloo>
    pipenv run pytest
