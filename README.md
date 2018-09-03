# Baloo

Implementing the [*bare necessities*](https://www.youtube.com/watch?v=08NlhjpVFsU) 
of [Pandas](https://pandas.pydata.org/) with the *lazy* evaluating
and optimizing [Weld](https://github.com/weld-project/weld) framework.

## Requirements
Ideally, a `pip install pyweld` would be enough. However, the pyweld package is not currently maintained on pypi 
so need to build ourselves:

1) Weld. Follow instructions [here](https://github.com/weld-project/weld). Latest work is being done on llvm-st branch.
2) pyweld. Baloo currently requires [this](https://github.com/radujica/weld/tree/pyweld3) branch for Python 3 support. 
For Python 2, main Weld repo pyweld should be fine. 

## Install ~ users
    pip install <path-to-pyweld>
    git clone https://github.com/radujica/baloo.git
    cd baloo && python setup.py install
    
Shall be later published on pypi.

## Develop
    git clone https://github.com/radujica/baloo.git && cd baloo
    // update path to pyweld in Pipfile
    pipenv install --dev                        // install all requirements
    pipenv run pip install -e <path-to-baloo>   // install baloo in editable mode
    // making the convertors requires running through correct python version, i.e. through pipenv shell
    
    pipenv run pytest                           // run tests
    
    cd doc && make html                         // generate documentation in baloo-adjacent dir baloo-docs
