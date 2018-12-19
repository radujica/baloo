When you contribute code, you affirm that the contribution is your original work and 
that you license the work to the project under the project's open source license. 
Whether or not you state this explicitly, by submitting any copyrighted material via 
pull request, email, or other means you agree to license the material under the 
project's open source license and warrant that you have the legal authority to do so.

# Notes about development

I used Python 3.5.2, different versions require more testing ~ (TODO) setup tox.

Install pipenv:

    pip install pipenv

## Install
    git clone https://github.com/radujica/baloo.git
    pipenv install --dev                        // install all requirements
    
## Tests
    
    pipenv run pytest                           // run tests
    cd doc && make doctest                      // run doc examples
    
## Build

### Weld
    follow https://github.com/weld-project/weld#building to build Weld
    
### Convertors
    pipenv shell
    cd baloo/weld/convertors && make
    
### Documentation
* clone repo in adjacent `baloo-docs/html` directory and switch to gh-pages branch; 
    this allows the documentation to be later pushed. 
    So you should have `<dir>/baloo` and `<dir>/baloo-docs/html`
    
* back in baloo root dir:


    pipenv shell
    cd doc && make html
    
## Benchmarks
    pipenv shell
    cd benchmarks && python run.py              // correctness checks, plots, and memory profile
    
## Distribution
    pipenv shell
    python setup.py bdist_wheel                 // build wheel distribution ~ binary
    twine upload -r pypi --username <ask-me> dist/*.whl
    
- Source code distribution, i.e. through `sdist`, not currently implemented.
- pypi-test does not have tabulate so can't test there
