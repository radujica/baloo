When you contribute code, you affirm that the contribution is your original work and 
that you license the work to the project under the project's open source license. 
Whether or not you state this explicitly, by submitting any copyrighted material via 
pull request, email, or other means you agree to license the material under the 
project's open source license and warrant that you have the legal authority to do so.

# Contributing

With that out of the way, if interested to contribute there are many smaller or larger
patches that need to be done:
 
- The code itself has several TODOs that can be checked/implemented. Most importantly, 
    there are several places where the Weld code can be optimized (e.g. through annotations) but
    also the Python code (e.g. by restricting the number of new objects created and caching)
- Need to setup tox and/or manylinux project for different Python versions but also linux 32bit and MAC.
    Windows would be amazing however it has proved very difficult to make it work (check Weld issues)
- Some other deployment steps could also be automated with travis, e.g. publishing documentation or pypi.
- Could also add coveralls or landscape.
- Many new features are desired and most of them are tracked in the github project. Contact me if
interested.

# Development

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
Travis is set to build Weld from scratch and test with that, however the pypi deployment contains only
specific binaries for Weld and the converters. Currently, the latest Weld master has been used.

    pipenv shell
    // build wheel distribution ~ binary
    python setup.py bdist_wheel (-p posix)?            
    // upload on pypi
    twine upload -r pypi --username <ask-me> dist/<wheel>.whl
    // check long-description aka README rendering
    twine check dist/<wheel>.whl
    
- Source code distribution, i.e. through `sdist`, not currently implemented.
- pypi-test does not have tabulate so can't test there

## Other Notes

I used Python 3.5.2 and built on Mint 64bit distribution.
