# Baloo

Implementing the [*bare necessities*](https://www.youtube.com/watch?v=08NlhjpVFsU) 
of [Pandas](https://pandas.pydata.org/) with the *lazy* evaluating
and optimizing [Weld](https://github.com/weld-project/weld) framework.

## Benchmarks
Preliminary benchmark results over seeded randomized data are shown below. 
The generated data consists of 4 columns of different numerical types, each a NumPy array of different numerical types. 
For the first plot, the data is 280MB in total. 
The operations are:

    df = df[(df['col1'] > 0) & (df['col2'] >= 10) & (df['col3'] < 30)]  # filter
    df = df.max()                                                       # max
    df = df.agg(['min', 'prod', 'mean', 'std'])                         # 4x agg
    df['col4'] = df['col1'] * 2                                         # op
    df['col4'] = df['col1'] * 2 + 1 - 23                                # 3x op
    
![benchmark results](benchmarks/benchmarks.png)
![benchmark scalability](benchmarks/scalability.png)

Weld is, indeed, expected to scale well due to features such as vectorization, however the compilation time outweighs
the improved computation time for small datasets. Nevertheless, Baloo currently only supports a very limited subset of
Pandas. More work coming soon!

## Requirements
Ideally, a `pip install pyweld` would be enough. However, the pyweld package is not currently maintained on pypi 
so need to build ourselves:

1) Build Weld. Follow instructions [here](https://github.com/weld-project/weld). 
Latest work is being done on llvm-st branch.
2) pyweld (included in Weld ^): `pip install <path-to-pyweld-setup.py>` 

## Install ~ users
    git clone https://github.com/radujica/baloo.git
    cd baloo && python setup.py install
    
Shall be later published on pypi.

## Develop
    git clone https://github.com/radujica/baloo.git && cd baloo
    // update path to pyweld in Pipfile
    pipenv install --dev                        // install all requirements
    pipenv run pip install -e <path-to-baloo>   // install baloo in editable mode
    
    pipenv run pytest                           // run tests
    
    // the following require running through pipenv shell for correct python and path
    cd baloo/weld/convertors && make            // make the convertors
    // baloo-docs/html should have baloo repo as remote on gh-pages branch s.t. new docs can be pushed
    cd doc && make html                         // generate documentation in baloo-adjacent dir baloo-docs
    cd doc && make doctest                      // run doc examples
    cd benchmarks && python run.py              // run benchmarks ~ correctness checks, plots, and memory profile
    python setup.py sdist                       // source distribution
    python setup.py bdist_wheel                 // wheel distribution ~ built package; requires pip install wheel
