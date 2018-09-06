Intro to Baloo
==============

Baloo implements a subset of `Pandas <https://pandas.pydata.org/>`_ maintaining its behavior and API as much as possible.
However, the library is now lazily implemented using Weld. All operations are tracked through
`Weld <https://github.com/weld-project/weld>`_ computation graphs which are optimized
upon evaluation. In practice, this means that the `evaluate` method must be called whenever a result is required.
The core data structure at the surface level (API) is still the Numpy
`ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_,
despite it being converted to other structures internally by Weld. For efficiency reasons, only arrays of 1 dimension
are expected (please check `baloo/weld/convertors/numpy.cpp` for malloc reasons why).


Data Types
----------

Baloo currently accepts the following NumPy dtypes:

* int16, int32, int64

* float32, float64

* bool

* S/bytes\_

Note that strings require their byte versions, therefore unicode is not currently supported (at the Weld level(?)).


Missing Data
------------

Unlike Pandas, there are currently no special NaN/NA/NaT values for missing data. If NumPy accepts it, then it's valid.
In practice, this means that:

* floats have the special `np.nan` value,

* integers must use some arbitrary value, e.g. `-999`

* same for strings, e.g. `b'None'`

* booleans can use `False`


Lazy Evaluation
---------------

All Baloo objects follow the following contract:

* Calling `repr()` will return a lazy-friendly shortened description of the object. For example, calling it on a Series \
  would return `Series(name=<name>, dtype=<dtype>)` with no mention of the actual data inside. This is because the data \
  can be either raw data (which gets printed nicely) or a string Weld computation graph (which, well, does not).

* Calling `str()` will pretty print the object. Note that for DataFrame, it cannot be called unless all the data is evaluated.

* `.values` property returns the internal data, be it raw or lazy.

* `evaluate()` returns a copy of the object with raw evaluated data inside.


Type Casting
------------

Unlike Pandas, there are currently no automatic type casts for array types, e.g. `[1, 2] + [2.35, 3.54]` is likely to
fail. Similarly, when creating a Series using data of some dtype but specifying a different dtype in the constructor,
it will *not* convert to that dtype. These are usability details that shall be tackled later.

Note, however, that literal/scalar values are an exception and do get casted to the `dtype` of the array, e.g.
`[1, 2] < 2.0` is interpreted as `[1, 2] < 2`. Lastly, aggregation results get converted to `float64` if possible, like
in Pandas.
