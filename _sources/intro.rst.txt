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


Usability
---------

Pandas is an immense library with a lot of implemented functionality. Baloo cannot replace it but only speed up
common operations through Weld. To facilitate the interaction between the libraries, the pandas API has been mimicked
as much as possible. Thus, it might be possible to simply replace the import statements from pandas to baloo. However,
there are also `from_pandas` and `to_pandas` methods available to freely move from one to the other. Nevertheless,
for baloo to be useful, most operations should go through it.


Functionality Overview
----------------------

Baloo contains documentation and examples for all the supported operations. However, as a first look into Baloo,
here is some of the implemented functionality:

* Numerical operations, such as `+` or `*`.

* Unary operations, such as `sqrt` or `log`.

* Filters, bool operations, and sub-setting, including `head` and `tail`.

* Aggregations, such as `max`, `mean`, or `std`.

* DataFrame joins, including all of `inner`, `left`, `right`, `outer`.

* GroupBy and aggregations.

* Multiple kinds of UDFs.

* To/from Pandas conversion.

* Missing values support, such as `dropna` and `fillna`.

* `iloc`.

* String manipulation through `.str` for methods, such as `split` or `contains`.

... and more.


Data Types
----------

Baloo currently accepts the following NumPy dtypes:

* int16, int32, int64

* float32, float64

* bool/bool\_ *

* S/bytes\_

Note that strings require their byte versions, therefore unicode is not currently supported (at the Weld level(?)).
* Bool is internally converted to np.bool\_ s.t. its module points to NumPy and not builtins.


Missing Data
------------

Unlike Pandas, there are currently no special NaN/NA/NaT values for missing data. If NumPy accepts it, then it's valid.
For Baloo, the following were chosen as defaults:

* floats : -999.

* integers : -999

* S/bytes\_ : b'None'

* bool : false


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

Unlike Pandas, there are currently no automatic type casts for array types, e.g. `Series([1, 2]) + Series([2.35, 3.54])`
is likely to fail. Casting is lazily available through the `astype` operator. Note that on Series creation, data can be
automatically casted to given dtype only if the data is raw. If lazy, encode it through astype.

Note that literal/scalar values are an exception and do get casted to the `dtype` of the array, e.g.
`Series([1, 2]) < 2.0` is interpreted as `Series([1, 2]) < 2`. Lastly, aggregation results get converted to `float64`
however currently post-aggregation, not before, e.g. `Series([1, 2]).sum()` first computes the integer sum
then casts the result to float.
