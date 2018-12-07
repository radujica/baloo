from collections import OrderedDict

import numpy as np
from pandas import Series as PandasSeries
from tabulate import tabulate

from .generic import BinaryOps, BitOps, BalooCommon
from .indexes import Index, MultiIndex
from .utils import infer_dtype, default_index, check_type, is_scalar, check_valid_int_slice, check_weld_bit_array, \
    convert_to_numpy, check_dtype, shorten_data
from ..weld import LazyArrayResult, weld_compare, numpy_to_weld_type, weld_filter, \
    weld_slice, weld_array_op, weld_invert, weld_tail, weld_element_wise_op, LazyDoubleResult, LazyScalarResult, \
    weld_mean, weld_variance, weld_standard_deviation, WeldObject, weld_agg, weld_iloc_indices, \
    weld_iloc_indices_with_missing, weld_unique, default_missing_data_literal, weld_replace, weld_udf


class Series(LazyArrayResult, BinaryOps, BitOps, BalooCommon):
    """Weld-ed Pandas Series.

    Attributes
    ----------
    index
    dtype
    name
    iloc

    See Also
    --------
    pandas.Series : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> sr = bl.Series([0, 1, 2])
    >>> sr
    Series(name=None, dtype=int64)
    >>> sr.index
    RangeIndex(start=0, stop=3, step=1)
    >>> sr = sr.evaluate()
    >>> sr  # repr
    Series(name=None, dtype=int64)
    >>> print(sr)  # str
    <BLANKLINE>
    ---  --
      0   0
      1   1
      2   2
    >>> sr.index
    Index(name=None, dtype=int64)
    >>> print(sr.index)
    [0 1 2]
    >>> len(sr)  # eager computation
    3
    >>> sr.values
    array([0, 1, 2])
    >>> (sr + 2).evaluate().values
    array([2, 3, 4])
    >>> (sr - bl.Index(np.arange(3))).evaluate().values
    array([0, 0, 0])
    >>> print(sr.max().evaluate())
    2
    >>> print(sr.var().evaluate())
    1.0
    >>> print(sr.agg(['min', 'std']).evaluate())
    <BLANKLINE>
    ---  --
    min   0
    std   1

    """
    _empty_text = 'Empty Series'

    # TODO: Fix en/decoding string's dtype; e.g. filter returns max length dtype (|S11) even if actual result is |S7
    def __init__(self, data=None, index=None, dtype=None, name=None):
        """Initialize a Series object.

        Parameters
        ----------
        data : numpy.ndarray or WeldObject or list, optional
            Raw data or Weld expression.
        index : Index or RangeIndex or MultiIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.
        dtype : numpy.dtype or type, optional
            Desired Numpy dtype for the elements. If type, it must be a NumPy type, e.g. np.float32.
            If data is np.ndarray with a dtype different to dtype argument,
            it is astype'd to the argument dtype. Note that if data is WeldObject, one must explicitly astype
            to convert type. Inferred from `data` by default.
        name : str, optional
            Name of the Series.

        """
        data, dtype = _process_input(data, dtype)
        self.index = _process_index(index, data)
        self.dtype = dtype
        self.name = check_type(name, str)
        # TODO: this should be used to annotate Weld code for speedups
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Series, self).__init__(data, numpy_to_weld_type(self.dtype))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def iloc(self):
        """Retrieve Indexer by index.

        Supported iloc functionality exemplified below.

        Returns
        -------
        _ILocIndexer

        Examples
        --------
        >>> sr = bl.Series(np.arange(3))
        >>> print(sr.iloc[2].evaluate())
        2
        >>> print(sr.iloc[0:2].evaluate())
        <BLANKLINE>
        ---  --
          0   0
          1   1
        >>> print(sr.iloc[bl.Series(np.array([0, 2]))].evaluate())
        <BLANKLINE>
        ---  --
          0   0
          2   2

        """
        from .indexing import _ILocIndexer

        return _ILocIndexer(self)

    @property
    def str(self):
        """Get Access to string functions.

        Returns
        -------
        StringMethods

        Examples
        --------
        >>> sr = bl.Series([b' aB ', b'GoOsfrABA'])
        >>> print(sr.str.lower().evaluate())
        <BLANKLINE>
        ---  ---------
          0   ab
          1  goosfraba

        """
        if self.dtype.char != 'S':
            raise AttributeError('Can only use .str when data is of type np.bytes_')

        from .strings import StringMethods

        return StringMethods(self)

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        if self.empty:
            return self._empty_text

        # index
        str_data = OrderedDict()
        str_data.update((name, shorten_data(data.values)) for name, data in self.index._gather_data(' ').items())

        # self data
        name = '' if self.name is None else self.name
        str_data[name] = shorten_data(self.values)

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if other is None:
            other = default_missing_data_literal(self.weld_type)

            return _series_compare(self, other, comparison)
        elif is_scalar(other):
            return _series_compare(self, other, comparison)
        else:
            raise TypeError('Can currently only compare with scalars')

    def _bitwise_operation(self, other, operation):
        check_type(other, LazyArrayResult)
        check_weld_bit_array(other)
        check_weld_bit_array(self)

        return _series_array_op(self, other, operation)

    def _element_wise_operation(self, other, operation):
        if isinstance(other, LazyArrayResult):
            return _series_array_op(self, other, operation)
        elif is_scalar(other):
            return _series_element_wise_op(self, other, operation)
        else:
            raise TypeError('Can only apply operation with scalar or Series')

    def astype(self, dtype):
        check_dtype(dtype)

        return Series(self._astype(dtype),
                      self.index,
                      dtype,
                      self.name)

    def __getitem__(self, item):
        """Select from the Series.

        Supported selection functionality exemplified below.

        Examples
        --------
        >>> sr = bl.Series(np.arange(5, dtype=np.float32), name='Test')
        >>> sr = sr[sr > 0]
        >>> sr
        Series(name=Test, dtype=float32)
        >>> print(sr.evaluate())
               Test
        ---  ------
          1       1
          2       2
          3       3
          4       4
        >>> sr = sr[(sr != 1) & ~(sr > 3)]
        >>> print(sr.evaluate())
               Test
        ---  ------
          2       2
          3       3
        >>> print(sr[:1].evaluate())
               Test
        ---  ------
          2       2

        """
        if isinstance(item, LazyArrayResult):
            check_weld_bit_array(item)

            return _series_filter(self, item, self.index[item])
        elif isinstance(item, slice):
            check_valid_int_slice(item)

            return _series_slice(self, item, self.index[item])
        else:
            raise TypeError('Expected a LazyArrayResult or a slice')

    def __invert__(self):
        check_weld_bit_array(self)

        return Series(weld_invert(self.weld_expr),
                      self.index,
                      self.dtype,
                      self.name)

    # TODO: perhaps skip making a new object if data is raw already?
    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating a Series containing evaluated data and index.

        See `LazyResult`

        Returns
        -------
        Series
            Series with evaluated data and index.

        Examples
        --------
        >>> sr = bl.Series(np.arange(3)) > 0
        >>> weld_code = sr.values  # accessing values now returns the weld code as a string
        >>> sr = sr.evaluate()
        >>> sr.values  # now it is evaluated to raw data
        array([False,  True,  True])

        """
        # TODO: work on masking (branch masking) ~ evaluate the mask first and use on both index and data;
        # TODO right now the filter gets computed twice, once for index and once for the data
        evaluated_data = super(Series, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)
        evaluated_index = self.index.evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Series(evaluated_data, evaluated_index, self.dtype, self.name)

    def head(self, n=5):
        """Return Series with first n values.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        Series
            Series containing the first n values.

        Examples
        --------
        >>> sr = bl.Series(np.arange(3))
        >>> print(sr.head(2).evaluate())
        <BLANKLINE>
        ---  --
          0   0
          1   1

        """
        return self[:n]

    def tail(self, n=5):
        """Return Series with the last n values.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        Series
            Series containing the last n values.

        Examples
        --------
        >>> sr = bl.Series(np.arange(3))
        >>> print(sr.tail(2).evaluate())
        <BLANKLINE>
        ---  --
          1   1
          2   2

        """
        if self._length is not None:
            length = self._length
        else:
            length = self._lazy_len().weld_expr

        return _series_tail(self, self.index.tail(n), length, n)

    def sum(self):
        return LazyScalarResult(self._aggregate('+').weld_expr, self.weld_type)

    def prod(self):
        return LazyScalarResult(self._aggregate('*').weld_expr, self.weld_type)

    def count(self):
        return self._lazy_len()

    def mean(self):
        return LazyDoubleResult(weld_mean(self.weld_expr, self.weld_type))

    def var(self):
        return LazyDoubleResult(weld_variance(self.weld_expr, self.weld_type))

    def std(self):
        return LazyDoubleResult(weld_standard_deviation(self.weld_expr, self.weld_type))

    # TODO: currently casting everything to float64 (even if already f64 ~ weld_aggs TODO);
    # TODO maybe for min/max/count/sum/prod/.. cast ints to int64 like pandas does
    def agg(self, aggregations):
        """Multiple aggregations optimized.

        Parameters
        ----------
        aggregations : list of str
            Which aggregations to perform.

        Returns
        -------
        Series
            Series with resulting aggregations.

        """
        check_type(aggregations, list)

        new_index = Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_))

        return _series_agg(self, aggregations, new_index)

    def unique(self):
        """Return unique values in the Series.

        Note that because it is hash-based, the result will NOT be in the same order (unlike pandas).

        Returns
        -------
        LazyArrayResult
            Unique values in random order.

        """
        return LazyArrayResult(weld_unique(self.values,
                                           self.weld_type),
                               self.weld_type)

    def dropna(self):
        """Returns Series without null values according to Baloo's convention.

        Returns
        -------
        Series
            Series with no null values.

        """
        return self[self.notna()]

    def fillna(self, value):
        """Returns Series with missing values replaced with value.

        Parameters
        ----------
        value : {int, float, bytes, bool}
            Scalar value to replace missing values with.

        Returns
        -------
        Series
            With missing values replaced.

        """
        if not is_scalar(value):
            raise TypeError('Value to replace with is not a valid scalar')

        return Series(weld_replace(self.weld_expr,
                                   self.weld_type,
                                   default_missing_data_literal(self.weld_type),
                                   value),
                      self.index,
                      self.dtype,
                      self.name)

    def apply(self, func, mapping=None, new_dtype=None, **kwargs):
        """Apply an element-wise UDF to the Series.

        There are currently 6 options for using a UDF. First 4 are lazy,
        other 2 are eager and require the use of the raw decorator:

        - One of the predefined functions in baloo.functions.
        - Implementing a function which encodes the result. kwargs are automatically passed to it.
        - Pure Weld code and mapping.
        - Weld code and mapping along with a dynamically linked C++ lib containing the UDF.
        - Using a NumPy function, which however is EAGER and hence requires self.values to be raw. Additionally, NumPy
            does not support kwargs in (all) functions so must use raw decorator to strip away weld_type.
        - Implementing an eager function with the same precondition as above. Use the raw decorator to check this.

        Parameters
        ----------
        func : function or str
            Weld code as a str to encode or function from baloo.functions.
        mapping : dict, optional
            Additional mappings in the weld_template to replace on execution.
            self is added by default to reference to this Series.
        new_dtype : numpy.dtype, optional
            Specify the new dtype of the result Series.
            If None, it assumes it's the same dtype as before the apply.

        Returns
        -------
        Series
            With UDF result.

        Examples
        --------
        >>> import baloo as bl
        >>> sr = bl.Series([1, 2, 3])
        >>> weld_template = 'map({self}, |e| e + {scalar})'
        >>> mapping = {'scalar': '2L'}
        >>> print(sr.apply(weld_template, mapping).evaluate())
        <BLANKLINE>
        ---  --
          0   3
          1   4
          2   5
        >>> weld_template2 = 'map({self}, |e| e + 3L)'
        >>> print(sr.apply(weld_template2).evaluate())
        <BLANKLINE>
        ---  --
          0   4
          1   5
          2   6
        >>> print(bl.Series([1., 4., 100.]).apply(bl.sqrt).evaluate())  # lazy predefined function
        <BLANKLINE>
        ---  --
          0   1
          1   2
          2  10
        >>> sr = bl.Series([4, 2, 3, 1])
        >>> print(sr.apply(bl.sort, kind='q').evaluate())  # eager wrapper over np.sort (which uses raw decorator)
        <BLANKLINE>
        ---  --
          0   1
          1   2
          2   3
          3   4
        >>> print(sr.apply(bl.raw(np.sort, kind='q')).evaluate())  # np.sort directly
        <BLANKLINE>
        ---  --
          0   1
          1   2
          2   3
          3   4
        >>> print(sr.apply(bl.raw(lambda x: np.sort(x, kind='q'))).evaluate())  # lambda also works, with x = np.array
        <BLANKLINE>
        ---  --
          0   1
          1   2
          2   3
          3   4

        # check tests/core/cudf/* and tests/core/test_series.test_cudf for C UDF example

        """
        if callable(func):
            return Series(func(self.values,
                               weld_type=self.weld_type,
                               **kwargs),
                          self.index,
                          self.dtype,
                          self.name)
        elif isinstance(func, str):
            check_type(mapping, dict)
            check_dtype(new_dtype)

            default_mapping = {'self': self.values}
            if mapping is None:
                mapping = default_mapping
            else:
                mapping.update(default_mapping)

            if new_dtype is None:
                new_dtype = self.dtype

            return Series(weld_udf(func,
                                   mapping),
                          self.index,
                          new_dtype,
                          self.name)
        else:
            raise TypeError('Expected function or str defining a weld_template')

    @classmethod
    def from_pandas(cls, series):
        """Create baloo Series from pandas Series.

        Parameters
        ----------
        series : pandas.series.Series

        Returns
        -------
        Series

        """
        from pandas import Index as PandasIndex, MultiIndex as PandasMultiIndex

        if isinstance(series.index, PandasIndex):
            baloo_index = Index.from_pandas(series.index)
        elif isinstance(series.index, PandasMultiIndex):
            baloo_index = MultiIndex.from_pandas(series.index)
        else:
            raise TypeError('Cannot convert pandas index of type={} to baloo'.format(type(series.index)))

        return _series_from_pandas(series, baloo_index)

    def to_pandas(self):
        """Convert to pandas Series

        Returns
        -------
        pandas.series.Series

        """
        pandas_index = self.index.to_pandas()

        return _series_to_pandas(self, pandas_index)


def _process_input(data, dtype):
    if data is None:
        return np.empty(0), np.dtype(np.float64)
    else:
        check_type(data, (np.ndarray, WeldObject, list))
        check_dtype(dtype)

        if isinstance(data, list):
            data = convert_to_numpy(data)

        inferred_dtype = infer_dtype(data, dtype)
        if isinstance(data, np.ndarray) and data.dtype.char != inferred_dtype.char:
            data = data.astype(inferred_dtype)

        return data, inferred_dtype


def _process_index(index, data):
    if index is None:
        return default_index(data)
    else:
        return check_type(index, (Index, MultiIndex))


def _series_compare(series, other, comparison):
    return Series(weld_compare(series.weld_expr,
                               other,
                               comparison,
                               series.weld_type),
                  series.index,
                  np.dtype(np.bool),
                  series.name)


# the following methods are shortcuts for DataFrame ops to avoid e.g. recomputing index
def _series_agg(series, aggregations, index):
    return Series(weld_agg(series.weld_expr,
                           series.weld_type,
                           aggregations),
                  index,
                  np.dtype(np.float64))


def _series_array_op(series, other, operation):
    return Series(weld_array_op(series.weld_expr,
                                other.weld_expr,
                                series.weld_type,
                                operation),
                  series.index,
                  series.dtype,
                  series.name)


def _series_element_wise_op(series, other, operation):
    return Series(weld_element_wise_op(series.weld_expr,
                                       series.weld_type,
                                       other,
                                       operation),
                  series.index,
                  series.dtype,
                  series.name)


def _series_filter(series, item, index):
    return Series(weld_filter(series.weld_expr,
                              series.weld_type,
                              item.weld_expr),
                  index,
                  series.dtype,
                  series.name)


def _series_slice(series, item, index):
    return Series(weld_slice(series.weld_expr,
                             series.weld_type,
                             item),
                  index,
                  series.dtype,
                  series.name)


def _series_tail(series, index, length, n):
    return Series(weld_tail(series.weld_expr, length, n),
                  index,
                  series.dtype,
                  series.name)


def _series_iloc(series, item, new_index):
    return Series(weld_iloc_indices(series.weld_expr,
                                    series.weld_type,
                                    item),
                  new_index,
                  series.dtype,
                  series.name)


def _series_iloc_with_missing(series, item, new_index):
    return Series(weld_iloc_indices_with_missing(series.weld_expr,
                                                 series.weld_type,
                                                 item),
                  new_index,
                  series.dtype,
                  series.name)


def _series_from_pandas(series, baloo_index):
    return Series(series.values,
                  baloo_index,
                  series.dtype,
                  series.name)


def _series_to_pandas(series, pandas_index):
    return PandasSeries(series.values,
                        pandas_index,
                        series.dtype,
                        series.name)
