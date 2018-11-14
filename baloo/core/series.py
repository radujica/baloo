import numpy as np

from .generic import BinaryOps, BitOps, BalooCommon
from .indexes import Index, MultiIndex
from .utils import infer_dtype, default_index, check_type, is_scalar, check_valid_int_slice, check_weld_bit_array, \
    convert_to_numpy
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
    [0 1 2]
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
    [0. 1.]

    """
    # TODO: when passed a dtype, pandas converts to it; do the same?
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
        dtype : numpy.dtype, optional
            Numpy dtype of the elements. Inferred from `data` by default.
        name : str, optional
            Name of the Series.

        """
        data = _process_input_data(data)
        self.index = _process_index(index, data)
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
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

        Examples
        --------
        >>> sr = bl.Series(np.arange(3))
        >>> print(sr.iloc[2].evaluate())
        2
        >>> print(sr.iloc[0:2].evaluate())
        [0 1]
        >>> print(sr.iloc[bl.Series(np.array([0, 2]))].evaluate())
        [0 2]

        """
        from .indexing import _ILocIndexer

        return _ILocIndexer(self)

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

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
        [1. 2. 3. 4.]
        >>> sr = sr[(sr != 1) & ~(sr > 3)]
        >>> print(sr.evaluate())
        [2. 3.]
        >>> print(sr[:1].evaluate())
        [2.]

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
        [0 1]

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
        [1 2]

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

    def apply(self, weld_template, mapping=None, new_dtype=None):
        """Apply an element-wise UDF to the Series.

        Parameters
        ----------
        weld_template : str
            Weld code to execute.
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
        [3 4 5]
        >>> weld_template2 = 'map({self}, |e| e + 3L)'
        >>> print(sr.apply(weld_template2).evaluate())
        [4 5 6]

        """
        check_type(weld_template, str)
        check_type(mapping, dict)
        check_type(new_dtype, np.dtype)

        default_mapping = {'self': self.values}
        if mapping is None:
            mapping = default_mapping
        else:
            mapping.update(default_mapping)

        if new_dtype is None:
            new_dtype = self.dtype

        return Series(weld_udf(weld_template,
                               mapping),
                      self.index,
                      new_dtype,
                      self.name)


def _process_input_data(data):
    if data is None:
        return np.empty(0)
    else:
        check_type(data, (np.ndarray, WeldObject, list))

        if isinstance(data, list):
            data = convert_to_numpy(data)

        return data


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
