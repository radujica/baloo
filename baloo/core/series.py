import numpy as np
from weld.weldobject import WeldObject, WeldLong, WeldBit

from .generic import BinaryOps
from .indexes import RangeIndex, Index
from .utils import infer_dtype, default_index, check_type, is_scalar, valid_int_slice
from ..weld import LazyArrayResult, LazyScalarResult, weld_count, weld_compare, numpy_to_weld_type, weld_filter, \
    weld_slice, weld_array_op, weld_invert, weld_tail, weld_element_wise_op


class Series(LazyArrayResult, BinaryOps):
    """Weld-ed Pandas Series.

    Attributes
    ----------
    index : Index or RangeIndex
        Index linked to the data; it is assumed to be of the same length.
    dtype : numpy.dtype
        Numpy dtype of the elements.
    name : str
        Name of the Series.

    See Also
    --------
    pandas.Series : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> sr = bl.Series(np.arange(3))
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
    Index(name=Index, dtype=int64)
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

    """

    # TODO: when passed a dtype, pandas converts to it; do the same?
    def __init__(self, data, index=None, dtype=None, name=None):
        """Initialize a Series object.

        Parameters
        ----------
        data : numpy.ndarray or WeldObject
            Raw data or Weld expression.
        index : Index or RangeIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.
        dtype : numpy.dtype, optional
            Numpy dtype of the elements. Inferred from `data` by default.
        name : str, optional
            Name of the Series.

        """
        data = check_type(data, (np.ndarray, WeldObject))
        self.index = default_index(data) if index is None else check_type(index, (RangeIndex, Index))
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

    # TODO: maybe worthwhile to cache; chose not to init in __init__ for lazy approach
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

        """
        from .indexing import _ILocIndexer

        return _ILocIndexer(self)

    def __len__(self):
        """Eagerly get the length of the Series.

        Note that if the length is unknown (such as for a WeldObject),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the Series.

        """
        if self._length is None:
            self._length = LazyScalarResult(weld_count(self.weld_expr), WeldLong()).evaluate()

        return self._length

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def _comparison(self, other, comparison):
        if is_scalar(other):
            return Series(weld_compare(self.weld_expr,
                                       other,
                                       comparison,
                                       self.weld_type),
                          self.index,
                          np.dtype(np.bool),
                          self.name)
        else:
            raise TypeError('Can currently only compare with scalars')

    @staticmethod
    def _series_array_op(series, other, operation):
        return Series(weld_array_op(series.weld_expr,
                                    other.weld_expr,
                                    series.weld_type,
                                    operation),
                      series.index,
                      series.dtype,
                      series.name)

    def _bitwise_operation(self, other, operation):
        if not isinstance(other, LazyArrayResult):
            raise TypeError('Expected another Series')
        elif self.dtype.char != '?' or other.dtype.char != '?':
            raise TypeError('Binary operations currently supported only on bool Series')

        return Series._series_array_op(self, other, operation)

    @staticmethod
    def _series_element_wise_op(series, other, operation):
        return Series(weld_element_wise_op(series.weld_expr,
                                           series.weld_type,
                                           other,
                                           operation),
                      series.index,
                      series.dtype,
                      series.name)

    def _element_wise_operation(self, other, operation):
        if isinstance(other, LazyArrayResult):
            return Series._series_array_op(self, other, operation)
        elif is_scalar(other):
            return Series._series_element_wise_op(self, other, operation)
        else:
            raise TypeError('Can only apply operation with scalar or Series')

    @staticmethod
    def _filter_series(series, item, index):
        # shortcut of Series.__getitem__ when index is known and item is checked
        return Series(weld_filter(series.weld_expr,
                                  series.weld_type,
                                  item.weld_expr),
                      index,
                      series.dtype,
                      series.name)

    @staticmethod
    def _slice_series(series, item, index):
        # shortcut of Series.__getitem__ when index is known and item is checked
        return Series(weld_slice(series.weld_expr,
                                 series.weld_type,
                                 item),
                      index,
                      series.dtype,
                      series.name)

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
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            return Series._filter_series(self, item, self.index[item])
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            return Series._slice_series(self, item, self.index[item])
        else:
            raise TypeError('Expected a LazyResult or a slice')

    def __invert__(self):
        if self.weld_type != WeldBit():
            raise TypeError('Can only invert bool Series')

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

    @staticmethod
    def _tail_series(series, index, length, n):
        return Series(weld_tail(series.weld_expr, length, n),
                      index,
                      series.dtype,
                      series.name)

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
            length = LazyScalarResult(weld_count(self.weld_expr), WeldLong())

        return Series._tail_series(self, self.index.tail(n), length, n)
