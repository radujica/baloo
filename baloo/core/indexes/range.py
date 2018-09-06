import numpy as np
from weld.types import WeldLong
from weld.weldobject import WeldObject

from .base import Index
from ..generic import BinaryOps
from ..utils import check_type, valid_int_slice
from ...weld import weld_range, LazyArrayResult, LazyScalarResult, WeldBit, weld_filter, weld_slice, weld_compare, \
    weld_count, weld_tail, weld_array_op


class RangeIndex(LazyArrayResult, BinaryOps):
    """Weld-ed Pandas RangeIndex.

    Attributes
    ----------
    start : int
    stop : int or WeldObject
    step : int
    dtype : np.dtype
        Always int64.

    See Also
    --------
    pandas.RangeIndex : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.RangeIndex.html#pandas.RangeIndex

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> ind = bl.RangeIndex(3)
    >>> ind  # repr
    RangeIndex(start=0, stop=3, step=1)
    >>> weld_code = str(ind)  # weld_code
    >>> ind.evaluate()
    Index(name=Index, dtype=int64)
    >>> print(ind.evaluate())
    [0 1 2]
    >>> len(ind)  # eager
    3

    """
    def __init__(self, start=None, stop=None, step=1, name=None):
        """Initialize a RangeIndex object.

        If only 1 value (`start`) is passed, it will be considered the `stop` value.
        Note that this 1 value may also be a WeldObject for cases such as creating
        a Series with no index as argument.

        Parameters
        ----------
        start : int or WeldObject
        stop : int or WeldObject, optional
        step : int, optional

        """
        if start is None and stop is None and step == 1:
            raise ValueError('Must supply at least one integer')
        # allow pd.RangeIndex(123) to represent pd.RangeIndex(0, 123, 1)
        elif start is not None and stop is None and step == 1:
            stop = start
            start = 0

        self.start = check_type(start, int)
        self.stop = check_type(stop, (int, WeldObject))
        self.step = check_type(step, int)
        self.name = check_type(name, str)
        self.dtype = np.dtype(np.int64)

        self._length = len(range(start, stop, step)) if isinstance(stop, int) else None

        super(RangeIndex, self).__init__(weld_range(start, stop, step), WeldLong())

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __len__(self):
        """Eagerly get the length of the RangeIndex.

        Note that if the length is unknown (such as for a WeldObject stop),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the RangeIndex.

        """
        if self._length is None:
            self._length = len(self.evaluate())

        return self._length

    def __repr__(self):
        return "{}(start={}, stop={}, step={})".format(self.__class__.__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.step)

    def _comparison(self, other, comparison):
        if isinstance(other, int):
            return Index(weld_compare(self.weld_expr,
                                      other,
                                      comparison,
                                      self.weld_type),
                         np.dtype(np.bool))
        else:
            raise TypeError('Can only compare with integers')

    def _bitwise_operation(self, other, operation):
        if not isinstance(other, Index):
            raise TypeError('Expected another Series')
        elif self.dtype.char != '?' or other.dtype.char != '?':
            raise TypeError('Binary operations currently supported only on bool Series')

        return Index(weld_array_op(self.weld_expr,
                                   other.weld_expr,
                                   self.weld_type,
                                   operation),
                     self.dtype,
                     self.name)

    def __getitem__(self, item):
        """Select from the RangeIndex. Currently used internally through DataFrame and Series.

        Examples
        --------
        >>> ind = bl.RangeIndex(3)
        >>> print(ind[ind < 2].evaluate())
        [0 1]
        >>> print(ind[1:2].evaluate())
        [1]

        """
        if isinstance(item, LazyArrayResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            return Index(weld_filter(self.weld_expr,
                                     self.weld_type,
                                     item.weld_expr),
                         np.dtype(np.int64))
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            return Index(weld_slice(self.weld_expr,
                                    self.weld_type,
                                    item),
                         np.dtype(np.int64))
        else:
            raise TypeError('Expected a LazyResult or a slice')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating an Index containing evaluated data.

        See `LazyResult`

        Returns
        -------
        Index
            Index with evaluated data.

        """
        evaluated_data = super(RangeIndex, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(evaluated_data, np.dtype(np.int64))

    def head(self, n=5):
        """Return Index with first n values.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        Series
            Index containing the first n values.

        Examples
        --------
        >>> ind = bl.RangeIndex(3)
        >>> print(ind.head(2).evaluate())
        [0 1]

        """
        return self[:n]

    def tail(self, n=5):
        """Return Index with the last n values.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        Series
            Index containing the last n values.

        Examples
        --------
        >>> ind = bl.RangeIndex(3)
        >>> print(ind.tail(2).evaluate())
        [1 2]

        """
        if self._length is not None:
            length = self._length
        else:
            length = LazyScalarResult(weld_count(self.weld_expr), WeldLong())

        return Index(weld_tail(self.weld_expr, length, n),
                     np.dtype(np.int64))
