import numpy as np
from weld.types import WeldLong
from weld.weldobject import WeldObject

from .base import Index
from ..utils import check_type, valid_int_slice
from ...weld import weld_range, LazyResult, WeldBit, weld_filter, weld_slice, weld_compare, weld_count, weld_tail


class RangeIndex(LazyResult):
    """Weld-ed Pandas RangeIndex.

    Attributes
    ----------
    start : int
    stop : int or WeldObject
    step : int

    See Also
    --------
    pandas.RangeIndex

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

        self._length = len(range(start, stop, step)) if isinstance(stop, int) else None

        super(RangeIndex, self).__init__(weld_range(start, stop, step), WeldLong(), 1)

    @property
    def values(self):
        """The internal data representation.

        Returns
        -------
        numpy.ndarray or WeldObject
            The internal data representation.

        """
        return self.weld_expr

    @property
    def name(self):
        """The name of the RangeIndex.

        Returns
        -------
        str
            The name of the RangeIndex.

        """
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

    def __str__(self):
        return str(self.weld_expr)

    def _comparison(self, other, comparison):
        if isinstance(other, int):
            return Index(weld_compare(self.weld_expr,
                                      other,
                                      comparison,
                                      self.weld_type),
                         np.dtype(np.bool))
        else:
            raise TypeError('Can only compare with integers')

    def __lt__(self, other):
        return self._comparison(other, '<')

    def __le__(self, other):
        return self._comparison(other, '<=')

    def __eq__(self, other):
        return self._comparison(other, '==')

    def __ne__(self, other):
        return self._comparison(other, '!=')

    def __ge__(self, other):
        return self._comparison(other, '>=')

    def __gt__(self, other):
        return self._comparison(other, '>')

    def __getitem__(self, item):
        """Select from the RangeIndex. Currently used internally through DataFrame and Series.

        Supported functionality:

        - Filter: ind[ind <comparison> <scalar>]
        - Slice: ind[<start>:<stop>:<step>]

        """
        if isinstance(item, LazyResult):
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

        """
        if self._length is not None:
            length = self._length
        else:
            length = LazyResult(weld_count(self.weld_expr), WeldLong(), 0)

        return Index(weld_tail(self.weld_expr, length, n),
                     np.dtype(np.int64))
