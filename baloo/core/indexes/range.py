import numpy as np
from weld.types import WeldLong
from weld.weldobject import WeldObject

from .base import Index
from ..utils import check_type
from ...weld import weld_range, LazyResult, WeldBit, weld_filter


class RangeIndex(LazyResult):
    """Weld-ed Pandas RangeIndex.

    Attributes
    ----------
    start : int
    stop : int or WeldObject
    step : int

    See also
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
        return self.weld_expr

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

    def __str__(self):
        return str(self.weld_expr)

    def __getitem__(self, item):
        if isinstance(item, LazyResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected Series of bool data to filter values')

            return Index(weld_filter(self.weld_expr,
                                     self.weld_type,
                                     item.weld_expr),
                         np.dtype(np.int64))
        else:
            raise TypeError('Expected a Series')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        evaluated_data = super(RangeIndex, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(evaluated_data, np.dtype(np.int64))
