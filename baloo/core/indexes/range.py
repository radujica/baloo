import numpy as np
from weld.types import WeldLong
from weld.weldobject import WeldObject

from .base import Index
from ..utils import Typed
from ...weld import weld_range, LazyResult


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
    start = Typed((int, WeldObject))
    stop = Typed((int, WeldObject))
    step = Typed(int)

    def __init__(self, start=None, stop=None, step=1):
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

        self._length = len(range(start, stop, step)) if isinstance(stop, int) else None

        super().__init__(weld_range(start, stop, step), WeldLong(), 1)

    @property
    def data(self):
        """Get the data representing this RangeIndex.

        Returns
        -------
        WeldObject
            Representation of this RangeIndex as a WeldObject.

        """
        return self.weld_expr

    def __len__(self):
        """Eagerly get the length of the RangeIndex.

        Note that if the length is unknown (such as for a WeldObject stop),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the RangeIndex.

        """
        if self._length is not None:
            return self._length
        else:
            return len(self.evaluate())

    def __repr__(self):
        return "{}(start={}, stop={}, step={})".format(self.__class__.__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.step)

    def __str__(self):
        return str(self.data)

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        data = super(RangeIndex, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(data, np.dtype(np.int64))
