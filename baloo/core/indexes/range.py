import numpy as np
from weld.weldobject import WeldObject

from .base import Index
from ..utils import check_type
from ...weld import weld_range


class RangeIndex(Index):
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
    Index(name=None, dtype=int64)
    >>> print(ind.evaluate())
    [0 1 2]
    >>> len(ind)  # eager
    3
    >>> (ind * 2).evaluate().values
    array([0, 2, 4])
    >>> (ind - bl.Series(np.arange(1, 4))).evaluate().values
    array([-1, -1, -1])

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

        super(RangeIndex, self).__init__(weld_range(start, stop, step), np.dtype(np.int64))

    def __repr__(self):
        return "{}(start={}, stop={}, step={})".format(self.__class__.__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.step)

    def _comparison(self, other, comparison):
        if isinstance(other, int):
            return super(RangeIndex, self)._comparison(other, comparison)
        else:
            raise TypeError('Can only compare with integers')
