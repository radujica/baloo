import numpy as np

from .base import Index
from ..utils import check_type, replace_if_none
from ...weld import weld_range, WeldObject


class RangeIndex(Index):
    """Weld-ed Pandas RangeIndex.

    Attributes
    ----------
    start
    stop
    step
    dtype

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
    # TODO: implement negative step!
    def __init__(self, start=None, stop=None, step=None, name=None):
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
        self.start, self.stop, self.step = _check_input(start, stop, step)
        self.name = check_type(name, str)
        self.dtype = np.dtype(np.int64)

        self._length = len(range(self.start, self.stop, self.step)) if isinstance(stop, int) else None

        super(RangeIndex, self).__init__(weld_range(self.start, self.stop, self.step), np.dtype(np.int64))

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

    @property
    def empty(self):
        return self.start == 0 and self.stop == 0

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating an Index containing evaluated data.

        See `LazyResult`

        Returns
        -------
        Index
            Index with evaluated data.

        """
        if self.start == 0 and self.stop == 0:
            evaluated_data = np.empty(0, dtype=np.int64)
        else:
            evaluated_data = super(Index, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(evaluated_data, self.dtype, self.name)


def _check_input(start, stop, step):
    if start is None and stop is None and step is None:
        raise TypeError('Must be called with at least one integer')
    elif step is not None and step < 0:
        raise ValueError('Only positive steps are currently supported')
    elif start is not None and stop is None and step is None:
        stop = start
        start = None

    check_type(start, int)
    check_type(stop, (int, WeldObject))
    check_type(step, int)

    start = replace_if_none(start, 0)
    stop = replace_if_none(stop, 0)
    step = replace_if_none(step, 1)

    return start, stop, step
