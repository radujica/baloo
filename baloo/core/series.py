import numpy as np
from weld.weldobject import WeldObject, WeldLong

from .indexes import RangeIndex, Index
from .utils import infer_dtype, default_index, check_type
from ..weld import LazyResult, weld_count


class Series(LazyResult):
    """Weld-ed Pandas Series.

    Attributes
    ----------
    index : Index or RangeIndex
        Index linked to the data; it is assumed to be of the same length.
    dtype : np.dtype
        Numpy dtype of the elements.
    name : str
        Name of the series.

    See also
    --------
    pandas.Series

    """

    # TODO: when passed a dtype, pandas converts to it; do the same?
    def __init__(self, data, index=None, dtype=None, name=None):
        """Initialize a Series object.

        Parameters
        ----------
        data : np.ndarray or WeldObject
            Raw data or Weld expression.
        index : Index or RangeIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.
        dtype : np.dtype, optional
            Numpy dtype of the elements. Inferred from `data` by default.
        name : str, optional
            Name of the Series.

        """
        self.data = check_type(data, (np.ndarray, WeldObject))
        self.index = default_index(data) if index is None else check_type(index, (RangeIndex, Index))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        # TODO: this should be used to annotate Weld code for speedups
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Series, self).__init__(data, self.dtype, 1)

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
            self._length = LazyResult(weld_count(self.data), WeldLong(), 0).evaluate()

        return self._length

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        return str(self.data)

    # TODO: perhaps skip making a new object if data is raw already?
    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        data = super(Series, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Series(data, self.index, self.dtype, self.name)
