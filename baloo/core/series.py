import numpy as np
from weld.weldobject import WeldObject, WeldLong, WeldBit

from .indexes import RangeIndex, Index
from .utils import infer_dtype, default_index, check_type, is_scalar
from ..weld import LazyResult, weld_count, weld_compare, numpy_to_weld_type, weld_filter


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
        data = check_type(data, (np.ndarray, WeldObject))
        self.index = default_index(data) if index is None else check_type(index, (RangeIndex, Index))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        # TODO: this should be used to annotate Weld code for speedups
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Series, self).__init__(data, numpy_to_weld_type(self.dtype), 1)

    @property
    def values(self):
        return self.weld_expr

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
            self._length = LazyResult(weld_count(self.values), WeldLong(), 0).evaluate()

        return self._length

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        return str(self.weld_expr)

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

    @staticmethod
    def _filter_series(series, item, index):
        # shortcut of Series.__getitem__ when index is known and item is checked
        return Series(weld_filter(series.weld_expr,
                                  series.weld_type,
                                  item.weld_expr),
                      index,
                      series.dtype,
                      series.name)

    def __getitem__(self, item):
        if isinstance(item, LazyResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            new_index = self.index[item]

            return Series._filter_series(self, item, new_index)
        else:
            raise TypeError('Expected a LazyResult')

    # TODO: perhaps skip making a new object if data is raw already?
    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        evaluated_data = super(Series, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)
        evaluated_index = self.index.evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Series(evaluated_data, evaluated_index, self.dtype, self.name)
