import numpy as np
from weld.weldobject import WeldObject, WeldLong, WeldBit

from ...core.utils import check_type, infer_dtype, valid_int_slice, is_scalar
from ...weld import LazyResult, numpy_to_weld_type, weld_count, weld_filter, weld_slice, weld_compare, weld_tail


class Index(LazyResult):
    """Weld-ed Pandas Index.

    Attributes
    ----------
    dtype : np.dtype
        Numpy dtype of the elements.
    name : str
        Name of the series.

    See Also
    --------
    pandas.Index : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> ind = bl.Index(np.array(['a', 'b', 'c'], dtype=np.dtype(np.bytes_)))
    >>> ind  # repr
    Index(name=Index, dtype=|S1)
    >>> print(ind)  # str
    [b'a' b'b' b'c']
    >>> ind.values
    array([b'a', b'b', b'c'], dtype='|S1')
    >>> len(ind)  # eager
    3

    """
    def __init__(self, data, dtype=None, name=None):
        """Initialize an Index object.

        Parameters
        ----------
        data : np.ndarray or WeldObject
            Raw data or Weld expression.
        dtype : np.dtype, optional
            Numpy dtype of the elements. Inferred from `data` by default.
        name : str, optional
            Name of the Index.

        """
        data = check_type(data, (np.ndarray, WeldObject))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Index, self).__init__(data, numpy_to_weld_type(self.dtype), 1)

    @property
    def name(self):
        """The name of the Index.

        Returns
        -------
        str
            The name of the Index.

        """
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def values(self):
        """The internal data representation.

        Returns
        -------
        numpy.ndarray or WeldObject
            The internal data representation.

        """
        return self.weld_expr

    def __len__(self):
        """Eagerly get the length of the Index.

        Note that if the length is unknown (such as for a WeldObject),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the Index.

        """
        if self._length is not None:
            return self._length
        else:
            return LazyResult(weld_count(self.values), WeldLong(), 0).evaluate()

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        return str(self.values)

    def _comparison(self, other, comparison):
        if is_scalar(other):
            return Index(weld_compare(self.weld_expr,
                                      other,
                                      comparison,
                                      self.weld_type),
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

    def __getitem__(self, item):
        """Select from the Index. Currently used internally through DataFrame and Series.

        Supported selection functionality exemplified below.

        Examples
        --------
        >>> ind = bl.Index(np.arange(3))
        >>> print(ind[ind < 2].evaluate())
        [0 1]
        >>> print(ind[1:2].evaluate())
        [1]

        """
        if isinstance(item, LazyResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            return Index(weld_filter(self.weld_expr,
                                     self.weld_type,
                                     item.weld_expr),
                         self.dtype,
                         self.name)
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            return Index(weld_slice(self.weld_expr,
                                    self.weld_type,
                                    item),
                         self.dtype,
                         self.name)
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
        evaluated_data = super(Index, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(evaluated_data, self.dtype, self.name)

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
        >>> ind = bl.Index(np.arange(3, dtype=np.float64))
        >>> print(ind.head(2).evaluate())
        [0. 1.]

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
        >>> ind = bl.Index(np.arange(3, dtype=np.float64))
        >>> print(ind.tail(2).evaluate())
        [1. 2.]

        """
        if self._length is not None:
            length = self._length
        else:
            length = LazyResult(weld_count(self.weld_expr), WeldLong(), 0)

        return Index(weld_tail(self.weld_expr, length, n),
                     self.dtype,
                     self.name)
