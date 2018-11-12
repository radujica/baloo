import numpy as np

from ..generic import BinaryOps, IndexCommon, BalooCommon
from ...core.utils import check_type, infer_dtype, is_scalar, check_weld_bit_array, check_valid_int_slice
from ...weld import LazyArrayResult, numpy_to_weld_type, weld_filter, weld_slice, \
    weld_compare, weld_tail, weld_array_op, weld_element_wise_op, WeldObject, weld_iloc_indices, \
    weld_iloc_indices_with_missing


class Index(LazyArrayResult, BinaryOps, IndexCommon, BalooCommon):
    """Weld-ed Pandas Index.

    Attributes
    ----------
    dtype
    name

    See Also
    --------
    pandas.Index : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> ind = bl.Index(np.array(['a', 'b', 'c'], dtype=np.dtype(np.bytes_)))
    >>> ind  # repr
    Index(name=None, dtype=|S1)
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
        check_type(data, (np.ndarray, WeldObject))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Index, self).__init__(data, numpy_to_weld_type(self.dtype))

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

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

    def _bitwise_operation(self, other, operation):
        check_type(other, LazyArrayResult)
        check_weld_bit_array(other)
        check_weld_bit_array(self)

        return Index(weld_array_op(self.weld_expr,
                                   other.weld_expr,
                                   self.weld_type,
                                   operation),
                     self.dtype,
                     self.name)

    def _element_wise_operation(self, other, operation):
        # Pandas converts result to a Series; unclear why atm
        if isinstance(other, LazyArrayResult):
            return Index(weld_array_op(self.weld_expr,
                                       other.weld_expr,
                                       self.weld_type,
                                       operation),
                         self.dtype,
                         self.name)
        elif is_scalar(other):
            return Index(weld_element_wise_op(self.weld_expr,
                                              self.weld_type,
                                              other,
                                              operation),
                         self.dtype,
                         self.name)
        else:
            raise TypeError('Can only apply operation with scalar or LazyArrayResult')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def _gather_names(self, name='index'):
        return [name if self.name is None else self.name]

    def _gather_data_for_weld(self):
        return [self.weld_expr]

    def _gather_weld_types(self):
        return [self.weld_type]

    def _gather_data(self, name='index'):
        return {self._gather_names(name)[0]: self}

    def _iloc_indices(self, indices):
        return Index(weld_iloc_indices(self.weld_expr,
                                       self.weld_type,
                                       indices),
                     self.dtype,
                     self.name)

    def _iloc_indices_with_missing(self, indices):
        return Index(weld_iloc_indices_with_missing(self.weld_expr,
                                                    self.weld_type,
                                                    indices),
                     self.dtype,
                     self.name)

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
        if isinstance(item, LazyArrayResult):
            check_weld_bit_array(item)

            return Index(weld_filter(self.weld_expr,
                                     self.weld_type,
                                     item.weld_expr),
                         self.dtype,
                         self.name)
        elif isinstance(item, slice):
            check_valid_int_slice(item)
            if self.empty:
                return self
            else:
                return Index(weld_slice(self.weld_expr,
                                        self.weld_type,
                                        item),
                             self.dtype,
                             self.name)
        else:
            raise TypeError('Expected LazyArrayResult or slice')

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
        if self.empty:
            return self
        else:
            if self._length is not None:
                length = self._length
            else:
                length = self._lazy_len().weld_expr

            # not computing slice here to use with __getitem__ because we'd need to use len which is eager
            return Index(weld_tail(self.weld_expr, length, n),
                         self.dtype,
                         self.name)
