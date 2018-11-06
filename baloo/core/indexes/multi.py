from collections import OrderedDict

import numpy as np
from tabulate import tabulate

from .base import Index
from ..generic import IndexCommon, BalooCommon
from ..utils import check_inner_types, check_type, infer_length, shorten_data, check_weld_bit_array, \
    check_valid_int_slice
from ...weld import LazyArrayResult


class MultiIndex(IndexCommon, BalooCommon):
    """Weld-ed MultiIndex, however completely different to Pandas.

    This version merely groups a few columns together to act as an index
    and hence does not follow the labels/levels approach of Pandas.

    Attributes
    ----------
    data : list of Index
        The internal data.
    names : list of str
        The names of the data.
    dtypes : list of numpy.dtype
        The Numpy dtypes of the data in the same order
        as the data and the names themselves.

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> ind = bl.MultiIndex([np.array([1, 2, 3]), np.array([4, 5, 6], dtype=np.float64)], names=['i1', 'i2'])
    >>> ind  # repr
    MultiIndex(names=['i1', 'i2'], dtypes=[dtype('int64'), dtype('float64')])
    >>> print(ind)  # str
      i1    i2
    ----  ----
       1     4
       2     5
       3     6
    >>> ind.values
    [Index(name=i1, dtype=int64), Index(name=i2, dtype=float64)]
    >>> len(ind)  # eager
    3

    """
    @staticmethod
    def _init_indexes(data, names):
        if names is not None:
            if len(data) != len(names):
                raise ValueError('Expected all or none of the data columns to be named')
        else:
            names = [None] * len(data)

        data_as_indexes = []

        for n, v in zip(names, data):
            if isinstance(v, np.ndarray):
                v = Index(v, v.dtype, n)

            data_as_indexes.append(v)

        return data_as_indexes

    def __init__(self, data, names=None):
        """Initialize a MultiIndex object.

        Parameters
        ----------
        data : list of numpy.ndarray or list of Index
            The internal data.
        names : list of str, optional
            The names of the data.

        """
        check_inner_types(check_type(data, list), (np.ndarray, Index))
        self._length = infer_length(data)
        self.names = check_inner_types(check_type(names, list), str)
        self._data = MultiIndex._init_indexes(data, names)

    @property
    def values(self):
        """Retrieve internal data.

        Returns
        -------
        list
            The internal list data representation.

        """
        return self._data

    @property
    def dtypes(self):
        return [v.dtype for v in self.values]

    def __len__(self):
        """Eagerly get the length of the MultiIndex.

        Note that if the length is unknown (such as for WeldObjects),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the MultiIndex.

        """
        if self._length is not None:
            return self._length
        else:
            # first check again for raw data
            length = infer_length(self.values)
            if length is None:
                # empty DataFrame
                if len(self.values) == 0:
                    return 0

                # use the first column to compute the length
                length = len(self.values[0])

            self._length = length

            return length

    def __repr__(self):
        return "{}(names={}, dtypes={})".format(self.__class__.__name__,
                                                self.names,
                                                self.dtypes)

    def __str__(self):
        str_data = OrderedDict(((k, shorten_data(v.values)) for k, v in zip(self.names, self.values)))

        return tabulate(str_data, headers='keys')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating a MultiIndex containing evaluated data and index.

        See `LazyResult`

        Returns
        -------
        MultiIndex
            MultiIndex with evaluated data.

        """
        evaluated_data = [v.evaluate(verbose, decode, passes, num_threads, apply_experimental) for v in self.values]

        return MultiIndex(evaluated_data, self.names)

    def _gather_names(self):
        names = [None] * len(self.values) if self.names is None else self.names
        return ['level_' + str(i) if name is None else name for i, name in enumerate(names)]

    def _gather_data(self):
        return self._data

    def _gather_data_for_weld(self):
        return [index.weld_expr for index in self._data]

    def _gather_weld_types(self):
        return [index.weld_type for index in self._data]

    def _iloc_indices(self, indices):
        return MultiIndex([index._iloc_indices(indices) for index in self.values], self.names)

    def _iloc_indices_with_missing(self, indices):
        return MultiIndex([index._iloc_indices_with_missing(indices) for index in self.values], self.names)

    def __getitem__(self, item):
        """Select from the MultiIndex.

        Supported functionality exemplified below.

        Examples
        --------
        >>> mi = bl.MultiIndex([np.array([1, 2, 3]), np.array([4., 5., 6.])], names=['i1', 'i2'])
        >>> print(mi.values[0])
        [1 2 3]
        >>> print(mi[:2].evaluate())
          i1    i2
        ----  ----
           1     4
           2     5
        >>> print(mi[mi.values[0] != 2].evaluate())
          i1    i2
        ----  ----
           1     4
           3     6

        """
        if isinstance(item, LazyArrayResult):
            check_weld_bit_array(item)

            return MultiIndex([column[item] for column in self.values], self.names)
        elif isinstance(item, slice):
            check_valid_int_slice(item)

            return MultiIndex([column[item] for column in self.values], self.names)
        else:
            raise TypeError('Expected LazyArrayResult or slice')

    # this method shouldn't exist however is kept to avoid checking for MultiIndex in DataFrame.tail() ~ generalizing
    def tail(self, n=5):
        """Return MultiIndex with the last n values in each column.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        MultiIndex
            MultiIndex containing the last n values in each column.

        """
        # not computing slice here to use with __getitem__ because we'd need to use len which is eager
        return MultiIndex([v.tail(n) for v in self.values], self.names)
