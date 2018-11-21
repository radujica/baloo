from collections import OrderedDict
from functools import reduce

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
    names
    dtypes

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> ind = bl.MultiIndex([[1, 2, 3], np.array([4, 5, 6], dtype=np.float64)], names=['i1', 'i2'])
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
    def __init__(self, data, names=None):
        """Initialize a MultiIndex object.

        Parameters
        ----------
        data : list of (numpy.ndarray or Index or list)
            The internal data.
        names : list of str, optional
            The names of the data.

        """
        check_inner_types(check_type(data, list), (np.ndarray, Index, list))
        self._length = infer_length(data)
        self.name = None
        self.names = _init_names(len(data), names)
        self._data = _init_indexes(data, self.names)

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
    def empty(self):
        return len(self._data) == 0 or all(index.empty for index in self._data)

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

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def _gather_names(self, name='level_'):
        names = [None] * len(self.values) if self.names is None else self.names
        return [name + str(i) if n is None else n for i, n in enumerate(names)]

    def _gather_data_for_weld(self):
        return [index.weld_expr for index in self._data]

    def _gather_data(self, name='level_'):
        return OrderedDict(zip(self._gather_names(name), self._data))

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

    def dropna(self):
        """Returns MultiIndex without any rows containing null values according to Baloo's convention.

        Returns
        -------
        MultiIndex
            MultiIndex with no null values.

        """
        not_nas = [v.notna() for v in self.values]
        and_filter = reduce(lambda x, y: x & y, not_nas)

        return self[and_filter]

    @classmethod
    def from_pandas(cls, index):
        """Create baloo MultiIndex from pandas MultiIndex.

        Parameters
        ----------
        index : pandas.multi.MultiIndex

        Returns
        -------
        MultiIndex

        """
        from pandas import MultiIndex as PandasMultiIndex
        check_type(index, PandasMultiIndex)

        baloo_level_values = [Index.from_pandas(index.get_level_values(level))
                              for level in range(len(index.levels))]

        return MultiIndex(baloo_level_values, list(index.names))

    def to_pandas(self):
        """Convert to pandas MultiIndex.

        Returns
        -------
        pandas.base.MultiIndex

        """
        if not all(ind.is_raw() for ind in self.values):
            raise ValueError('Cannot convert to pandas MultiIndex if not evaluated.')

        from pandas import MultiIndex as PandasMultiIndex

        arrays = [ind.values for ind in self.values]

        return PandasMultiIndex.from_arrays(arrays, names=self.names)


def _init_names(number_columns, names):
    check_inner_types(check_type(names, list), str)

    if names is None:
        names = [None] * number_columns
    elif number_columns != len(names):
        raise ValueError('Expected all or none of the columns to be named')

    return names


def _init_indexes(data, names):
    data_as_indexes = []
    for n, v in zip(names, data):
        if isinstance(v, np.ndarray):
            v = Index(v, v.dtype, n)
        elif isinstance(v, list):
            v = Index(v, name=n)
        data_as_indexes.append(v)

    return data_as_indexes
