from collections import OrderedDict

import numpy as np
from tabulate import tabulate
from weld.types import WeldBit

from .generic import BinaryOps
from .indexes import RangeIndex, Index
from .series import Series
from .utils import check_type, is_scalar, valid_int_slice
from ..weld import weld_count, WeldLong, LazyArrayResult, LazyScalarResult


class DataFrame(BinaryOps):
    """ Weld-ed pandas DataFrame.

    Attributes
    ----------
    data : dict
        Data as a dict of column names -> numpy.ndarray or Series.
    index : Index or RangeIndex
        Index of the data.

    See Also
    --------
    pandas.DataFrame : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> from collections import OrderedDict
    >>> df = bl.DataFrame(OrderedDict([('a', np.arange(5, 8)), ('b', np.arange(3))]))
    >>> df.index  # repr
    RangeIndex(start=0, stop=3, step=1)
    >>> df  # repr
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=['a', 'b'])
    >>> print(df.evaluate())  # omitting evaluate would trigger exception as index is now an unevaluated RangeIndex
      Index    a    b
    -------  ---  ---
          0    5    0
          1    6    1
          2    7    2
    >>> print(len(df))
    3
    >>> print((df * 2).evaluate())  # note that atm there is no type casting, i.e. if b was float32, it would fail
      Index    a    b
    -------  ---  ---
          0   10    0
          1   12    2
          2   14    4
    >>> sr = bl.Series(np.array([2] * 3))
    >>> print((df * sr).evaluate())
      Index    a    b
    -------  ---  ---
          0   10    0
          1   12    2
          2   14    4

    """
    @staticmethod
    def _infer_length(data):
        for value in data.values():
            if isinstance(value, np.ndarray):
                return len(value)
            # must be a Series then
            elif isinstance(value.values, np.ndarray):
                return len(value.values)

        return None

    @staticmethod
    def _check_data_types(data):
        for value in data.values():
            check_type(value, (np.ndarray, Series))

        return data

    @staticmethod
    def _gather_dtypes(data):
        return {k: v.dtype for k, v in data.items()}

    @staticmethod
    def _default_dataframe_index(data, length):
        from .indexes import RangeIndex

        if length is not None:
            return RangeIndex(length)
        else:
            # empty
            if len(data) == 0:
                return None
            else:
                # must encode from a random column then
                keys = list(data.keys())

                return RangeIndex(weld_count(data[keys[0]]))

    def __init__(self, data, index=None):
        """Initialize a DataFrame object.

        Parameters
        ----------
        data : dict
            Data as a dict of str -> np.ndarray or Series.
        index : Index or RangeIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.

        """
        self.data = DataFrame._check_data_types(data)
        self._dtypes = DataFrame._gather_dtypes(data)
        self._length = DataFrame._infer_length(data)
        self.index = DataFrame._default_dataframe_index(data, self._length) if index is None else index

    @property
    def values(self):
        """Alias for `data` attribute.

        Returns
        -------
        dict
            The internal dict data representation.

        """
        return self.data

    def __len__(self):
        """Eagerly get the length of the DataFrame.

        Note that if the length is unknown (such as for WeldObjects),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the DataFrame.

        """
        if self._length is not None:
            return self._length
        else:
            # first check again for raw data
            length = DataFrame._infer_length(self.data)
            if length is None:
                keys = list(self.data.keys())

                # empty DataFrame
                if len(keys) == 0:
                    return 0

                # pick a 'random' column (which is a Series) and compute its length
                length = len(self.data[keys[0]])

            self._length = length

            return length

    def __repr__(self):
        return "{}(index={}, columns={})".format(self.__class__.__name__,
                                                 repr(self.index),
                                                 repr(list(self.data.keys())))

    @staticmethod
    def _shorten_data(data):
        if not isinstance(data, np.ndarray):
            raise TypeError('Cannot print unevaluated data. First call evaluate()')

        if len(data) > 50:
            return list(np.concatenate([data[:20], np.array(['...']), data[-20:]]))
        else:
            return data

    def __str__(self):
        # TODO: find a better way to handle empty dataframe; this assumes it's impossible to have data with index=None
        if self.index is None:
            return 'Empty DataFrame'

        str_data = OrderedDict()

        str_data[self.index.name] = DataFrame._shorten_data(self.index.values)

        for column_name in self:
            str_data[column_name] = DataFrame._shorten_data(self[column_name].values)

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if is_scalar(other):
            new_data = OrderedDict(((column_name, self[column_name]._comparison(other, comparison))
                                   for column_name in self))

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can currently only compare with scalars')

    def _element_wise_operation(self, other, operation):
        if isinstance(other, LazyArrayResult):
            new_data = OrderedDict(((column_name, Series._series_array_op(self[column_name], other, operation))
                                    for column_name in self))

            return DataFrame(new_data, self.index)
        elif is_scalar(other):
            new_data = OrderedDict(((column_name, Series._series_element_wise_op(self[column_name], other, operation))
                                    for column_name in self))

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can only apply operation with scalar or Series')

    # TODO: handle empty
    def __getitem__(self, item):
        """Select from the DataFrame.

        Supported functionality exemplified below.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict({'a': np.arange(5, 8)}))
        >>> print(df['a'])
        [5 6 7]
        >>> print(df[df['a'] < 7].evaluate())
          Index    a
        -------  ---
              0    5
              1    6

        """
        if isinstance(item, str):
            value = self.data[item]

            if isinstance(value, np.ndarray):
                value = Series(value, self.index, value.dtype, item)
                # store the newly created Series to avoid remaking it
                self.data[item] = value

            return value
        elif isinstance(item, LazyArrayResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            new_index = self.index[item]
            new_data = OrderedDict(((column_name, Series._filter_series(self[column_name], item, new_index))
                                    for column_name in self))

            return DataFrame(new_data, new_index)
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            new_index = self.index[item]
            new_data = OrderedDict(((column_name, Series._slice_series(self[column_name], item, new_index))
                                    for column_name in self))

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('Expected a column name as a string')

    def __setitem__(self, key, value):
        """Add/update DataFrame column.

        Parameters
        ----------
        key : str
            Column name.
        value : numpy.ndarray or Series
            Note that it does NOT check for the same length as the other columns due to possibly not knowing
            the length before evaluation. Also note that for Series, it currently does NOT match Index as in a join but
            the Series inherits the index of the DataFrame.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict({'a': np.arange(5, 8)}))
        >>> df['b'] = np.arange(3)
        >>> print(df.evaluate())
          Index    a    b
        -------  ---  ---
              0    5    0
              1    6    1
              2    7    2

        """
        key = check_type(key, str)
        value = check_type(value, (np.ndarray, Series))

        # inherit the index, no join atm
        if isinstance(value, Series):
            value.index = self.index

        self.data[key] = value

    def __iter__(self):
        for column_name in self.data:
            yield column_name

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating a DataFrame containing evaluated data and index.

        See `LazyResult`

        Returns
        -------
        DataFrame
            DataFrame with evaluated data and index.

        """
        evaluated_index = self.index.evaluate(verbose, decode, passes, num_threads, apply_experimental)

        evaluated_data = OrderedDict()
        for column_name in self:
            evaluated_data[column_name] = self[column_name].evaluate(verbose, decode, passes,
                                                                     num_threads, apply_experimental)

        return DataFrame(evaluated_data, evaluated_index)

    def head(self, n=5):
        """Return DataFrame with first n values per column.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        DataFrame
            DataFrame containing the first n values per column.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict([('a', np.arange(5, 8)), ('b', np.arange(3))]))
        >>> print(df.head(2).evaluate())
          Index    a    b
        -------  ---  ---
              0    5    0
              1    6    1

        """
        return self[:n]

    def tail(self, n=5):
        """Return DataFrame with last n values per column.

        Parameters
        ----------
        n : int
            Number of values.

        Returns
        -------
        DataFrame
            DataFrame containing the last n values per column.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict([('a', np.arange(5, 8)), ('b', np.arange(3))]))
        >>> print(df.tail(2).evaluate())
          Index    a    b
        -------  ---  ---
              1    6    1
              2    7    2

        """
        if self._length is not None:
            length = self._length
        else:
            # first check again for raw data
            length = DataFrame._infer_length(self.data)
            # if still None, get a random column and encode length
            if length is None:
                keys = list(self.data.keys())
                length = LazyScalarResult(weld_count(self[keys[0]]), WeldLong())

        new_index = self.index.tail(n)
        new_data = OrderedDict(((column_name, Series._tail_series(self[column_name], new_index, length, n))
                                for column_name in self))

        return DataFrame(new_data, new_index)

    def keys(self):
        """Retrieve column names as Index, i.e. for axis=1.

        Returns
        -------
        Index
             Column names as an Index.

        """
        data = np.array(list(self.data.keys()), dtype=np.bytes_)

        return Index(data, np.dtype(np.bytes_))
