from collections import OrderedDict

import numpy as np
from tabulate import tabulate

from .generic import BinaryOps
from .indexes import RangeIndex, Index
from .series import Series
from .utils import check_type, is_scalar, valid_int_slice
from ..weld import WeldLong, LazyArrayResult, LazyScalarResult, weld_to_numpy_dtype, weld_combine_scalars, weld_count, \
    WeldBit, weld_cast_double, WeldDouble


# TODO: handle empty dataframe case throughout operations
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
    >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', np.arange(3)))))
    >>> df.index  # repr
    RangeIndex(start=0, stop=3, step=1)
    >>> df  # repr
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=['a', 'b'])
    >>> print(df.evaluate())  # omitting evaluate would trigger exception as index is now an unevaluated RangeIndex
           a    b
    ---  ---  ---
      0    5    0
      1    6    1
      2    7    2
    >>> print(len(df))
    3
    >>> print((df * 2).evaluate())  # note that atm there is no type casting, i.e. if b was float32, it would fail
           a    b
    ---  ---  ---
      0   10    0
      1   12    2
      2   14    4
    >>> sr = bl.Series(np.array([2] * 3))
    >>> print((df * sr).evaluate())
           a    b
    ---  ---  ---
      0   10    0
      1   12    2
      2   14    4
    >>> print(df.min().evaluate())
    [5 0]
    >>> print(df.mean().evaluate())
    [6. 1.]
    >>> print(df.agg(['var', 'count']).evaluate())
             a    b
    -----  ---  ---
    var      1    1
    count    3    3
    >>> df.rename({'a': 'c'})
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=['c', 'b'])
    >>> df.drop('a')
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=['b'])

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

    def _gather_dtypes(self):
        return OrderedDict(((k, v.dtype) for k, v in self.data.items()))

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

    @property
    def dtypes(self):
        return Series(np.array(list(self._gather_dtypes().values()), dtype=np.bytes_),
                      self.keys())

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

        if self.index.name is None:
            index_name = ' '
        else:
            index_name = self.index.name

        str_data[index_name] = DataFrame._shorten_data(self.index.values)

        for column_name in self:
            str_data[column_name] = DataFrame._shorten_data(self[column_name].values)

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if is_scalar(other):
            new_data = OrderedDict((column_name, self[column_name]._comparison(other, comparison))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can currently only compare with scalars')

    def _element_wise_operation(self, other, operation):
        if isinstance(other, LazyArrayResult):
            new_data = OrderedDict((column_name, Series._series_array_op(self[column_name], other, operation))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        elif is_scalar(other):
            new_data = OrderedDict((column_name, Series._series_element_wise_op(self[column_name], other, operation))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can only apply operation with scalar or Series')

    def __getitem__(self, item):
        """Select from the DataFrame.

        Supported functionality exemplified below.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict({'a': np.arange(5, 8)}))
        >>> print(df['a'])
        [5 6 7]
        >>> print(df[['a']].evaluate())
               a
        ---  ---
          0    5
          1    6
          2    7
        >>> print(df[df['a'] < 7].evaluate())
               a
        ---  ---
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
        elif isinstance(item, list):
            new_data = OrderedDict()

            for column_name in item:
                if not isinstance(column_name, str):
                    raise TypeError('Expected a column name as a string: {}'.format(column_name))
                elif column_name not in self:
                    raise KeyError('Column name not in DataFrame: {}'.format(str(column_name)))

                new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(item, LazyArrayResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            new_index = self.index[item]
            new_data = OrderedDict((column_name, Series._filter_series(self[column_name], item, new_index))
                                   for column_name in self)

            return DataFrame(new_data, new_index)
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            new_index = self.index[item]
            new_data = OrderedDict((column_name, Series._slice_series(self[column_name], item, new_index))
                                   for column_name in self)

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
               a    b
        ---  ---  ---
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

    def __contains__(self, item):
        return item in self.data

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
        >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', np.arange(3)))))
        >>> print(df.head(2).evaluate())
               a    b
        ---  ---  ---
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
        >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', np.arange(3)))))
        >>> print(df.tail(2).evaluate())
               a    b
        ---  ---  ---
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
        new_data = OrderedDict((column_name, Series._tail_series(self[column_name], new_index, length, n))
                               for column_name in self)

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

    def rename(self, columns):
        """Returns a new DataFrame with renamed columns.

        Currently a simplified version of Pandas' rename.

        Parameters
        ----------
        columns : dict
            Old names to new names.

        Returns
        -------
        DataFrame
            With columns renamed, if found.

        """
        new_data = OrderedDict()
        for column_name in self:
            if column_name in columns.keys():
                new_data[columns[column_name]] = self.data[column_name]
            else:
                new_data[column_name] = self.data[column_name]

        return DataFrame(new_data, self.index)

    def drop(self, columns):
        """Drop 1 or more columns. Any column which does not exist in the DataFrame is skipped, i.e. not removed,
        without raising an exception.

        Unlike Pandas' drop, this is currently restricted to dropping columns.

        Parameters
        ----------
        columns : str or list of str
            Column name or list of column names to drop.

        Returns
        -------
        DataFrame
            A new DataFrame without these columns.

        """
        if isinstance(columns, str):
            new_data = OrderedDict()
            for column_name in self:
                if column_name != columns:
                    new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(columns, list):
            new_data = OrderedDict()
            for column_name in self:
                if column_name not in columns:
                    new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Expected columns as a str or a list of str')

    # TODO: currently if the data has multiple types, the results are casted to f64; perhaps be more flexible about it
    # TODO: cast data to relevant 64-bit format pre-aggregation ~ i16, i32 -> i64, f32 -> f64
    def _aggregate_columns(self, func_name):
        new_index = self.keys()

        agg_lazy_results = [getattr(self[column_name], func_name)() for column_name in self]

        # if there are multiple types, cast to float64
        if len(set(self.dtypes.values)) > 1:
            weld_type = WeldDouble()
            dtype = weld_to_numpy_dtype(weld_type)
            agg_lazy_results = (weld_cast_double(result.weld_expr) for result in agg_lazy_results)
        else:
            weld_type = agg_lazy_results[0].weld_type
            dtype = weld_to_numpy_dtype(weld_type)
            agg_lazy_results = (agg.weld_expr for agg in agg_lazy_results)

        new_data = weld_combine_scalars(agg_lazy_results, weld_type)

        return Series(new_data, new_index, dtype)

    def min(self):
        return self._aggregate_columns('min')

    def max(self):
        return self._aggregate_columns('max')

    def sum(self):
        return self._aggregate_columns('sum')

    def prod(self):
        return self._aggregate_columns('prod')

    def count(self):
        return self._aggregate_columns('count')

    def mean(self):
        return self._aggregate_columns('mean')

    def var(self):
        return self._aggregate_columns('var')

    def std(self):
        return self._aggregate_columns('std')

    def agg(self, aggregations):
        """Multiple aggregations optimized.

        Parameters
        ----------
        aggregations : list of str
            Which aggregations to perform.

        Returns
        -------
        DataFrame
            DataFrame with the aggregations per column.

        """
        if not isinstance(aggregations, list):
            raise TypeError('Expected aggregations as a list')

        new_index = Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_))
        new_data = OrderedDict((column_name, Series._agg_series(self[column_name], aggregations, new_index))
                               for column_name in self)

        return DataFrame(new_data, new_index)
