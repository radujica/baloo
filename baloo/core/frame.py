from collections import OrderedDict

import numpy as np
from tabulate import tabulate

from .generic import BinaryOps, BalooCommon
from .indexes import RangeIndex, Index, MultiIndex
from .series import Series
from .utils import check_type, is_scalar, check_inner_types, infer_length, shorten_data, \
    check_weld_bit_array, check_valid_int_slice, as_list, default_index
from ..weld import LazyArrayResult, weld_to_numpy_dtype, weld_combine_scalars, weld_count, \
    weld_cast_double, WeldDouble, weld_sort, LazyLongResult, weld_merge_join, weld_iloc_indices, \
    weld_merge_outer_join


# TODO: handle empty dataframe case throughout operations
# TODO: wrap all internal data in Series to avoid always checking for raw/weldobject ~ like in multiindex
# TODO: maybe add type hints
class DataFrame(BinaryOps, BalooCommon):
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
    >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', np.array([1, 0, 2])))))
    >>> df.index  # repr
    RangeIndex(start=0, stop=3, step=1)
    >>> df  # repr
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=[a: int64, b: int64])
    >>> print(df.evaluate())  # omitting evaluate would trigger exception as index is now an unevaluated RangeIndex
           a    b
    ---  ---  ---
      0    5    1
      1    6    0
      2    7    2
    >>> print(len(df))
    3
    >>> print((df * 2).evaluate())  # note that atm there is no type casting, i.e. if b was float32, it would fail
           a    b
    ---  ---  ---
      0   10    2
      1   12    0
      2   14    4
    >>> sr = bl.Series(np.array([2] * 3))
    >>> print((df * sr).evaluate())
           a    b
    ---  ---  ---
      0   10    2
      1   12    0
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
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=[c: int64, b: int64])
    >>> df.drop('a')
    DataFrame(index=RangeIndex(start=0, stop=3, step=1), columns=[b: int64])
    >>> print(df.reset_index().evaluate())
           index    a    b
    ---  -------  ---  ---
      0        0    5    1
      1        1    6    0
      2        2    7    2
    >>> print(df.set_index('b').evaluate())
      b    a
    ---  ---
      1    5
      0    6
      2    7
    >>> print(df.sort_values('b').evaluate())
           a    b
    ---  ---  ---
      1    6    0
      0    5    1
      2    7    2
    >>> df2 = bl.DataFrame({'b': np.array([0, 2])})
    >>> print(df.merge(df2, on='b').evaluate())
      b    index_x    a    index_y
    ---  ---------  ---  ---------
      2          2    7          1

    """
    @staticmethod
    def _default_dataframe_index(data, length):
        if length is not None:
            return default_index(length)
        else:
            if len(data) == 0:
                return None
            else:
                # must encode from a random column then
                return default_index(data[list(data.keys())[0]])

    @staticmethod
    def _process_index(index, data, length):
        if index is None:
            return DataFrame._default_dataframe_index(data, length)
        else:
            return check_type(index, (Index, MultiIndex))

    @staticmethod
    def _process_data(data, index):
        for k, v in data.items():
            if isinstance(v, Series):
                v.name = k
            else:
                # must be ndarray
                data[k] = Series(v, index, name=k)

        return data

    def __init__(self, data, index=None):
        """Initialize a DataFrame object.

        Parameters
        ----------
        data : dict
            Data as a dict of str -> np.ndarray or Series.
        index : Index or RangeIndex or MultiIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.

        """
        check_inner_types(check_type(data, dict).values(), (np.ndarray, Series))
        # TODO: length could also be inferred from index, if passed
        self._length = infer_length(data.values())
        self.index = DataFrame._process_index(index, data, self._length)
        self._data = DataFrame._process_data(data, self.index)

    @property
    def values(self):
        """Alias for `data` attribute.

        Returns
        -------
        dict
            The internal dict data representation.

        """
        return self._data

    def _gather_dtypes(self):
        return OrderedDict(((k, v.dtype) for k, v in self._data.items()))

    @property
    def dtypes(self):
        return Series(np.array(list(self._gather_dtypes().values()), dtype=np.bytes_),
                      self.keys())

    def _gather_column_names(self):
        return list(self._data.keys())

    @property
    def columns(self):
        return Index(np.array(self._gather_column_names(), dtype=np.bytes_), np.dtype(np.bytes_))

    def _gather_data_for_weld(self):
        return [column.values for column in self._data.values()]

    def _gather_weld_types(self):
        return [column.weld_type for column in self._data.values()]

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
            length = infer_length(self._data.values())
            if length is None:
                keys = list(self._data.keys())

                # empty DataFrame
                if len(keys) == 0:
                    return 0

                # pick a 'random' column (which is a Series) and compute its length
                length = len(self._data[keys[0]])

            self._length = length

            return length

    @property
    def iloc(self):
        """Retrieve Indexer by index.

        Supported iloc functionality exemplified below.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', np.array([1, 0, 2])))))
        >>> print(df.iloc[0:2].evaluate())
               a    b
        ---  ---  ---
          0    5    1
          1    6    0
        >>> print(df.iloc[bl.Series(np.array([0, 2]))].evaluate())
               a    b
        ---  ---  ---
          0    5    1
          2    7    2

        """
        from .indexing import _ILocIndexer

        return _ILocIndexer(self)

    def __repr__(self):
        columns = '[' + ', '.join(['{}: {}'.format(k, v) for k, v in self._gather_dtypes().items()]) + ']'

        return "{}(index={}, columns={})".format(self.__class__.__name__,
                                                 repr(self.index),
                                                 columns)

    def __str__(self):
        # TODO: find a better way to handle empty dataframe; this assumes it's impossible to have data with index=None
        if self.index is None:
            return 'Empty DataFrame'

        str_data = OrderedDict()
        str_data.update((name, data) for name, data in self.index._gather_data().items())
        str_data.update((column.name, shorten_data(column.values)) for column in self._iter())

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if is_scalar(other):
            new_data = OrderedDict((column_name, self._data[column_name]._comparison(other, comparison))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can currently only compare with scalars')

    def _element_wise_operation(self, other, operation):
        if isinstance(other, LazyArrayResult):
            new_data = OrderedDict((column_name, Series._series_array_op(self._data[column_name], other, operation))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        elif is_scalar(other):
            new_data = OrderedDict((column_name, Series._series_element_wise_op(self._data[column_name], other, operation))
                                   for column_name in self)

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can only apply operation with scalar or LazyArrayResult')

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
            return self._data[item]
        elif isinstance(item, list):
            check_inner_types(item, str)
            new_data = OrderedDict()

            for column_name in item:
                if column_name not in self:
                    raise KeyError('Column name not in DataFrame: {}'.format(str(column_name)))

                new_data[column_name] = self._data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(item, LazyArrayResult):
            check_weld_bit_array(item)

            new_index = self.index[item]
            new_data = OrderedDict((column_name, Series._filter_series(self._data[column_name], item, new_index))
                                   for column_name in self)

            return DataFrame(new_data, new_index)
        elif isinstance(item, slice):
            check_valid_int_slice(item)

            new_index = self.index[item]
            new_data = OrderedDict((column_name, Series._slice_series(self._data[column_name], item, new_index))
                                   for column_name in self)

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('Expected a column name, list of columns, LazyArrayResult, or a slice')

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

        # inherit the index, no join/alignment atm
        # TODO: add the inner join alignment
        if isinstance(value, Series):
            value.index = self.index
            value.name = key
        else:
            value = Series(value, self.index, value.dtype, key)

        self._data[key] = value

    def _iter(self):
        for column in self._data.values():
            yield column

    def __iter__(self):
        for column_name in self._data:
            yield column_name

    def __contains__(self, item):
        return item in self._data

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
            evaluated_data[column_name] = self._data[column_name].evaluate(verbose, decode, passes,
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
            length = infer_length(self._data.values())
            # if still None, get a random column and encode length
            if length is None:
                keys = list(self._data.keys())
                length = LazyLongResult(weld_count(self._data[keys[0]]))

        new_index = self.index.tail(n)
        new_data = OrderedDict((column_name, Series._tail_series(self._data[column_name], new_index, length, n))
                               for column_name in self)

        return DataFrame(new_data, new_index)

    def keys(self):
        """Retrieve column names as Index, i.e. for axis=1.

        Returns
        -------
        Index
             Column names as an Index.

        """
        return self.columns

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
                new_data[columns[column_name]] = self._data[column_name]
            else:
                new_data[column_name] = self._data[column_name]

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
                    new_data[column_name] = self._data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(columns, list):
            new_data = OrderedDict()
            for column_name in self:
                if column_name not in columns:
                    new_data[column_name] = self._data[column_name]

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Expected columns as a str or a list of str')

    # TODO: currently if the data has multiple types, the results are casted to f64; perhaps be more flexible about it
    # TODO: cast data to relevant 64-bit format pre-aggregation ~ i16, i32 -> i64, f32 -> f64
    # TODO: update gather_dtypes, str check
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
        check_type(aggregations, list)

        new_index = Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_))
        new_data = OrderedDict((column_name, Series._agg_series(self._data[column_name], aggregations, new_index))
                               for column_name in self)

        return DataFrame(new_data, new_index)

    def reset_index(self):
        """Returns a new DataFrame with previous index as column(s).

        Returns
        -------
        DataFrame
            DataFrame with the new index a RangeIndex of its length.

        """
        new_columns = OrderedDict()

        # assumes at least 1 column
        length = self._length
        if length is None:
            length = infer_length(self._data.values())
        if length is None:
            a_column = self._data[list(self._data.keys())[-1]]
            length = weld_count(a_column.values)
        new_index = RangeIndex(0, length, 1)

        for name, data in self.index._gather_data().items():
            new_columns[name] = Series(data.values, new_index, data.dtype, name)

        # the data/columns
        new_columns.update(self._data)

        return DataFrame(new_columns, new_index)

    def set_index(self, keys):
        """Set the index of the DataFrame to be the keys columns.

        Note this means that the old index is removed.

        Parameters
        ----------
        keys : str or list of str
            Which column(s) to set as the index.

        Returns
        -------
        DataFrame
            DataFrame with the index set to the column(s) corresponding to the keys.

        """
        if isinstance(keys, str):
            column = self._data[keys]
            new_index = Index(column.values, column.dtype, column.name)

            new_data = OrderedDict(self.values)
            del new_data[keys]

            return DataFrame(new_data, new_index)
        elif isinstance(keys, list):
            check_inner_types(keys, str)

            new_index_data = []
            for column_name in keys:
                column = self._data[column_name]
                new_index_data.append(Index(column.values, column.dtype, column.name))
            new_index = MultiIndex(new_index_data, keys)

            new_data = OrderedDict(self.values)
            for column_name in keys:
                del new_data[column_name]

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('Expected a string or a list of strings')

    def sort_index(self, ascending=True):
        """Sort the index of the DataFrame.

        Currently MultiIndex is not supported since Weld is missing multiple-column sort.

        Note this is an expensive operation (brings all data to Weld).

        Parameters
        ----------
        ascending : bool, optional

        Returns
        -------
        DataFrame
            DataFrame sorted according to the index.

        """
        if isinstance(self.index, MultiIndex):
            raise NotImplementedError('Weld does not yet support sorting on multiple columns')

        return self.sort_values(self.index._gather_names(), ascending)

    def sort_values(self, by, ascending=True):
        """Sort the DataFrame based on a column.

        Unlike Pandas, one can sort by data from both index and regular columns.

        Currently possible to sort only on a single column since Weld is missing multiple-column sort.
        Note this is an expensive operation (brings all data to Weld).

        Parameters
        ----------
        by : str or list of str
            Column names to sort.
        ascending : bool, optional

        Returns
        -------
        DataFrame
            DataFrame sorted according to the column.

        """
        check_type(ascending, bool)
        check_inner_types(by, str) if isinstance(by, list) else check_type(by, str)
        by = as_list(by)

        if len(by) > 1:
            raise NotImplementedError('Weld does not yet support sorting on multiple columns')

        all_data = self.reset_index()
        by_data = all_data[by]

        sorted_indices = weld_sort(by_data._gather_data_for_weld(),
                                   by_data._gather_weld_types(),
                                   'sort_index',
                                   ascending=ascending)

        new_index = self.index._iloc_indices(sorted_indices)
        new_columns = [column for column in self._data.values()]
        new_column_names = [column.name for column in new_columns]
        new_columns = [column.iloc._iloc_series(sorted_indices, new_index) for column in new_columns]
        new_data = OrderedDict(zip(new_column_names, new_columns))

        return DataFrame(new_data, new_index)

    @staticmethod
    def _compute_on(self, other, on, all_names_self, all_names_other):
        if on is None:
            self_index_names = self.index._gather_names()
            other_index_names = other.index._gather_names()

            if len(self_index_names) != len(other_index_names):
                raise ValueError('Expected indexes to be of the same dimensions when on=None')
            elif self_index_names != other_index_names:
                raise ValueError('When on=None, the names of both indexes must be the same')
            else:
                return self_index_names
        else:
            on = as_list(on)
            set_on = set(on)

            if not set_on.issubset(set(all_names_self)):
                raise ValueError('On column(s) not included in the self DataFrame')
            elif not set_on.issubset(set(all_names_other)):
                raise ValueError('On column(s) not included in the other DataFrame')
            else:
                return on

    @staticmethod
    def _compute_new_names(names_self, names_other, suffixes):
        common_names = set(names_self).intersection(set(names_other))
        self_new_names = names_self
        other_new_names = names_other

        if len(common_names) != 0:
            for name in common_names:
                self_new_names[self_new_names.index(name)] += suffixes[0]
                other_new_names[other_new_names.index(name)] += suffixes[1]

        return self_new_names, other_new_names

    # TODO: perhaps just split into 4 methods for each join type
    @staticmethod
    def _compute_new_index(weld_objects_indexes, how, on, self_on_cols, other_on_cols, filter_func):
        if how in ['inner', 'left']:
            extract_index_from = self_on_cols
            index_index = 0
        else:
            extract_index_from = other_on_cols
            index_index = 1

        if how == 'outer':
            data_arg = 'weld_objects_indexes[2]'
        else:
            data_arg = 'column.values'

        new_indexes = []
        data_arg = data_arg if how != 'outer' else data_arg + '[i]'
        for i, column_name in enumerate(on):
            column = extract_index_from[column_name]
            new_indexes.append(Index(filter_func(eval(data_arg),
                                                 column.weld_type,
                                                 weld_objects_indexes[index_index]),
                                     column.dtype,
                                     column.name))
        if len(on) > 1:
            new_index = MultiIndex(new_indexes, on)
        else:
            new_index = new_indexes[0]

        return new_index

    def merge(self, other, how='inner', on=None, suffixes=('_x', '_y'),
              algorithm='merge', is_on_sorted=True, is_on_unique=True):
        """Database-like join this DataFrame with the other DataFrame.

        Currently assumes the `on` columns are sorted and the on-column(s) values are unique!
        Next work handles the other cases.

        Note there's no automatic cast if the type of the on columns differs.

        Algorithms and limitations:

        - Merge algorithms: merge-join or hash-join. Typical pros and cons apply when choosing between the two.
          Merge-join shall be used on fairly equally-sized DataFrames while a hash-join would be better when
          one of the DataFrames is (much) smaller.
        - Limitations:

          + Hash-join requires the (smaller) hashed DataFrame
            (more precisely, the on columns) to contain no duplicates!
          + Merge-join requires the on-columns to be sorted!
          + For unsorted data can only sort a single column! (current Weld limitation)

        - Sortedness. If the on-columns are sorted, merge-join does not require to sort the data so it can be
          significantly faster. Do add is_on_sorted=True if this is known to be true!
        - Uniqueness. If the on-columns data contains duplicates, the algorithm is more complicated, i.e. slow.
          Also hash-join cannot be used on a hashed (smaller) DataFrame with duplicates. Do add is_on_unique=True
          if this is known to be true!
        - Setting the above 2 flags incorrectly, e.g. is_on_sorted to True when data is in fact not sorted,
          will produce undefined results.

        Parameters
        ----------
        other : DataFrame
            With which to merge.
        how : {'inner', 'left', 'right', 'outer'}, optional
            Which kind of join to do.
        on : str or list or None, optional
            The columns from both DataFrames on which to join.
            If None, will join on the index if it has the same name.
        suffixes : tuple of str, optional
            To append on columns not in `on` that have the same name in the DataFrames.
        algorithm : {'merge', 'hash'}, optional
            Which algorithm to use. Note that for 'hash', the `other` DataFrame is the one hashed.
        is_on_sorted : bool, optional
            If we know that the on columns are already sorted, can employ faster algorithm.
        is_on_unique : bool, optional
            If we know that the values are unique, can employ faster algorithm.

        Returns
        -------
        DataFrame
            DataFrame containing the merge result, with the `on` columns as index.

        """
        check_type(other, DataFrame)
        check_type(how, str)
        check_type(algorithm, str)
        check_inner_types(on, str) if isinstance(on, list) else check_type(on, str)
        check_inner_types(check_type(suffixes, tuple), str)
        check_type(is_on_sorted, bool)
        check_type(is_on_unique, bool)

        # TODO: change defaults on flags & remove after implementation
        assert is_on_sorted
        assert is_on_unique

        self_reset = self.reset_index()
        other_reset = other.reset_index()
        on = DataFrame._compute_on(self, other, on,
                                   self_reset._gather_column_names(),
                                   other_reset._gather_column_names())
        self_on_cols = self_reset[on]
        other_on_cols = other_reset[on]

        if algorithm == 'merge':
            # for left and right joins, the on columns can just be copied; no need for filter
            def fake_filter_func(x, y, z):
                return x

            if how == 'inner':
                index_filter_func = weld_iloc_indices
                data_filter_func = '_iloc_series'
                weld_merge_func = weld_merge_join
            elif how in {'left', 'right'}:
                index_filter_func = fake_filter_func
                data_filter_func = '_iloc_series_with_missing'
                weld_merge_func = weld_merge_join
            else:
                index_filter_func = fake_filter_func
                data_filter_func = '_iloc_series_with_missing'
                weld_merge_func = weld_merge_outer_join

            weld_objects_indexes = weld_merge_func(self_on_cols._gather_data_for_weld(),
                                                   self_on_cols._gather_weld_types(),
                                                   other_on_cols._gather_data_for_weld(),
                                                   other_on_cols._gather_weld_types(),
                                                   how, is_on_sorted, is_on_unique, 'merge-join')

            new_index = DataFrame._compute_new_index(weld_objects_indexes,
                                                     how,
                                                     on,
                                                     self_on_cols,
                                                     other_on_cols,
                                                     index_filter_func)

            new_data = OrderedDict()
            self_no_on = self_reset.drop(on)
            other_no_on = other_reset.drop(on)
            self_new_names, other_new_names = DataFrame._compute_new_names(self_no_on._gather_column_names(),
                                                                           other_no_on._gather_column_names(),
                                                                           suffixes)

            for column_name, new_name in zip(self_no_on, self_new_names):
                new_data[new_name] = getattr(self_no_on[column_name].iloc,
                                             data_filter_func)(weld_objects_indexes[0], new_index)

            for column_name, new_name in zip(other_no_on, other_new_names):
                new_data[new_name] = getattr(other_no_on[column_name].iloc,
                                             data_filter_func)(weld_objects_indexes[1], new_index)

            return DataFrame(new_data, new_index)
        elif algorithm == 'hash':
            raise NotImplementedError('Not yet supported')
        else:
            raise NotImplementedError('Only merge- and hash-join algorithms are supported')

    def join(self, other, on=None, how='left', lsuffix=None, rsuffix=None,
             algorithm='merge', is_on_sorted=True, is_on_unique=True):
        """Database-like join this DataFrame with the other DataFrame.

        Currently assumes the `on` columns are sorted and the on-column(s) values are unique!
        Next work handles the other cases.

        Note there's no automatic cast if the type of the on columns differs.

        Check DataFrame.merge() for more details.

        Parameters
        ----------
        other : DataFrame
            With which to merge.
        on : str or list or None, optional
            The columns from both DataFrames on which to join.
            If None, will join on the index if it has the same name.
        how : {'inner', 'left', 'right', 'outer'}, optional
            Which kind of join to do.
        lsuffix : str, optional
            Suffix to use on columns that overlap from self.
        rsuffix : str, optional
            Suffix to use on columns that overlap from other.
        algorithm : {'merge', 'hash'}, optional
            Which algorithm to use. Note that for 'hash', the `other` DataFrame is the one hashed.
        is_on_sorted : bool, optional
            If we know that the on columns are already sorted, can employ faster algorithm.
        is_on_unique : bool, optional
            If we know that the values are unique, can employ faster algorithm.

        Returns
        -------
        DataFrame
            DataFrame containing the merge result, with the `on` columns as index.

        """
        check_type(lsuffix, str)
        check_type(rsuffix, str)

        self_names = self._gather_column_names()
        other_names = other._gather_column_names()
        common_names = set(self_names).intersection(set(other_names))
        if len(common_names) > 0 and lsuffix is None and rsuffix is None:
            raise ValueError('Columns overlap but no suffixes supplied')

        # need to ensure that some str suffixes are passed to merge
        lsuffix = '' if lsuffix is None else lsuffix
        rsuffix = '' if rsuffix is None else rsuffix

        # TODO: pandas is more flexible, e.g. allows the index names to be different when joining on index
        # TODO i.e. df(ind + a, b) join df(ind2 + b, c) does work and the index is now called ind

        return self.merge(other, how, on, (lsuffix, rsuffix), algorithm, is_on_sorted, is_on_unique)
