from collections import OrderedDict
from functools import reduce

import numpy as np
from tabulate import tabulate

from .generic import BinaryOps, BalooCommon
from .indexes import Index, MultiIndex
from .series import Series, _series_slice, _series_filter, _series_element_wise_op, _series_agg, _series_tail, \
    _series_iloc, _series_iloc_with_missing, _series_from_pandas
from .utils import check_type, is_scalar, check_inner_types, infer_length, shorten_data, \
    check_weld_bit_array, check_valid_int_slice, as_list, default_index, same_index, check_str_or_list_str
from ..weld import LazyArrayResult, weld_to_numpy_dtype, weld_combine_scalars, weld_count, \
    weld_cast_double, WeldDouble, weld_sort, LazyLongResult, weld_merge_join, weld_iloc_indices, \
    weld_merge_outer_join, weld_align, weld_drop_duplicates


class DataFrame(BinaryOps, BalooCommon):
    """ Weld-ed pandas DataFrame.

    Attributes
    ----------
    index
    dtypes
    columns
    iloc

    See Also
    --------
    pandas.DataFrame : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

    Examples
    --------
    >>> import baloo as bl
    >>> import numpy as np
    >>> from collections import OrderedDict
    >>> df = bl.DataFrame(OrderedDict((('a', np.arange(5, 8)), ('b', [1, 0, 2]))))
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
    >>> print((df * 2).evaluate())
           a    b
    ---  ---  ---
      0   10    2
      1   12    0
      2   14    4
    >>> print((df * [2, 3]).evaluate())
           a    b
    ---  ---  ---
      0   10    3
      1   12    0
      2   14    6
    >>> print(df.min().evaluate())
    <BLANKLINE>
    ---  --
    a     5
    b     0
    >>> print(df.mean().evaluate())
    <BLANKLINE>
    ---  --
    a     6
    b     1
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
      0          1    6          0
      2          2    7          1
    >>> df3 = bl.DataFrame({'a': [1., -999., 3.]}, bl.Index([-999, 1, 2]))
    >>> print(df3.dropna().evaluate())
            a
    ----  ---
    -999    1
       2    3
    >>> print(df3.fillna({'a': 15}).evaluate())
            a
    ----  ---
    -999    1
       1   15
       2    3
    >>> print(bl.DataFrame({'a': [0, 1, 1, 2], 'b': [1, 2, 3, 4]}).groupby('a').sum().evaluate())
      a    b
    ---  ---
      0    1
      2    4
      1    5

    """
    _empty_text = 'Empty DataFrame'

    def __init__(self, data=None, index=None):
        """Initialize a DataFrame object.

        Note that (unlike pandas) there's currently no index inference or alignment between the indexes of any Series
        passed as data. That is, all data, be it raw or Series, inherits the index of the DataFrame. Alignment
        is currently restricted to setitem

        Parameters
        ----------
        data : dict, optional
            Data as a dict of str -> np.ndarray or Series or list.
        index : Index or RangeIndex or MultiIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.

        """
        data = _check_input_data(data)
        self._length = _infer_length(index, data)
        self.index = _process_index(index, data, self._length)
        self._data = _process_data(data, self.index)

    @property
    def values(self):
        """Alias for `data` attribute.

        Returns
        -------
        dict
            The internal dict data representation.

        """
        return self._data

    @property
    def empty(self):
        return self.index.empty and (len(self._data) == 0 or all(series.empty for series in self._iter()))

    def _gather_dtypes(self):
        return OrderedDict(((k, v.dtype) for k, v in self._data.items()))

    @property
    def dtypes(self):
        """Series of NumPy dtypes present in the DataFrame with index of column names.

        Returns
        -------
        Series

        """
        return Series(np.array(list(self._gather_dtypes().values()), dtype=np.bytes_),
                      self.keys())

    def _gather_column_names(self):
        return list(self._data.keys())

    @property
    def columns(self):
        """Index of the column names present in the DataFrame in order.

        Returns
        -------
        Index

        """
        return Index(np.array(self._gather_column_names(), dtype=np.bytes_), np.dtype(np.bytes_))

    def _gather_data_for_weld(self):
        return [column.weld_expr for column in self._data.values()]

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
        self._length = LazyLongResult(_obtain_length(self._length, self._data)).evaluate()

        return self._length

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

    # TODO: extend tabulate to e.g. insert a line between index and values
    def __str__(self):
        if self.empty:
            return self._empty_text

        default_index_name = ' '
        str_data = OrderedDict()
        str_data.update((name, shorten_data(data.values))
                        for name, data in self.index._gather_data(default_index_name).items())
        str_data.update((column.name, shorten_data(column.values)) for column in self._iter())

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if other is None or is_scalar(other):
            df = _drop_str_columns(self)
            new_data = OrderedDict((column.name, column._comparison(other, comparison))
                                   for column in df._iter())

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can currently only compare with scalars')

    def _element_wise_operation(self, other, operation):
        if isinstance(other, list):
            check_inner_types(other, (int, float))

            df = _drop_str_columns(self)
            if len(other) != len(df._gather_column_names()):
                raise ValueError('Expected same number of values in other as the number of non-string columns')

            new_data = OrderedDict((column.name, _series_element_wise_op(column, scalar, operation))
                                   for column, scalar in zip(df._iter(), other))

            return DataFrame(new_data, self.index)
        elif is_scalar(other):
            df = _drop_str_columns(self)
            new_data = OrderedDict((column.name, _series_element_wise_op(column, other, operation))
                                   for column in df._iter())

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can only apply operation with scalar or LazyArrayResult')

    def astype(self, dtype):
        """Cast DataFrame columns to given dtype.

        Parameters
        ----------
        dtype : numpy.dtype or dict
            Dtype or column_name -> dtype mapping to cast columns to. Note index is excluded.

        Returns
        -------
        DataFrame
            With casted columns.

        """
        if isinstance(dtype, np.dtype):
            new_data = OrderedDict((column.name, column.astype(dtype))
                                   for column in self._iter())

            return DataFrame(new_data, self.index)
        elif isinstance(dtype, dict):
            check_inner_types(dtype.values(), np.dtype)

            new_data = OrderedDict(self._data)
            for column in self._iter():
                column_name = column.name
                if column_name in dtype:
                    new_data[column_name] = column.astype(dtype[column_name])

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Expected numpy.dtype or dict mapping column names to dtypes')

    def __getitem__(self, item):
        """Select from the DataFrame.

        Supported functionality exemplified below.

        Examples
        --------
        >>> df = bl.DataFrame(OrderedDict({'a': np.arange(5, 8)}))
        >>> print(df['a'].evaluate())
               a
        ---  ---
          0    5
          1    6
          2    7
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
            new_data = OrderedDict((column.name, _series_filter(column, item, new_index))
                                   for column in self._iter())

            return DataFrame(new_data, new_index)
        elif isinstance(item, slice):
            check_valid_int_slice(item)

            new_index = self.index[item]
            new_data = OrderedDict((column.name, _series_slice(column, item, new_index))
                                   for column in self._iter())

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('Expected a column name, list of columns, LazyArrayResult, or a slice')

    def __setitem__(self, key, value):
        """Add/update DataFrame column.

        Note that for raw data, it does NOT check for the same length with the DataFrame due to possibly not knowing
        the length before evaluation. Hence, columns of different lengths are possible if using raw data which might
        lead to unexpected behavior. To avoid this, use the more expensive setitem by wrapping with a Series.
        This, in turn, means that if knowing the indexes match and the data has the same length as the DataFrame,
        it is more efficient to setitem using the raw data.

        Parameters
        ----------
        key : str
            Column name.
        value : numpy.ndarray or Series
            If a Series, the data will be aligned based on the index of the DataFrame,
            i.e. df.index left join sr.index.

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

        if isinstance(value, Series):
            if not same_index(self.index, value.index):
                value = Series(weld_align(self.index._gather_data_for_weld(),
                                          self.index._gather_weld_types(),
                                          value.index._gather_data_for_weld(),
                                          value.index._gather_weld_types(),
                                          value.values,
                                          value.weld_type),
                               self.index,
                               value.dtype,
                               key)
                # else keep as is
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
        evaluated_data = OrderedDict((column.name, column.evaluate(verbose, decode, passes,
                                                                   num_threads, apply_experimental))
                                     for column in self._iter())

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
        length = _obtain_length(self._length, self._data)

        new_index = self.index.tail(n)
        new_data = OrderedDict((column.name, _series_tail(column, new_index, length, n))
                               for column in self._iter())

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
                column = self._data[column_name]
                new_name = columns[column_name]
                new_data[new_name] = Series(column.values, column.index, column.dtype, new_name)
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
            if columns not in self._gather_column_names():
                raise KeyError('Key {} not found'.format(columns))

            for column_name in self:
                if column_name != columns:
                    new_data[column_name] = self._data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(columns, list):
            check_inner_types(columns, str)

            df = self
            for column in columns:
                df = df.drop(column)

            return df
        else:
            raise TypeError('Expected columns as a str or a list of str')

    # TODO: currently if the data has multiple types, the results are casted to f64; perhaps be more flexible about it
    # TODO: cast data to relevant 64-bit format pre-aggregation? ~ i16, i32 -> i64, f32 -> f64
    def _aggregate_columns(self, func_name):
        df = _drop_str_columns(self)
        if len(df._data) == 0:
            return Series()

        new_index = df.keys()

        agg_lazy_results = [getattr(column, func_name)() for column in df._iter()]

        # if there are multiple types, cast to float64
        if len(set(df._gather_dtypes().values())) > 1:
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

        df = _drop_str_columns(self)
        if len(df._data) == 0:
            # conforming to what pandas does
            raise ValueError('No results')

        new_index = Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_))
        new_data = OrderedDict((column.name, _series_agg(column, aggregations, new_index))
                               for column in df._iter())

        return DataFrame(new_data, new_index)

    def reset_index(self):
        """Returns a new DataFrame with previous index as column(s).

        Returns
        -------
        DataFrame
            DataFrame with the new index a RangeIndex of its length.

        """
        new_columns = OrderedDict()

        new_index = default_index(_obtain_length(self._length, self._data))

        new_columns.update((name, Series(data.values, new_index, data.dtype, name))
                           for name, data in self.index._gather_data().items())

        # the data/columns
        new_columns.update((sr.name, Series(sr.values, new_index, sr.dtype, sr.name))
                           for sr in self._iter())

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

            new_data = OrderedDict((sr.name, Series(sr.values, new_index, sr.dtype, sr.name))
                                   for sr in self._iter())
            del new_data[keys]

            return DataFrame(new_data, new_index)
        elif isinstance(keys, list):
            check_inner_types(keys, str)

            new_index_data = []
            for column_name in keys:
                column = self._data[column_name]
                new_index_data.append(Index(column.values, column.dtype, column.name))
            new_index = MultiIndex(new_index_data, keys)

            new_data = OrderedDict((sr.name, Series(sr.values, new_index, sr.dtype, sr.name))
                                   for sr in self._iter())
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
        check_str_or_list_str(by)
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
        new_columns = list(self._iter())
        new_column_names = [column.name for column in new_columns]
        new_columns = [_series_iloc(column, sorted_indices, new_index) for column in new_columns]
        new_data = OrderedDict(zip(new_column_names, new_columns))

        return DataFrame(new_data, new_index)

    def merge(self, other, how='inner', on=None, suffixes=('_x', '_y'),
              algorithm='merge', is_on_sorted=False, is_on_unique=True):
        """Database-like join this DataFrame with the other DataFrame.

        Currently assumes the on-column(s) values are unique!

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
            If we know that the on columns are already sorted, can employ faster algorithm. If False,
            the DataFrame will first be sorted by the on columns.
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
        check_str_or_list_str(on)
        check_inner_types(check_type(suffixes, tuple), str)
        check_type(is_on_sorted, bool)
        check_type(is_on_unique, bool)

        # TODO: change defaults on flag & remove after implementation
        assert is_on_unique

        # TODO: this materialization/cache step could be skipped by encoding the whole sort + merge;
        # TODO this would use the sorted on columns from weld_sort ($.1) in the join to obtain join-output-indices
        # TODO which would then be passed through a 'translation table' (of $.0) to obtain the original indices to keep
        if not is_on_sorted:
            self_df = self.sort_values(on)
            other_df = other.sort_values(on)
        else:
            self_df = self
            other_df = other

        self_reset = self_df.reset_index()
        other_reset = other_df.reset_index()
        on = _compute_on(self_df, other_df, on,
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
                data_filter_func = _series_iloc
                weld_merge_func = weld_merge_join
            elif how in {'left', 'right'}:
                index_filter_func = fake_filter_func
                data_filter_func = _series_iloc_with_missing
                weld_merge_func = weld_merge_join
            else:
                index_filter_func = fake_filter_func
                data_filter_func = _series_iloc_with_missing
                weld_merge_func = weld_merge_outer_join

            weld_objects_indexes = weld_merge_func(self_on_cols._gather_data_for_weld(),
                                                   self_on_cols._gather_weld_types(),
                                                   other_on_cols._gather_data_for_weld(),
                                                   other_on_cols._gather_weld_types(),
                                                   how, is_on_sorted, is_on_unique, 'merge-join')

            new_index = _compute_new_index(weld_objects_indexes, how, on,
                                           self_on_cols, other_on_cols,
                                           index_filter_func)

            new_data = OrderedDict()
            self_no_on = self_reset.drop(on)
            other_no_on = other_reset.drop(on)
            self_new_names, other_new_names = _compute_new_names(self_no_on._gather_column_names(),
                                                                 other_no_on._gather_column_names(),
                                                                 suffixes)

            for column_name, new_name in zip(self_no_on, self_new_names):
                new_data[new_name] = data_filter_func(self_no_on[column_name], weld_objects_indexes[0], new_index)

            for column_name, new_name in zip(other_no_on, other_new_names):
                new_data[new_name] = data_filter_func(other_no_on[column_name], weld_objects_indexes[1], new_index)

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

    def drop_duplicates(self, subset=None, keep='min'):
        """Return DataFrame with duplicate rows (excluding index) removed,
        optionally only considering subset columns.

        Note that the row order is NOT maintained due to hashing.

        Parameters
        ----------
        subset : list of str, optional
            Which columns to consider
        keep : {'+', '*', 'min', 'max'}, optional
            What to select from the duplicate rows. These correspond to the possible merge operations in Weld.
            Note that '+' and '-' might produce unexpected results for strings.

        Returns
        -------
        DataFrame
            DataFrame without duplicate rows.

        """
        subset = check_and_obtain_subset_columns(subset, self)

        df = self.reset_index()
        df_names = df._gather_column_names()
        subset_indices = [df_names.index(col_name) for col_name in subset]

        weld_objects = weld_drop_duplicates(df._gather_data_for_weld(),
                                            df._gather_weld_types(),
                                            subset_indices,
                                            keep)

        index_data = self.index._gather_data(name=None)
        new_index = [Index(weld_objects[i], v.dtype, k)
                     for i, k, v in zip(list(range(len(index_data))), index_data.keys(), index_data.values())]
        if len(new_index) > 1:
            new_index = MultiIndex(new_index, self.index._gather_names())
        else:
            new_index = new_index[0]

        new_data = OrderedDict((sr.name, Series(obj, new_index, sr.dtype, sr.name))
                               for sr, obj in zip(self._iter(), weld_objects[len(index_data):]))

        return DataFrame(new_data, new_index)

    def dropna(self, subset=None):
        """Remove missing values according to Baloo's convention.

        Parameters
        ----------
        subset : list of str, optional
            Which columns to check for missing values in.

        Returns
        -------
        DataFrame
            DataFrame with no null values in columns.

        """
        subset = check_and_obtain_subset_columns(subset, self)
        not_nas = [v.notna() for v in self[subset]._iter()]
        and_filter = reduce(lambda x, y: x & y, not_nas)

        return self[and_filter]

    def fillna(self, value):
        """Returns DataFrame with missing values replaced with value.

        Parameters
        ----------
        value : {int, float, bytes, bool} or dict
            Scalar value to replace missing values with. If dict, replaces missing values
            only in the key columns with the value scalar.

        Returns
        -------
        DataFrame
            With missing values replaced.

        """
        if is_scalar(value):
            new_data = OrderedDict((column.name, column.fillna(value))
                                   for column in self._iter())

            return DataFrame(new_data, self.index)
        elif isinstance(value, dict):
            new_data = OrderedDict((column.name, column.fillna(value[column.name]) if column.name in value else column)
                                   for column in self._iter())

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can only fill na given a scalar or a dict mapping columns to their respective scalar')

    def groupby(self, by):
        """Group by certain columns, excluding index.

        Simply reset_index if desiring to group by some index column too.

        Parameters
        ----------
        by  : str or list of str
            Column(s) to groupby.

        Returns
        -------
        DataFrameGroupBy
            Object encoding the groupby operation.

        """
        check_str_or_list_str(by)
        by = as_list(by)
        if len(set(by)) == len(self._data):
            raise ValueError('Cannot groupby all columns')

        from .groupby import DataFrameGroupBy

        return DataFrameGroupBy(self, by)

    @classmethod
    def from_pandas(cls, df):
        """Create baloo DataFrame from pandas DataFrame.

        Parameters
        ----------
        df : pandas.frame.DataFrame

        Returns
        -------
        DataFrame

        """
        from pandas import DataFrame as PandasDataFrame, Index as PandasIndex, MultiIndex as PandasMultiIndex

        check_type(df, PandasDataFrame)

        if isinstance(df.index, PandasIndex):
            baloo_index = Index.from_pandas(df.index)
        elif isinstance(df.index, PandasMultiIndex):
            baloo_index = MultiIndex.from_pandas(df.index)
        else:
            raise TypeError('Cannot convert pandas index of type={} to baloo'.format(type(df.index)))

        baloo_data = OrderedDict((column_name, _series_from_pandas(df[column_name], baloo_index))
                                 for column_name in df)

        return DataFrame(baloo_data, baloo_index)

    def to_pandas(self):
        """Convert to pandas DataFrame.

        Note the data is expected to be evaluated.

        Returns
        -------
        pandas.frame.DataFrame

        """
        from pandas import DataFrame as PandasDataFrame

        pandas_index = self.index.to_pandas()
        pandas_data = OrderedDict((column.name, column.to_pandas())
                                  for column in self._iter())

        return PandasDataFrame(pandas_data, pandas_index)

    # TODO: once more are implemented, perhaps move to a mixin-like class for io
    def to_csv(self, filepath, sep=',', header=True, index=True):
        """Save DataFrame as csv.

        Parameters
        ----------
        filepath : str
        sep : str, optional
            Separator used between values.
        header : bool, optional
            Whether to save the header.
        index : bool, optional
            Whether to save the index columns.

        Returns
        -------
        None

        """
        from ..io import to_csv

        return to_csv(self, filepath, sep=sep, header=header, index=index)


def _default_index(dataframe_data, length):
    if length is not None:
        return default_index(length)
    else:
        if len(dataframe_data) == 0:
            return default_index(0)
        else:
            # must encode from a random column then
            return default_index(dataframe_data[list(dataframe_data.keys())[0]])


# TODO: if there's no index, pandas tries to get an index from the data
# TODO if there are multiple indexes, there's an outer join on the index;
# TODO: if an index is passed though, it overrides any index inferences ^
def _process_index(index, data, length):
    if index is None:
        return _default_index(data, length)
    else:
        return check_type(index, (Index, MultiIndex))


def _check_input_data(data):
    if data is None:
        return OrderedDict()
    else:
        check_type(data, dict)
        check_inner_types(data.values(), (np.ndarray, Series, list))

        return data


def _process_data(data, index):
    for k, v in data.items():
        # TODO: pandas does alignment here
        if isinstance(v, Series):
            v.name = k
            v.index = index
        else:
            # must be ndarray or list
            data[k] = Series(v, index, name=k)

    return data


def _infer_length(index, data):
    index_length = None
    if index is not None:
        index_length = infer_length(index._gather_data().values())

    if index_length is not None:
        return index_length
    else:
        return infer_length(data.values())


def _obtain_length(length, dataframe_data):
    if length is not None:
        return length
    else:
        # first check again for raw data
        length = infer_length(dataframe_data.values())
        if length is None:
            keys = list(dataframe_data.keys())
            # empty DataFrame
            if len(keys) == 0:
                return 0
            # pick first column (which is a Series) and encode its length
            length = weld_count(dataframe_data[keys[0]].weld_expr)

        return length


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
        data_arg = 'column.weld_expr'

    new_indexes = []
    data_arg = data_arg if how != 'outer' else data_arg + '[i]'
    for i, column_name in enumerate(on):
        column = extract_index_from._data[column_name]
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


def _drop_str_columns(df):
    """

    Parameters
    ----------
    df : DataFrame

    Returns
    -------

    """
    str_columns = filter(lambda pair: pair[1].char == 'S', df._gather_dtypes().items())
    str_column_names = list(map(lambda pair: pair[0], str_columns))

    return df.drop(str_column_names)


def check_and_obtain_subset_columns(columns, df):
    check_type(columns, list)
    check_inner_types(columns, str)

    if columns is None:
        return df._gather_column_names()
    elif len(columns) < 1:
        raise ValueError('Need at least one column')
    elif not set(columns).issubset(set(df._gather_column_names())):
        raise ValueError('Subset={} is not all part of the columns={}'.format(columns, df._gather_column_names()))
    else:
        return columns
