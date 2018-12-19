from collections import OrderedDict

from .frame import DataFrame
from .indexes import Index, MultiIndex
from .series import Series
from ..weld import weld_groupby, weld_groupby_aggregate, weld_vec_of_struct_to_struct_of_vec, LazyStructOfVecResult, \
    Cache, extract_placeholder_weld_objects, weld_to_numpy_dtype, WeldDouble, WeldLong, \
    weld_groupby_aggregate_dictmerger


class DataFrameGroupBy(object):
    """Object encoding a groupby operation."""
    _dictmerger_aggregations = {'+', '*', 'min', 'max'}

    def __init__(self, df, by):
        """Create a groupby object.

        Parameters
        ----------
        df : DataFrame
            Which will be grouped by.
        by : list of str
            Which columns to group by.

        """
        self._index_df = df[by]
        self._columns_df = df.drop(by)
        self._data_for_weld = df._gather_data_for_weld()
        self._weld_types = df._gather_weld_types()
        self._by = by
        self._by_indices = _compute_by_indices(self._by, df)

    def _group_dictmerger(self, aggregation):
        return weld_groupby_aggregate_dictmerger(self._data_for_weld,
                                                 self._weld_types,
                                                 self._by_indices,
                                                 aggregation)

    def _group_groupmerger(self, aggregation, result_type=None):
        grouped = weld_groupby(self._data_for_weld,
                               self._weld_types,
                               self._by_indices)

        return weld_groupby_aggregate(grouped,
                                      self._weld_types,
                                      self._by_indices,
                                      aggregation,
                                      result_type)

    def _group_aggregate(self, aggregation, result_type=None):
        if aggregation in self._dictmerger_aggregations:
            return self._group_dictmerger(aggregation)
        else:
            return self._group_groupmerger(aggregation, result_type)

    def _aggregate(self, aggregation, result_type=None):
        weld_types, vec_of_struct = self._group_aggregate(aggregation, result_type)
        struct_of_vec = weld_vec_of_struct_to_struct_of_vec(vec_of_struct, weld_types)
        intermediate_result = LazyStructOfVecResult(struct_of_vec, weld_types)
        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'group_aggr')

        weld_objects = extract_placeholder_weld_objects(dependency_name, len(weld_types), 'group_aggr')

        new_index = [Index(weld_objects[i], v, k)
                     for i, k, v in zip(list(range(len(self._index_df._data))),
                                        self._index_df._gather_column_names(),
                                        self._index_df._gather_dtypes().values())]
        if len(new_index) > 1:
            new_index = MultiIndex(new_index, [index.name for index in new_index])
        else:
            new_index = new_index[0]

        new_dtypes = self._columns_df._gather_dtypes().values() if result_type is None \
            else (weld_to_numpy_dtype(result_type) for _ in weld_types)
        new_data = OrderedDict((name, Series(obj, new_index, dtype, name))
                               for name, obj, dtype in zip(self._columns_df._gather_column_names(),
                                                           weld_objects[len(self._by):],
                                                           new_dtypes))

        return DataFrame(new_data, new_index)

    def min(self):
        return self._aggregate('min')

    def max(self):
        return self._aggregate('max')

    def sum(self):
        return self._aggregate('+')

    def prod(self):
        return self._aggregate('*')

    def mean(self):
        return self._aggregate('mean', result_type=WeldDouble())

    def var(self):
        return self._aggregate('var', result_type=WeldDouble())

    def std(self):
        return self._aggregate('std', result_type=WeldDouble())

    def size(self):
        return self._aggregate('size', result_type=WeldLong())


def _compute_by_indices(by, df):
    column_names = df._gather_column_names()

    return [column_names.index(column_name) for column_name in by]
