from collections import OrderedDict

from .frame import DataFrame
from .indexes import Index, MultiIndex
from .series import Series
from ..weld import weld_groupby, weld_groupby_aggregate, weld_vec_of_struct_to_struct_of_vec, LazyStructOfVecResult, \
    Cache, extract_placeholder_weld_objects


class DataFrameGroupBy(object):
    """Object encoding a groupby operation."""
    def __init__(self, df, by: list):
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

    # TODO: this shall decide if we can use dictmerger directly OR need groupmerger
    def _group(self, aggregations):
        return weld_groupby(self._data_for_weld,
                            self._weld_types,
                            self._by_indices)

    def _aggregate(self, aggregation):
        grouped = self._group(aggregation)
        weld_types, vec_of_struct = weld_groupby_aggregate(grouped,
                                                           self._weld_types,
                                                           self._by_indices,
                                                           aggregation)
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

        new_data = OrderedDict((sr.name, Series(obj, new_index, sr.dtype, sr.name))
                               for sr, obj in zip(self._columns_df._iter(), weld_objects[len(self._by):]))

        return DataFrame(new_data, new_index)

    def min(self):
        return self._aggregate('min')

    def max(self):
        return self._aggregate('max')

    def sum(self):
        return self._aggregate('+')

    def prod(self):
        return self._aggregate('*')


def _compute_by_indices(by, df):
    column_names = df._gather_column_names()

    return [column_names.index(column_name) for column_name in by]
