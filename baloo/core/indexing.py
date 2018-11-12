from collections import OrderedDict

from .frame import DataFrame
from .series import Series, _series_iloc, _series_iloc_with_missing
from .utils import check_type, check_weld_long_array
from ..weld import LazyScalarResult, weld_iloc_int, LazyArrayResult


class _ILocIndexer(object):
    """Implements iloc indexing.

    Attributes
    ----------
    data : Series or DataFrame
        Which data to select from by int/slice indexing.

    """
    def __init__(self, data):
        self.data = check_type(data, (Series, DataFrame))

    def __getitem__(self, item):
        if isinstance(item, int):
            if isinstance(self.data, Series):
                return LazyScalarResult(weld_iloc_int(self.data.weld_expr,
                                                      item),
                                        self.data.weld_type)
            elif isinstance(self.data, DataFrame):
                # requires the supertype 'object' in a numpy array which perhaps in Weld could be strings;
                # this is because the expected return is a Series and method needs to put ints/floats/strings in the
                # same data structure, i.e. the same numpy ndarray
                raise NotImplementedError()
        elif isinstance(item, slice):
            return self.data[item]
        elif isinstance(item, LazyArrayResult):
            check_weld_long_array(item)
            item = item.weld_expr

            if isinstance(self.data, Series):
                return _series_iloc(self.data, item, self.data.index._iloc_indices(item))
            elif isinstance(self.data, DataFrame):
                new_index = self.data.index._iloc_indices(item)

                new_data = OrderedDict((column.name, _series_iloc(column, item, new_index))
                                       for column in self.data._iter())

                return DataFrame(new_data, new_index)
        else:
            raise TypeError('Expected an int, slice, or indices array')

    def _iloc_with_missing(self, item):
        if isinstance(self.data, Series):
            return _series_iloc_with_missing(self.data, item, self.data.index._iloc_indices_with_missing(item))
        elif isinstance(self.data, DataFrame):
            raise NotImplementedError()
