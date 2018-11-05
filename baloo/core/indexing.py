from collections import OrderedDict

from .frame import DataFrame
from .series import Series
from .utils import check_type, check_weld_long_array
from ..weld import LazyScalarResult, weld_iloc_int, LazyArrayResult, weld_iloc_indices, weld_iloc_indices_with_missing


# TODO: perhaps instead make Series and DataFrame inherit an ILoc interface and each implement their own
# TODO internal methods for iloc; then can just call directly on self.data
# TODO ~ simplifying this class but making others longer
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

            return self._iloc(item.weld_expr)
        else:
            raise TypeError('Expected an int, slice, or indices array')

    def _iloc_series(self, item, new_index):
        return Series(weld_iloc_indices(self.data.weld_expr,
                                        self.data.weld_type,
                                        item),
                      new_index,
                      self.data.dtype,
                      self.data.name)

    def _iloc(self, item):
        if isinstance(self.data, Series):
            return self._iloc_series(item, self.data.index._iloc_indices(item))
        elif isinstance(self.data, DataFrame):
            # this should only happen when called by _ILocIndexer.__getitem__
            new_index = self.data.index._iloc_indices(item)

            new_data = OrderedDict()
            for column_name in self.data:
                new_data[column_name] = self.data[column_name].iloc._iloc_series(item, new_index)

            return DataFrame(new_data, new_index)

    def _iloc_series_with_missing(self, item, new_index):
        return Series(weld_iloc_indices_with_missing(self.data.weld_expr,
                                                     self.data.weld_type,
                                                     item),
                      new_index,
                      self.data.dtype,
                      self.data.name)

    def _iloc_with_missing(self, item):
        if isinstance(self.data, Series):
            return self._iloc_series_with_missing(item, self.data.index._iloc_indices_with_missing(item))
        elif isinstance(self.data, DataFrame):
            raise NotImplementedError()
