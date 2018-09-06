from .frame import DataFrame
from .series import Series
from .utils import check_type
from ..weld import LazyScalarResult, weld_iloc_int


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
                # np.array of WeldObjects is not supported, so cannot just do the above for each column/Series. We also
                # don't want to eagerly compute this so need to have all columns in Weld (e.g. as a struct) and select
                # from each to build a single vec result. Perhaps too expensive atm.
                raise NotImplementedError('Requires bringing all data into Weld for a single evaluation. Postponed')
        elif isinstance(item, slice):
            return self.data[item]
        else:
            raise TypeError('Expected an int or a slice')
