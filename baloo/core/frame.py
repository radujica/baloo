from collections import OrderedDict

import numpy as np
from tabulate import tabulate

from ..weld import weld_count
from .indexes import RangeIndex, Index
from .series import Series
from .utils import check_type


class DataFrame(object):
    """ Weld-ed pandas DataFrame

    Attributes
    ----------
    data : dict
        column names -> np.array or Series
    index : Index or RangeIndex

    See also
    --------
    pandas.DataFrame

    """
    @staticmethod
    def _infer_length(data):
        for value in data.values():
            if isinstance(value, np.ndarray):
                return len(value)
            # must be a Series then
            elif isinstance(value.data, np.ndarray):
                return len(value.data)

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
            str -> np.ndarray or Series.
        index : Index or RangeIndex, optional
            Index linked to the data; it is assumed to be of the same length.
            RangeIndex by default.

        """
        self.data = DataFrame._check_data_types(data)
        self._dtypes = DataFrame._gather_dtypes(data)
        self._length = DataFrame._infer_length(data)
        self.index = DataFrame._default_dataframe_index(data, self._length) if index is None else index

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
                                                 repr(self.data.keys()))

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

        str_data[self.index.name] = DataFrame._shorten_data(self.index.data)

        for column_name in self:
            str_data[column_name] = DataFrame._shorten_data(self[column_name].data)

        return tabulate(str_data, headers='keys')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """ Evaluates by creating a str representation of the DataFrame.

        Parameters
        ----------
        see LazyResult

        Returns
        -------
        str

        """
        evaluated_index = self.index.evaluate(verbose, decode, passes, num_threads, apply_experimental)

        evaluated_data = OrderedDict()
        for column_name in self:
            evaluated_data[column_name] = self[column_name].evaluate(verbose, decode, passes,
                                                                     num_threads, apply_experimental)

        return DataFrame(evaluated_data, evaluated_index)

    def __getitem__(self, item):
        if isinstance(item, str):
            value = self.data[item]

            if isinstance(value, np.ndarray):
                value = Series(value, self.index, value.dtype, item)

            return value
        else:
            raise TypeError('Expected a column name as a string')

    def __iter__(self):
        for column_name in self.data:
            yield column_name
