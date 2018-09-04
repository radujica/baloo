from collections import OrderedDict

import numpy as np
from tabulate import tabulate
from weld.types import WeldBit

from .indexes import RangeIndex, Index
from .series import Series
from .utils import check_type, is_scalar, valid_int_slice
from ..weld import weld_count, LazyResult


class DataFrame(object):
    """ Weld-ed pandas DataFrame.

    Attributes
    ----------
    data : dict
        Data as a dict of column names -> numpy.ndarray or Series.
    index : Index or RangeIndex
        Index of the data.

    See Also
    --------
    pandas.DataFrame

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

        str_data[self.index.name] = DataFrame._shorten_data(self.index.values)

        for column_name in self:
            str_data[column_name] = DataFrame._shorten_data(self[column_name].values)

        return tabulate(str_data, headers='keys')

    def _comparison(self, other, comparison):
        if is_scalar(other):
            new_data = {column_name: self[column_name]._comparison(other, comparison)
                        for column_name in self}

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('Can currently only compare with scalars')

    def __lt__(self, other):
        return self._comparison(other, '<')

    def __le__(self, other):
        return self._comparison(other, '<=')

    def __eq__(self, other):
        return self._comparison(other, '==')

    def __ne__(self, other):
        return self._comparison(other, '!=')

    def __ge__(self, other):
        return self._comparison(other, '>=')

    def __gt__(self, other):
        return self._comparison(other, '>')

    # TODO: handle empty
    def __getitem__(self, item):
        """Select from the DataFrame.

        Supported functionality:

        - Select column: df[<column-name>]
        - Filter: df[df[<column>] <comparison> <scalar>]

        """
        if isinstance(item, str):
            value = self.data[item]

            if isinstance(value, np.ndarray):
                value = Series(value, self.index, value.dtype, item)
                # store the newly created Series to avoid remaking it
                self.data[item] = value

            return value
        elif isinstance(item, LazyResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            new_index = self.index[item]
            new_data = {column_name: Series._filter_series(self[column_name], item, new_index)
                        for column_name in self}

            return DataFrame(new_data, new_index)
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            new_index = self.index[item]
            new_data = {column_name: Series._slice_series(self[column_name], item, new_index)
                        for column_name in self}

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
