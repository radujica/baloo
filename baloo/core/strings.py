from .series import Series
from .utils import check_type
from ..weld import weld_str_lower, weld_str_upper, weld_str_capitalize, weld_str_get, weld_str_strip, weld_str_slice, \
    weld_str_contains, weld_to_numpy_dtype, WeldBit, weld_str_startswith, weld_str_endswith, weld_str_find, WeldLong, \
    weld_str_replace, weld_str_split


class StringMethods(object):
    def __init__(self, data):
        self._data = check_type(data, Series)

    def lower(self):
        """Convert all characters to lowercase.

        Returns
        -------
        Series

        """
        return _series_str_result(self, weld_str_lower)

    def upper(self):
        """Convert all characters to uppercase.

        Returns
        -------
        Series

        """
        return _series_str_result(self, weld_str_upper)

    def capitalize(self):
        """Convert first character to uppercase and remainder to lowercase.

        Returns
        -------
        Series

        """
        return _series_str_result(self, weld_str_capitalize)

    def get(self, i):
        """Extract i'th character of each element.

        Parameters
        ----------
        i : int

        Returns
        -------
        Series

        """
        check_type(i, int)

        return _series_str_result(self, weld_str_get, i=i)

    def strip(self):
        """Strip whitespace from start and end of each element.

        Note it currently only looks for whitespace (ASCII 32), not tabs or EOL.

        Returns
        -------
        Series

        """
        return _series_str_result(self, weld_str_strip)

    def slice(self, start=None, stop=None, step=None):
        """Slice substrings from each element.

        Note that negative step is currently not supported.

        Parameters
        ----------
        start : int
        stop : int
        step : int

        Returns
        -------
        Series

        """
        check_type(start, int)
        check_type(stop, int)
        check_type(step, int)

        if step is not None and step < 0:
            raise ValueError('Only positive steps are currently supported')

        return _series_str_result(self, weld_str_slice, start=start, stop=stop, step=step)

    def contains(self, pat):
        """Test if pat is included within elements.

        Parameters
        ----------
        pat : str

        Returns
        -------
        Series

        """
        check_type(pat, str)

        return _series_bool_result(self, weld_str_contains, pat=pat)

    def startswith(self, pat):
        """Test if elements start with pat.

        Parameters
        ----------
        pat : str

        Returns
        -------
        Series

        """
        check_type(pat, str)

        return _series_bool_result(self, weld_str_startswith, pat=pat)

    def endswith(self, pat):
        """Test if elements end with pat.

        Parameters
        ----------
        pat : str

        Returns
        -------
        Series

        """
        check_type(pat, str)

        return _series_bool_result(self, weld_str_endswith, pat=pat)

    def find(self, sub, start=0, end=None):
        """Test if elements contain substring.

        Parameters
        ----------
        sub : str
        start : int, optional
            Index to start searching from.
        end : int, optional
            Index to stop searching from.

        Returns
        -------
        Series

        """
        check_type(sub, str)
        check_type(start, int)
        check_type(end, int)

        if end is not None and start >= end:
            raise ValueError('End must be greater than start')

        return Series(weld_str_find(self._data.values, sub, start, end),
                      self._data.index,
                      weld_to_numpy_dtype(WeldLong()),
                      self._data.name)

    # TODO: replace multiple occurrences, not just first
    def replace(self, pat, rep):
        """Replace first occurrence of pat with rep in each element.

        Parameters
        ----------
        pat : str
        rep : str

        Returns
        -------
        Series

        """
        check_type(pat, str)
        check_type(rep, str)

        return _series_str_result(self, weld_str_replace, pat=pat, rep=rep)

    # TODO: rsplit
    def split(self, pat, side='left'):
        """Split once each element from the left and select a side to return.

        Note this is unlike pandas split in that it essentially combines the split with a select.

        Parameters
        ----------
        pat : str
        side : {'left', 'right'}
            Which side of the split to select and return in each element.

        Returns
        -------
        Series

        """
        check_type(pat, str)
        check_type(side, str)

        # don't want this made with the object
        _split_mapping = {
            'left': 0,
            'right': 1
        }

        if side not in _split_mapping:
            raise ValueError('Can only select left or right side of split')

        return _series_str_result(self, weld_str_split, pat=pat, side=_split_mapping[side])


def _series_str_result(series, func, **kwargs):
    return Series(func(series._data.values, **kwargs),
                  series._data.index,
                  series._data.dtype,
                  series._data.name)


def _series_bool_result(series, func, **kwargs):
    return Series(func(series._data.values, **kwargs),
                  series._data.index,
                  weld_to_numpy_dtype(WeldBit()),
                  series._data.name)
