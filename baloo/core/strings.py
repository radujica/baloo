from .series import Series
from .utils import check_type
from ..weld import weld_str_lower, weld_str_upper, weld_str_capitalize, weld_str_get, weld_str_strip, weld_str_slice, \
    weld_str_contains, weld_to_numpy_dtype, WeldBit, weld_str_startswith, weld_str_endswith, weld_str_find, WeldLong, \
    weld_str_replace


# TODO: docs
class StringMethods(object):
    def __init__(self, data):
        self._data = check_type(data, Series)

    def lower(self):
        return Series(weld_str_lower(self._data.values),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def upper(self):
        return Series(weld_str_upper(self._data.values),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def capitalize(self):
        return Series(weld_str_capitalize(self._data.values),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def get(self, i):
        check_type(i, int)

        return Series(weld_str_get(self._data.values, i),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def strip(self):
        return Series(weld_str_strip(self._data.values),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def slice(self, start=None, stop=None, step=None):
        check_type(start, int)
        check_type(stop, int)
        check_type(step, int)

        if step is not None and step < 0:
            raise ValueError('Only positive steps are currently supported')

        return Series(weld_str_slice(self._data.values, start, stop, step),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def contains(self, pat):
        check_type(pat, str)

        return Series(weld_str_contains(self._data.values, pat),
                      self._data.index,
                      weld_to_numpy_dtype(WeldBit()),
                      self._data.name)

    def startswith(self, pat):
        check_type(pat, str)

        return Series(weld_str_startswith(self._data.values, pat),
                      self._data.index,
                      weld_to_numpy_dtype(WeldBit()),
                      self._data.name)

    def endswith(self, pat):
        check_type(pat, str)

        return Series(weld_str_endswith(self._data.values, pat),
                      self._data.index,
                      weld_to_numpy_dtype(WeldBit()),
                      self._data.name)

    def find(self, sub, start=0, end=None):
        check_type(sub, str)
        check_type(start, int)
        check_type(end, int)

        if end is not None and start >= end:
            raise ValueError('End must be greater than start')

        return Series(weld_str_find(self._data.values, sub, start, end),
                      self._data.index,
                      weld_to_numpy_dtype(WeldLong()),
                      self._data.name)

    # TODO: replace multiple occurrences
    def replace(self, pat, rep):
        check_type(pat, str)
        check_type(rep, str)

        return Series(weld_str_replace(self._data.values, pat, rep),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)
