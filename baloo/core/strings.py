from .series import Series
from .utils import check_type
from ..weld import weld_str_lower, weld_str_upper, weld_str_capitalize, weld_str_get, weld_str_strip


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
        return Series(weld_str_get(self._data.values, i),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)

    def strip(self):
        return Series(weld_str_strip(self._data.values),
                      self._data.index,
                      self._data.dtype,
                      self._data.name)
