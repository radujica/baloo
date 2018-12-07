from .series import Series
from .utils import check_type
from ..weld import weld_str_lower, weld_str_upper


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
