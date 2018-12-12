import numpy as np
import pytest

from baloo import Series
from .test_series import assert_series_equal


class TestSeriesStr(object):
    @pytest.mark.parametrize('func, kwargs, expected_data', [
        ('lower', {}, [b'a', b'abc', b'goosfraba', b'   dc  ', b'secretariat']),
        ('upper', {}, [b'A', b'ABC', b'GOOSFRABA', b'   DC  ', b'SECRETARIAT']),
        ('capitalize', {}, [b'A', b'Abc', b'Goosfraba', b'   dc  ', b'Secretariat']),
        ('get', {'i': 1}, [b'None', b'b', b'o', b' ', b'e']),
        ('get', {'i': -1}, [b'a', b'c', b'a', b' ', b't']),
        ('slice', {'start': 1, 'stop': 5, 'step': 2}, [b'', b'b', b'os', b' d', b'er'])
    ])
    def test_str_operations(self, func, kwargs, expected_data, series_str, index_i64):
        actual = getattr(series_str.str, func)(**kwargs)
        expected = Series(expected_data, index_i64, np.bytes_)

        assert_series_equal(actual, expected)

    @pytest.mark.parametrize('func, kwargs, expected_data', [
        ('contains', {'pat': 'ab'}, [True, True, True, True, False]),
        ('startswith', {'pat': 'za'}, [False, True, True, False, False]),
        ('endswith', {'pat': 'bz'}, [True, True, False, False, False]),
        ('find', {'sub': 'ab'}, [0, 1, 1, 2, -1]),
        ('find', {'sub': 'ab', 'start': 1, 'end': 3}, [-1, 1, 1, -1, -1]),
        ('replace', {'pat': 'ab', 'rep': 'x'}, [b'xz', b'zxz', b'zx', b'  x  ', b'a']),
        ('split', {'pat': 'ab', 'side': 'left'}, [b'', b'z', b'z', b'  ', b'a']),
        ('split', {'pat': 'ab', 'side': 'right'}, [b'z', b'z', b'', b'  ', b'a'])
    ])
    def test_str_operations_other(self, func, kwargs, expected_data, series_str_2, index_i64):
        actual = getattr(series_str_2.str, func)(**kwargs)
        expected = Series(expected_data, index_i64)

        assert_series_equal(actual, expected)

    def test_strip(self, series_str, index_i64):
        actual = Series([b' a', b'Abc ', b'  dC   ', b'  ', b'secrET ariat'], index_i64).str.strip()
        expected = Series([b'a', b'Abc', b'dC', b'', b'secrET ariat'], index_i64)

        assert_series_equal(actual, expected)
