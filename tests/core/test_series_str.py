from baloo import Series
from .test_series import assert_series_equal


class TestSeriesStr(object):
    def test_lower(self, series_str, index_i64):
        actual = series_str.str.lower()
        expected = Series([b'a', b'abc', b'goosfraba', b'   dc  ', b'secretariat'], index_i64)

        assert_series_equal(actual, expected)
