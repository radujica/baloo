from baloo import Series
from .test_series import assert_series_equal


class TestSeriesStr(object):
    # TODO: parametrize?
    def test_lower(self, series_str, index_i64):
        actual = series_str.str.lower()
        expected = Series([b'a', b'abc', b'goosfraba', b'   dc  ', b'secretariat'], index_i64)

        assert_series_equal(actual, expected)

    def test_upper(self, series_str, index_i64):
        actual = series_str.str.upper()
        expected = Series([b'A', b'ABC', b'GOOSFRABA', b'   DC  ', b'SECRETARIAT'], index_i64)

        assert_series_equal(actual, expected)

    def test_capitalize(self, series_str, index_i64):
        actual = series_str.str.capitalize()
        expected = Series([b'A', b'Abc', b'Goosfraba', b'   dC  ', b'SecrETariat'], index_i64)

        assert_series_equal(actual, expected)
