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

    def test_get(self, series_str, index_i64):
        actual = series_str.str.get(1)
        expected = Series([b'None', b'b', b'o', b' ', b'e'], index_i64)

        assert_series_equal(actual, expected)

    def test_get_negative(self, series_str, index_i64):
        actual = series_str.str.get(-1)
        expected = Series([b'a', b'c', b'a', b' ', b't'], index_i64)

        assert_series_equal(actual, expected)

    def test_strip(self, series_str, index_i64):
        actual = Series([b' a', b'Abc ', b'  dC   ', b'  ', b'secrET ariat'], index_i64).str.strip()
        expected = Series([b'a', b'Abc', b'dC', b'', b'secrET ariat'], index_i64)

        assert_series_equal(actual, expected)

    def test_slice(self, series_str, index_i64):
        actual = series_str.str.slice(1, 5, 2)
        expected = Series([b'', b'b', b'os', b' d', b'er'], index_i64)

        assert_series_equal(actual, expected)

    def test_contains(self, series_str, index_i64):
        actual = Series([b'abz', b'zabz', b'zab', b' ab  ', b'a']).str.contains('ab')
        expected = Series([True, True, True, True, False], index_i64)

        assert_series_equal(actual, expected)
