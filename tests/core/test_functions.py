import numpy as np
import pytest

import baloo as bl
from .test_series import assert_series_equal


class TestUnary(object):
    def test_sqrt(self, series_f32, index_i64):
        actual = series_f32.apply(bl.sqrt)
        expected = bl.Series(np.array([1., 1.414214, 1.732051, 2., 2.236068], dtype=np.float32), index_i64)

        assert_series_equal(actual, expected, almost=5)


class TestRaw(object):
    def test_deco_no_args(self, series_unsorted, series_f32):
        actual = series_unsorted.apply(bl.sort)
        expected = series_f32

        assert_series_equal(actual, expected)

    def test_deco_good_args(self, series_unsorted, series_f32):
        actual = series_unsorted.apply(bl.sort, kind='q')
        expected = series_f32

        assert_series_equal(actual, expected)

    def test_deco_bad_args(self, series_unsorted):
        with pytest.raises(ValueError):
            series_unsorted.apply(bl.sort, kind='bla')

    def test_deco_unevaluated_data(self, series_unsorted):
        with pytest.raises(TypeError):
            series_unsorted.apply(bl.sqrt).apply(bl.sort, kind='bla')

    def test_deco_inline_no_args(self, series_unsorted, series_f32):
        actual = series_unsorted.apply(bl.raw(np.sort))
        expected = series_f32

        assert_series_equal(actual, expected)

    def test_deco_inline_good_args(self, series_unsorted, series_f32):
        actual = series_unsorted.apply(bl.raw(np.sort, kind='quicksort'))
        expected = series_f32

        assert_series_equal(actual, expected)

    def test_deco_inline_bad_args(self, series_unsorted):
        with pytest.raises(ValueError):
            series_unsorted.apply(bl.raw(np.sort, kind='bla'))

    def test_deco_inline_unevaluated_data(self, series_unsorted):
        with pytest.raises(TypeError):
            series_unsorted.apply(bl.sqrt).apply(bl.raw(np.sort, kind='bla'))

    def test_deco_inline_lambda(self, series_unsorted, series_f32):
        actual = series_unsorted.apply(bl.raw(lambda x: np.sort(x, kind='q')))
        expected = series_f32

        assert_series_equal(actual, expected)

    def test_deco_inline_lambda_bad_args(self, series_unsorted):
        with pytest.raises(ValueError):
            print(series_unsorted.apply(bl.raw(lambda x: np.sort(x, kind='bla'))))
