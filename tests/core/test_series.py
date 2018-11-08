import numpy as np
import pytest

from baloo import Series, RangeIndex, Index
from baloo.weld import create_placeholder_weld_object
from .indexes.utils import assert_indexes_equal


def assert_series_equal(actual, expected, almost=None):
    actual = actual.evaluate()
    expected = expected.evaluate()

    # for checking floats
    if almost is not None:
        np.testing.assert_array_almost_equal(actual.values, expected.values, almost)
    else:
        np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.dtype.char == expected.dtype.char
    assert actual._length == expected._length
    # might seem redundant but testing the __len__ function
    assert len(actual) == len(expected)
    assert actual.name == expected.name
    assert_indexes_equal(actual.index, expected.index)


class TestSeries(object):
    def test_evaluate(self, data_i64):
        actual = Series(data_i64)
        expected = Series(data_i64, RangeIndex(5))

        assert_series_equal(actual, expected)

    def test_len_raw(self, series_i64):
        actual = len(series_i64)
        expected = 5

        assert actual == expected

    def test_init_weld_object_no_dtype(self, data_i64_lazy):
        with pytest.raises(ValueError):
            Series(data_i64_lazy)

    def test_len_lazy(self, data_i64):
        weld_obj = create_placeholder_weld_object(data_i64)
        sr = Series(weld_obj, dtype=np.dtype(np.int64))

        actual = len(sr)
        expected = 5

        assert actual == expected

    @pytest.mark.parametrize('comparison, expected_data', [
        ('<', np.array([True, False, False, False, False])),
        ('<=', np.array([True, True, False, False, False])),
        ('==', np.array([False, True, False, False, False])),
        ('!=', np.array([True, False, True, True, True])),
        ('>=', np.array([False, True, True, True, True])),
        ('>', np.array([False, False, True, True, True]))
    ])
    def test_comparison(self, comparison, expected_data, series_i64, index_i64):
        actual = eval('series_i64 {} 2'.format(comparison))
        expected = Series(expected_data, index_i64, np.dtype(np.bool))

        assert_series_equal(actual, expected)

    def test_filter(self, series_i64):
        actual = series_i64[series_i64 != 2]
        expected = Series(np.array([1, 3, 4, 5]), Index(np.array([0, 2, 3, 4])), np.dtype(np.int64))

        assert_series_equal(actual, expected)

    def test_filter_combined(self, series_i64):
        actual = series_i64[(series_i64 != 2) & (series_i64 != 4)]
        expected = Series(np.array([1, 3, 5]), Index(np.array([0, 2, 4])), np.dtype(np.int64))

        assert_series_equal(actual, expected)

    def test_filter_str(self, series_str):
        actual = series_str[series_str != 'abc']
        expected = Series(np.array(['a', 'goosfraba', '   dc  ', 'secretariat'], dtype=np.bytes_),
                          Index(np.array([0, 2, 3, 4])), np.dtype(np.bytes_))

        assert_series_equal(actual, expected)

    def test_slice(self, series_f32):
        actual = series_f32[1:3]
        expected = Series(np.array([2, 3]), Index(np.array([1, 2])), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected_data', [
        ('&', np.array([True, False, False, False, False])),
        ('|', np.array([True, True, True, False, False]))
    ])
    def test_bool_operations(self, operation, expected_data, index_i64):
        sr1 = Series(np.array([True, True, False, False, False]), index_i64, np.dtype(np.bool))
        sr2 = Series(np.array([True, False, True, False, False]), index_i64, np.dtype(np.bool))

        actual = eval('sr1 {} sr2'.format(operation))
        expected = Series(expected_data, index_i64, np.dtype(np.bool))

        assert_series_equal(actual, expected)

    def test_invert(self, index_i64):
        sr = Series(np.array([True, False, True, False, False]), index_i64, np.dtype(np.bool))

        actual = ~sr
        expected = Series(np.array([False, True, False, True, True]), index_i64, np.dtype(np.bool))

        assert_series_equal(actual, expected)

    def test_head(self, series_f32):
        actual = series_f32.head(2)
        expected = Series(np.array([1, 2]), RangeIndex(2), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    def test_tail(self, series_f32):
        actual = series_f32.tail(2)
        expected = Series(np.array([4, 5]), Index(np.array([3, 4])), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    def test_iloc_int(self, series_f32):
        actual = series_f32.iloc[2].evaluate()
        expected = 3

        assert actual == expected

    def test_iloc_slice(self, series_f32):
        actual = series_f32.iloc[1:3]
        expected = Series(np.array([2, 3]), Index(np.array([1, 2])), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    def test_iloc_indices(self, series_f32):
        indices = Series(np.array([0, 3, 4]))

        actual = series_f32.iloc[indices]
        expected = Series(np.array([1, 4, 5], dtype=np.float32), Index(np.array([0, 3, 4])), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    def test_iloc_indices_missing(self, series_f32):
        indices = Series(np.array([0, 3, 5]))

        actual = series_f32.iloc._iloc_with_missing(indices.weld_expr)
        expected = Series(np.array([1, 4, -999], dtype=np.float32), Index(np.array([0, 3, -999])), np.dtype(np.float32))

        assert_series_equal(actual, expected)

    @pytest.fixture
    def op_array_other(self):
        return Series(np.array([2] * 5).astype(np.float32))

    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.array(np.arange(3, 8), dtype=np.float32)),
        ('-', np.array(np.arange(-1, 4), dtype=np.float32)),
        ('*', np.array(np.arange(2, 11, 2), dtype=np.float32)),
        ('/', np.array([0.5, 1, 1.5, 2, 2.5], dtype=np.float32)),
        ('**', np.array([1, 4, 9, 16, 25], dtype=np.float32))
    ])
    def test_op_array(self, operation, expected_data, series_f32, index_i64, op_array_other):
        actual = eval('series_f32 {} op_array_other'.format(operation))
        expected = Series(expected_data, index_i64, np.dtype(np.float32))

        assert_series_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.array(np.arange(3, 8))),
        ('-', np.array(np.arange(-1, 4))),
        ('*', np.array(np.arange(2, 11, 2))),
        ('/', np.array([0.5, 1, 1.5, 2, 2.5])),
        ('**', np.array([1, 4, 9, 16, 25]))
    ])
    def test_op_scalar(self, operation, expected_data, index_i64, series_f32):
        actual = eval('series_f32 {} 2'.format(operation))
        expected = Series(expected_data, index_i64, np.dtype(np.float32))

        assert_series_equal(actual, expected)

    @pytest.mark.parametrize('aggregation, expected', [
        ('min', 1),
        ('max', 5),
        ('sum', 15),
        ('prod', 120),
        ('count', 5),
        ('mean', 3.0),
        ('var', 2.5),
        ('std', 1.5811388)
    ])
    def test_aggregation(self, aggregation, expected, series_i64):
        actual = getattr(series_i64, aggregation)().evaluate()

        np.testing.assert_almost_equal(actual, expected, 5)

    def test_agg(self, series_f32):
        aggregations = ['max', 'var', 'count', 'mean']
        actual = series_f32.agg(aggregations)

        expected = Series(np.array([5, 2.5, 5, 3], dtype=np.float64),
                          Index(np.array(aggregations, dtype=np.bytes_)),
                          np.dtype(np.float64))

        assert_series_equal(actual, expected)
