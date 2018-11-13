import numpy as np
import pytest

from baloo import Index


def assert_index_equal(actual, expected, sort=False):
    actual = actual.evaluate()
    expected = expected.evaluate()

    actual_values = actual.values
    expected_values = expected.values
    if sort:
        actual_values = np.sort(actual_values)
        expected_values = np.sort(expected_values)
    np.testing.assert_array_equal(actual_values, expected_values)

    assert actual.dtype.char == expected.dtype.char
    assert actual._length == expected._length
    # might seem redundant but testing the __len__ function
    assert len(actual) == len(expected)
    assert actual.name == expected.name


class TestBaseIndex(object):
    def test_init_list(self):
        data = [1, 2, 3]
        actual = Index(data)
        expected = Index(np.array(data))

        assert_index_equal(actual, expected)

    def test_evaluate(self, data_i64):
        actual = Index(data_i64)
        expected = Index(data_i64, np.dtype(np.int64), None)

        assert_index_equal(actual, expected)

    def test_len_raw(self, data_i64):
        ind = Index(data_i64, np.dtype(np.int64))

        actual = len(ind)
        expected = 5

        assert actual == expected

    def test_len_lazy(self, data_i64_lazy):
        ind = Index(data_i64_lazy, np.dtype(np.int64))

        actual = len(ind)
        expected = 5

        assert actual == expected

    def test_comparison(self, index_i64):
        actual = index_i64 < 3
        expected = Index(np.array([True, True, True, False, False]))

        assert_index_equal(actual, expected)

    def test_filter(self, index_i64):
        actual = index_i64[Index(np.array([False, True, True, False, False]))]
        expected = Index(np.array([1, 2]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_slice(self, index_i64):
        actual = index_i64[1:3]
        expected = Index(np.array([1, 2]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_head(self, index_i64):
        actual = index_i64.head(2)
        expected = Index(np.array([0, 1]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_tail(self, index_i64):
        actual = index_i64.tail(2)
        expected = Index(np.array([3, 4]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    # implicitly tests if one can apply operation with Series too
    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.arange(3, 8, dtype=np.float32)),
        ('-', np.arange(-1, 4, dtype=np.float32)),
        ('*', np.arange(2, 11, 2, dtype=np.float32)),
        ('/', np.array([0.5, 1, 1.5, 2, 2.5], dtype=np.float32)),
        ('**', np.array([1, 4, 9, 16, 25], dtype=np.float32))
    ])
    def test_op_array(self, operation, expected_data, data_f32, op_array_other):
        data = Index(data_f32)

        actual = eval('data {} op_array_other'.format(operation))
        expected = Index(expected_data, np.dtype(np.float32))

        assert_index_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.arange(3, 8, dtype=np.float32)),
        ('-', np.arange(-1, 4, dtype=np.float32)),
        ('*', np.arange(2, 11, 2, dtype=np.float32)),
        ('/', np.array([0.5, 1, 1.5, 2, 2.5], dtype=np.float32)),
        ('**', np.array([1, 4, 9, 16, 25], dtype=np.float32))
    ])
    def test_op_scalar(self, operation, expected_data, data_f32):
        ind = Index(data_f32)

        actual = eval('ind {} 2'.format(operation))
        expected = Index(expected_data, np.dtype(np.float32))

        assert_index_equal(actual, expected)

    def test_isna(self):
        ind = Index([3, 2, -999, 4, -999])

        actual = ind.isna()
        expected = Index([False, False, True, False, True], np.dtype(np.bool))

        assert_index_equal(actual, expected)

    def test_dropna(self):
        ind = Index([3, 2, -999, 4, -999])

        actual = ind.dropna()
        expected = Index([3, 2, 4], np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_fillna(self):
        ind = Index([3, 2, -999, 4, -999])

        actual = ind.fillna(15)
        expected = Index([3, 2, 15, 4, 15], np.dtype(np.int64))

        assert_index_equal(actual, expected)
