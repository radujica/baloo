import numpy as np
import pytest

from baloo import Index, Series
from baloo.weld import create_placeholder_weld_object


def assert_index_equal(actual, expected):
    actual = actual.evaluate()
    expected = expected.evaluate()

    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.dtype.char == expected.dtype.char
    # might seem redundant but testing the __len__ function
    assert len(actual) == len(expected)
    assert actual.name == expected.name


class TestBaseIndex(object):
    def test_evaluate(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        actual = Index(data)
        expected = Index(data, np.dtype(np.int64), None)

        assert_index_equal(actual, expected)

    def test_len_raw(self):
        ind = Index(np.array([1, 2, 3], dtype=np.int64))

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_len_lazy(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        weld_obj = create_placeholder_weld_object(data)
        ind = Index(weld_obj, np.dtype(np.int64))

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_comparison(self):
        ind = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        actual = ind < 3.0
        expected = Index(np.array([True, True, False, False, False]))

        assert_index_equal(actual, expected)

    def test_filter(self):
        ind = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        actual = ind[Index(np.array([False, True, True, False, False]))]
        expected = Index(np.array([2, 3]), np.dtype(np.float32))

        assert_index_equal(actual, expected)

    def test_slice(self):
        ind = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        actual = ind[1:3]
        expected = Index(np.array([2, 3]), np.dtype(np.float32))

        assert_index_equal(actual, expected)

    def test_head(self):
        ind = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        actual = ind.head(2)
        expected = Index(np.array([1, 2]), np.dtype(np.float32))

        assert_index_equal(actual, expected)

    def test_tail(self):
        ind = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))

        actual = ind.tail(2)
        expected = Index(np.array([4, 5]), np.dtype(np.float32))

        assert_index_equal(actual, expected)

    # implicitly tests if one can apply operation with Series too
    @pytest.mark.parametrize('operation, expected', [
        ('+', Index(np.array(np.arange(2, 7), dtype=np.float32), np.dtype(np.float32))),
        ('-', Index(np.array(np.arange(-2, 3), dtype=np.float32), np.dtype(np.float32))),
        ('*', Index(np.array(np.arange(0, 9, 2), dtype=np.float32), np.dtype(np.float32))),
        ('/', Index(np.array([0, 0.5, 1, 1.5, 2], dtype=np.float32), np.dtype(np.float32))),
        ('**', Index(np.array([0, 1, 4, 9, 16], dtype=np.float32), np.dtype(np.float32)))
    ])
    def test_op_array(self, operation, expected):
        data = Index(np.arange(5).astype(np.float32))
        other = Series(np.array([2] * 5).astype(np.float32))

        actual = eval('data {} other'.format(operation))

        assert_index_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected', [
        ('+', Index(np.array(np.arange(2, 7)), np.dtype(np.int64))),
        ('-', Index(np.array(np.arange(-2, 3)), np.dtype(np.int64))),
        ('*', Index(np.array(np.arange(0, 9, 2)), np.dtype(np.int64))),
        ('/', Index(np.array([0, 0, 1, 1, 2]), np.dtype(np.int64))),
        ('**', Index(np.array([0, 1, 4, 9, 16], dtype=np.float32), np.dtype(np.float32)))
    ])
    def test_op_scalar(self, operation, expected):
        data = np.arange(5)
        # hack until pow is fully supported by Weld
        if operation == '**':
            data = data.astype(np.float32)

        ind = Index(data)
        scalar = 2

        actual = eval('ind {} scalar'.format(operation))

        assert_index_equal(actual, expected)
