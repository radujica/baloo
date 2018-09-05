import numpy as np

from baloo import Index
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
