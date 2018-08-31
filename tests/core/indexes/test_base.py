import numpy as np

from baloo import Index
from baloo.weld import create_placeholder_weld_object


def assert_index_equal(actual, expected):
    actual = actual.evaluate()
    expected = expected.evaluate()

    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.dtype == expected.dtype
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
