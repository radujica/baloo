import numpy as np

from baloo import Series, RangeIndex
from baloo.weld import create_placeholder_weld_object
from .indexes.utils import assert_indexes_equal


def assert_series_equal(actual, expected):
    actual = actual.evaluate()
    expected = expected.evaluate()

    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.dtype == expected.dtype
    # might seem redundant but testing the __len__ function
    assert len(actual) == len(expected)
    assert actual.name == expected.name
    assert_indexes_equal(actual.index, expected.index)


class TestSeries(object):
    def test_evaluate(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        actual = Series(data)
        expected = Series(data, RangeIndex(3))

        assert_series_equal(actual, expected)

    def test_len_raw(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        ind = Series(data)

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_len_lazy(self):
        data = np.array([1, 2, 3], dtype=np.int64)
        weld_obj = create_placeholder_weld_object(data)
        ind = Series(weld_obj, dtype=np.dtype(np.int64))

        actual = len(ind)
        expected = 3

        assert actual == expected
