import numpy as np

from baloo import MultiIndex, Index
from .test_base import assert_index_equal


def assert_multiindex_equal(actual, expected):
    actual = actual.evaluate()
    expected = expected.evaluate()

    assert len(actual) == len(expected)
    assert actual.names == expected.names
    assert actual.dtypes == expected.dtypes
    for i in range(len(actual.values)):
        assert_index_equal(actual.values[i], expected.values[i])


class TestMultiIndex(object):
    def test_evaluate(self):
        col1 = Index(np.array([1, 2, 3], dtype=np.int64), np.dtype(np.int64), 'i1')
        data = [col1, np.array([4, 5, 6], dtype=np.float64)]
        actual = MultiIndex(data).evaluate()
        expected = MultiIndex([col1, Index(data[1], np.dtype(np.float64))])

        assert_multiindex_equal(actual, expected)

    def test_len_raw(self):
        ind = MultiIndex([np.array([1, 2, 3], dtype=np.int64),
                          np.array([4, 5, 6], dtype=np.float64)])

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_filter(self):
        ind1 = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        ind2 = Index(np.array([6, 7, 8, 9, 10], dtype=np.int32))
        mi = MultiIndex([ind1, ind2])

        actual = mi[Index(np.array([False, True, True, False, False]))]
        expected = MultiIndex([Index(np.array([2, 3], dtype=np.float32), np.dtype(np.float32)),
                               Index(np.array([7, 8], dtype=np.int32), np.dtype(np.int32))])

        assert_multiindex_equal(actual, expected)

    def test_slice(self):
        ind1 = Index(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        ind2 = Index(np.array([6, 7, 8, 9, 10], dtype=np.int32))
        mi = MultiIndex([ind1, ind2])

        actual = mi[1:3]
        expected = MultiIndex([Index(np.array([2, 3], dtype=np.float32), np.dtype(np.float32)),
                               Index(np.array([7, 8], dtype=np.int32), np.dtype(np.int32))])

        assert_multiindex_equal(actual, expected)
