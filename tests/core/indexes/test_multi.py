import numpy as np

from baloo import MultiIndex, Index
from .test_base import assert_index_equal


def assert_multiindex_equal(actual, expected, sort=False):
    actual = actual.evaluate()
    expected = expected.evaluate()

    assert actual._length == expected._length
    assert len(actual) == len(expected)
    assert actual.names == expected.names
    assert actual.dtypes == expected.dtypes
    for i in range(len(actual.values)):
        assert_index_equal(actual.values[i], expected.values[i], sort=sort)


class TestMultiIndex(object):
    def test_evaluate(self, data_f32, index_i64):
        actual = MultiIndex([data_f32, index_i64]).evaluate()
        expected = MultiIndex([Index(data_f32, np.dtype(np.float32)), index_i64])

        assert_multiindex_equal(actual, expected)

    def test_len_raw(self, data_f32, data_i64):
        ind = MultiIndex([data_f32, data_i64])

        actual = len(ind)
        expected = 5

        assert actual == expected

    def test_filter(self, data_f32, index_i64):
        mi = MultiIndex([data_f32, index_i64])

        actual = mi[Index(np.array([False, True, True, False, False]))]
        expected = MultiIndex([Index(np.array([2, 3], dtype=np.float32), np.dtype(np.float32)),
                               Index(np.array([1, 2], dtype=np.int64), np.dtype(np.int64))])

        assert_multiindex_equal(actual, expected)

    def test_slice(self, data_f32, index_i64):
        mi = MultiIndex([data_f32, index_i64])

        actual = mi[1:3]
        expected = MultiIndex([Index(np.array([2, 3], dtype=np.float32), np.dtype(np.float32)),
                               Index(np.array([1, 2], dtype=np.int64), np.dtype(np.int64))])

        assert_multiindex_equal(actual, expected)

    def test_dropna(self):
        mi = MultiIndex([[0, -999, 2, -999], Index([1., -999., -999., 3.], dtype=np.dtype(np.float64))])

        actual = mi.dropna()
        expected = MultiIndex([[0], [1.]])

        assert_multiindex_equal(actual, expected)
