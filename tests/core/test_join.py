from collections import OrderedDict

import numpy as np

from baloo import DataFrame, Index, MultiIndex, Series
from .test_frame import assert_dataframe_equal


# TODO: fix |S4!!
class TestJoins(object):
    def test_merge_sorted_unique_single_on_inner(self, df1, df2):
        actual = df1.merge(df2, on='a')
        expected = DataFrame(OrderedDict((('index', np.array([3, 5])),
                                          ('b_x', np.array([2, 4], dtype=np.float32)),
                                          ('d', np.array(['abc', 'def'], dtype=np.dtype('|S4'))),
                                          ('b_y', Series(np.arange(3, 5, dtype=np.float32))),
                                          ('c', np.arange(4, 6)))),
                             Index(np.array([1, 3]), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_multi_on_inner(self, df1, df2):
        actual = df1.merge(df2, on=['a', 'b'], is_on_sorted=True)
        expected = DataFrame(OrderedDict((('index', np.array([5])),
                                          ('d', np.array(['def'], dtype=np.dtype('|S4'))),
                                          ('c', np.array([5])))),
                             MultiIndex([np.array([3]), np.array([4], dtype=np.float32)], ['a', 'b']))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_single_on_left(self, df1, df2):
        actual = df1.merge(df2, on='a', how='left')
        expected = DataFrame(OrderedDict((('index', np.arange(2, 7)),
                                          ('b_x', np.arange(1, 6, dtype=np.float32)),
                                          ('d', np.array(['None', 'abc', 'None', 'def', 'None'], dtype=np.bytes_)),
                                          ('b_y', Series(np.array([-999., 3., -999., 4., -999.], dtype=np.float32))),
                                          ('c', np.array([-999, 4, -999, 5, -999])))),
                             Index(np.arange(5), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_single_on_right(self, df1, df2):
        actual = df1.merge(df2, on='a', how='right')
        expected = DataFrame(OrderedDict((('index', np.array([3, 5, -999])),
                                          ('b_x', np.array([2, 4, -999.], dtype=np.float32)),
                                          ('d', np.array(['abc', 'def', 'efgh'], dtype=np.dtype('|S4'))),
                                          ('b_y', Series(np.array([3., 4., 5.], dtype=np.float32))),
                                          ('c', np.array([4, 5, 6])))),
                             Index(np.array([1, 3, 5]), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_single_on_outer(self, df1, df2):
        actual = df1.merge(df2, on='a', how='outer')
        expected = DataFrame(OrderedDict((('index', np.array([2, 3, 4, 5, 6, -999])),
                                          ('b_x', np.array([1, 2, 3, 4, 5, -999.], dtype=np.float32)),
                                          ('d', np.array(['None', 'abc', 'None', 'def', 'None', 'efgh'], dtype=np.dtype('|S4'))),
                                          ('b_y', Series(np.array([-999., 3., -999., 4., -999., 5.], dtype=np.float32))),
                                          ('c', np.array([-999, 4, -999, 5, -999, 6])))),
                             Index(np.arange(0, 6), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    # seems unnecessary to run for all cases since join just delegates to merge
    def test_join(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(0, 5)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                        Index(np.array(np.array([1, 3, 5]))))

        actual = df1.join(df2, lsuffix='_x')
        expected = DataFrame(OrderedDict((('a', np.arange(5)),
                                          ('b_x', np.arange(1, 6, dtype=np.float64)),
                                          ('b', Series(np.array([-999., 3., -999., 4., -999.], dtype=np.float32))),
                                          ('c', np.array([-999, 4, -999, 5, -999])))),
                             Index(np.arange(0, 5), np.dtype(np.int64), 'index'))

        assert_dataframe_equal(actual, expected)

    def test_merge_unsorted_unique_single_on_inner(self, df2, data_f32, index_i64_2):
        df1 = DataFrame(OrderedDict((('a', Series(np.array([3, 2, 0, 4, 1]))),
                                     ('b', np.array([4, 3, 1, 5, 2], dtype=np.float32)))),
                        Index([5, 4, 2, 6, 3]))

        actual = df1.merge(df2, on='a')
        expected = DataFrame(OrderedDict((('index', np.array([3, 5])),
                                          ('b_x', np.array([2, 4], dtype=np.float32)),
                                          ('d', np.array(['abc', 'def'], dtype=np.dtype('|S4'))),
                                          ('b_y', Series(np.arange(3, 5, dtype=np.float32))),
                                          ('c', np.arange(4, 6)))),
                             Index(np.array([1, 3]), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)
