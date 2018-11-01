from collections import OrderedDict

import numpy as np

from baloo import DataFrame, Index, MultiIndex, Series
from .test_frame import assert_dataframe_equal


# TODO: fixtures for df1/df2; also add other types; make sure they can be used for all tests; parametrize
class TestJoins(object):
    def test_merge_sorted_unique_single_on_inner(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(2, 7)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                        MultiIndex([np.array([1, 3, 5]), np.arange(5, 8)], ['a', 'd']))

        actual = df1.merge(df2, on='a')
        expected = DataFrame(OrderedDict((('index', np.array([3, 5])),
                                          ('b_x', np.array([2, 4], dtype=np.float64)),
                                          ('d', np.arange(5, 7)),
                                          ('b_y', Series(np.arange(3, 5, dtype=np.float32))),
                                          ('c', np.arange(4, 6)))),
                             Index(np.array([1, 3]), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_multi_on_inner(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(2, 7)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float64)), ('c', np.arange(4, 7)))),
                        MultiIndex([np.array([1, 3, 5]), np.arange(5, 8)], ['a', 'd']))

        actual = df1.merge(df2, on=['a', 'b'])
        expected = DataFrame(OrderedDict((('index', np.array([5])),
                                          ('d', np.array([6])),
                                          ('c', np.array([5])))),
                             MultiIndex([np.array([3]), np.array([4], dtype=np.float64)], ['a', 'b']))

        assert_dataframe_equal(actual, expected)

    # TODO: use this for generalization ~ contains str column too
    def test_merge_sorted_unique_single_on_left(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(2, 7)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                        MultiIndex([np.array([1, 3, 5]),
                                    np.array(['abc', 'def', 'efgh'], dtype=np.bytes_)],
                                   ['a', 'd']))

        actual = df1.merge(df2, on='a', how='left')
        expected = DataFrame(OrderedDict((('index', np.arange(2, 7)),
                                          ('b_x', np.arange(1, 6, dtype=np.float64)),
                                          ('d', np.array(['None', 'abc', 'None', 'def', 'None'], dtype=np.bytes_)),
                                          ('b_y', Series(np.array([-999., 3., -999., 4., -999.], dtype=np.float32))),
                                          ('c', np.array([-999, 4, -999, 5, -999])))),
                             Index(np.arange(0, 5), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_single_on_right(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(2, 7)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                        MultiIndex([np.array([1, 3, 5]), np.arange(5, 8)], ['a', 'd']))

        actual = df1.merge(df2, on='a', how='right')
        expected = DataFrame(OrderedDict((('index', np.array([3, 5, -999])),
                                          ('b_x', np.array([2, 4, -999.], dtype=np.float64)),
                                          ('d', np.array([5, 6, 7])),
                                          ('b_y', Series(np.array([3., 4., 5.], dtype=np.float32))),
                                          ('c', np.array([4, 5, 6])))),
                             Index(np.array([1, 3, 5]), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)

    def test_merge_sorted_unique_single_on_outer(self):
        df1 = DataFrame(OrderedDict((('a', np.arange(5)), ('b', np.arange(1, 6, dtype=np.float64)))),
                        Index(np.arange(2, 7)))
        df2 = DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                        MultiIndex([np.array([1, 3, 5]), np.arange(5, 8)], ['a', 'd']))

        actual = df1.merge(df2, on='a', how='outer')
        expected = DataFrame(OrderedDict((('index', np.array([2, 3, 4, 5, 6, -999])),
                                          ('b_x', np.array([1, 2, 3, 4, 5, -999.], dtype=np.float64)),
                                          ('d', np.array([-999, 5, -999, 6, -999, 7])),
                                          ('b_y', Series(np.array([-999., 3., -999., 4., -999., 5.], dtype=np.float32))),
                                          ('c', np.array([-999, 4, -999, 5, -999, 6])))),
                             Index(np.arange(0, 6), np.dtype(np.int64), 'a'))

        assert_dataframe_equal(actual, expected)
