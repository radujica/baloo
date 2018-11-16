from collections import OrderedDict

import numpy as np

from baloo import DataFrame, Index, MultiIndex
from .test_frame import assert_dataframe_equal


class TestGroupBy(object):
    # TODO: this input data is almost same as drop_duplicates
    def test_groupby_single(self, index_i64, series_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]),
                                    ('c', series_i64))),
                       index_i64)

        actual = df.groupby('b').sum()
        expected = DataFrame(OrderedDict((('a', np.array([0, 2, 5], dtype=np.float32)),
                                          ('c', [1, 5, 9]))),
                             Index(np.array([4, 5, 6]), np.dtype(np.int64), 'b'))

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_single_mean(self, index_i64, series_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]),
                                    ('c', series_i64))),
                       index_i64)

        actual = df.groupby('b').mean()
        expected = DataFrame(OrderedDict((('a', [0., 1., 2.5]),
                                          ('c', [1., 2.5, 4.5]))),
                             Index(np.array([4, 5, 6]), np.dtype(np.int64), 'b'))

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_single_size(self, index_i64, series_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]),
                                    ('c', series_i64))),
                       index_i64)

        actual = df.groupby('b').size()
        expected = DataFrame(OrderedDict((('a', [1, 2, 2]),
                                          ('c', [1, 2, 2]))),
                             Index(np.array([4, 5, 6]), np.dtype(np.int64), 'b'))

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_multi(self, index_i64, series_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]),
                                    ('c', series_i64))),
                       index_i64)

        actual = df.groupby(['a', 'b']).min()
        expected = DataFrame(OrderedDict({'c': [1, 2, 4, 5]}),
                             MultiIndex([Index(np.array([0, 1, 2, 3], dtype=np.float32), np.dtype(np.float32), 'a'),
                                         Index([4, 5, 6, 6], np.dtype(np.int64), 'b')], ['a', 'b']))

        assert_dataframe_equal(actual, expected, sort=True)
