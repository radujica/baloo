from collections import OrderedDict

import numpy as np

from baloo import DataFrame, Index, MultiIndex
from .test_frame import assert_dataframe_equal


class TestGroupBy(object):
    def test_groupby_single(self, index_i64, series_i64, df_dupl, df_dupl_exp_ind):
        actual = df_dupl.groupby('b').sum()
        expected = DataFrame(OrderedDict((('a', np.array([0, 2, 5], dtype=np.float32)),
                                          ('c', [1, 5, 9]))),
                             df_dupl_exp_ind)

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_single_mean(self, index_i64, series_i64, df_dupl, df_dupl_exp_ind):
        actual = df_dupl.groupby('b').mean()
        expected = DataFrame(OrderedDict((('a', [0., 1., 2.5]),
                                          ('c', [1., 2.5, 4.5]))),
                             df_dupl_exp_ind)

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_single_var(self, index_i64, series_i64, df_dupl, df_dupl_exp_ind):
        actual = df_dupl.groupby('b').var()
        expected = DataFrame(OrderedDict((('a', [0., 0., 0.5]),
                                          ('c', [0., 0.5, 0.5]))),
                             df_dupl_exp_ind)

        assert_dataframe_equal(actual, expected, almost=5, sort=True)

    def test_groupby_single_std(self, index_i64, series_i64, df_dupl, df_dupl_exp_ind):
        actual = df_dupl.groupby('b').std()
        expected = DataFrame(OrderedDict((('a', [0., 0., 0.707107]),
                                          ('c', [0., 0.70711, 0.70711]))),
                             df_dupl_exp_ind)

        assert_dataframe_equal(actual, expected, almost=5, sort=True)

    def test_groupby_single_size(self, index_i64, series_i64, df_dupl, df_dupl_exp_ind):
        actual = df_dupl.groupby('b').size()
        expected = DataFrame(OrderedDict((('a', [1, 2, 2]),
                                          ('c', [1, 2, 2]))),
                             df_dupl_exp_ind)

        assert_dataframe_equal(actual, expected, sort=True)

    def test_groupby_multi(self, index_i64, series_i64, df_dupl):
        actual = df_dupl.groupby(['a', 'b']).min()
        expected = DataFrame(OrderedDict({'c': [1, 2, 4, 5]}),
                             MultiIndex([Index(np.array([0, 1, 2, 3], dtype=np.float32), np.dtype(np.float32), 'a'),
                                         Index([4, 5, 6, 6], np.dtype(np.int64), 'b')], ['a', 'b']))

        assert_dataframe_equal(actual, expected, sort=True)
