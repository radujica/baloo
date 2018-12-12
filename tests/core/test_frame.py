from collections import OrderedDict

import numpy as np
import pytest

from baloo import DataFrame, Index, Series, MultiIndex
from .indexes.utils import assert_indexes_equal
from .test_series import assert_series_equal


def assert_dataframe_equal(actual, expected, almost=None, sort=False):
    actual = actual.evaluate()
    expected = expected.evaluate()

    assert actual._length == expected._length
    assert len(actual) == len(expected)
    assert_series_equal(actual.dtypes, expected.dtypes, sort=sort)
    assert_indexes_equal(actual.index, expected.index, sort=sort)
    assert_indexes_equal(actual.columns, expected.columns)
    assert len(actual._data) == len(expected._data)
    assert actual._data.keys() == expected._data.keys()
    for column_name in actual:
        assert_series_equal(actual._data[column_name], expected._data[column_name], almost=almost, sort=sort)


# TODO: fix |S11!!
class TestDataFrame(object):
    def test_init_list(self):
        data = [1, 2, 3]
        actual = DataFrame({'a': data})
        expected = DataFrame({'a': np.array(data)})

        assert_dataframe_equal(actual, expected)

    # just testing if they don't crash
    def test_repr_str(self, df_small):
        df = df_small.evaluate()
        repr(df)
        str(df)

    @pytest.mark.parametrize('size', [10, 1001])
    def test_evaluate(self, size, data_str):
        data_a = np.arange(size)
        data_b = np.random.choice(data_str, size)
        actual = DataFrame({'a': data_a, 'b': data_b})

        actual = actual.evaluate()
        assert len(actual) == size
        repr(actual)
        str(actual)

        expected = DataFrame({'a': data_a, 'b': data_b}, Index(np.arange(size)))

        assert_dataframe_equal(actual, expected)

    def test_len_lazy(self, series_i64):
        size = 5
        df = DataFrame({'a': series_i64})

        assert df._length is None
        assert len(df) == size
        assert df._length == size

    def test_getitem_str(self, df_small, data_f32, index_i64):
        actual = df_small['a']
        expected = Series(data_f32, index_i64, np.dtype(np.float32), 'a')

        assert_series_equal(actual, expected)

        with pytest.raises(KeyError):
            var = df_small['z']

    def test_getitem_list(self, df_small, data_f32, data_str):
        actual = df_small[['a', 'c']]
        expected = DataFrame(OrderedDict((('a', data_f32), ('c', data_str))))

        assert_dataframe_equal(actual, expected)

        with pytest.raises(TypeError):
            result = df_small[[1]]

        with pytest.raises(KeyError):
            result = df_small[['z']]

    def test_comparison(self, df_small, index_i64):
        actual = df_small < 3
        expected = DataFrame(OrderedDict((('a', np.array([True, True, False, False, False])),
                                          ('b', np.array([True, True, False, False, False])))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_filter(self, df_small):
        actual = df_small[Series(np.array([False, True, True, False, False]))]
        data = [2, 3]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)),
                                          ('c', np.array(['Abc', 'goosfraba'], dtype=np.dtype('|S11'))))),
                             Index(np.array([1, 2])))

        assert_dataframe_equal(actual, expected)

    def test_slice(self, df_small):
        actual = df_small[1:3]
        data = [2, 3]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)),
                                          ('c', np.array(['Abc', 'goosfraba'], dtype=np.dtype('|S11'))))),
                             Index(np.array([1, 2])))

        assert_dataframe_equal(actual, expected)

    def test_setitem_new_col(self, data_f32, series_i64, index_i64):
        actual = DataFrame(OrderedDict({'a': data_f32}))

        actual['b'] = series_i64
        expected = DataFrame(OrderedDict((('a', data_f32),
                                          ('b', series_i64))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_setitem_existing_col(self, data_f32, series_i64, data_str, index_i64):
        actual = DataFrame(OrderedDict((('a', data_f32),
                                        ('b', data_str))))

        actual['b'] = series_i64
        expected = DataFrame(OrderedDict((('a', data_f32),
                                          ('b', series_i64))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_setitem_alignment_needed(self, data_f32, index_i64):
        df = DataFrame({'a': data_f32}, index_i64)

        df['b'] = Series(np.array([4, 7, 5, 6, 8]), Index(np.array([0, 3, 1, 2, 5])))
        actual = df
        expected = DataFrame({'a': data_f32,
                              'b': np.array([4, 5, 6, 7, -999])},
                             index_i64)

        # so is WeldObject which was aligned
        assert not isinstance(df['b'].values, np.ndarray)

        assert_dataframe_equal(actual, expected)

    def test_setitem_alignment_not_needed(self, data_f32, data_i64, index_i64):
        df = DataFrame({'a': data_f32}, index_i64)

        df['b'] = Series(data_i64, index_i64)

        # so was directly added, no alignment
        assert isinstance(df['b'].values, np.ndarray)

    def test_head(self, df_small):
        actual = df_small.head(2)
        data = [1, 2]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)),
                                          ('c', np.array(['a', 'Abc'], dtype=np.dtype('|S11'))))),
                             Index(np.array([0, 1])))

        assert_dataframe_equal(actual, expected)

    def test_tail(self, df_small):
        actual = df_small.tail(2)
        data = [4, 5]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)),
                                          ('c', np.array(['   dC  ', 'secrETariat'], dtype=np.dtype('|S11'))))),
                             Index(np.array([3, 4])))

        assert_dataframe_equal(actual, expected)

    def test_iloc_indices(self, df_small):
        indices = Series(np.array([0, 2, 3]))

        actual = df_small.iloc[indices]
        data = [1, 3, 4]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)),
                                          ('c', np.array(['a', 'goosfraba', '   dC  '], dtype=np.dtype('|S11'))))),
                             Index(np.array([0, 2, 3])))

        assert_dataframe_equal(actual, expected)

    def test_keys(self, df_small, df_small_columns):
        assert_indexes_equal(df_small.keys(), df_small_columns)

    def test_op_array(self, df_small, index_i64, op_array_other):
        actual = df_small * [2, 3]
        expected = DataFrame(OrderedDict((('a', np.array([2, 4, 6, 8, 10], dtype=np.float32)),
                                          ('b', np.array([3, 6, 9, 12, 15])))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_op_scalar(self, df_small, index_i64):
        actual = df_small * 2
        data = [2, 4, 6, 8, 10]
        expected = DataFrame(OrderedDict((('a', np.array(data, dtype=np.float32)),
                                          ('b', np.array(data)))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize('aggregation, expected_data', [
        ('min', np.array([1., 1.])),
        ('max', np.array([5., 5.])),
        ('sum', np.array([15., 15.])),
        ('prod', np.array([120., 120.])),
        ('count', np.array([5., 5.])),
        ('var', np.array([2.5, 2.5])),
        ('std', np.array([1.581139, 1.581139]))
    ])
    def test_aggregations(self, aggregation, expected_data, df_small):
        actual = getattr(df_small, aggregation)()
        expected = Series(expected_data, Index(np.array(['a', 'b'], dtype=np.bytes_)))

        assert_series_equal(actual, expected, 5)

    def test_agg(self, df_small):
        aggregations = ['max', 'var', 'count', 'mean']

        actual = df_small.agg(aggregations)
        expected = DataFrame(OrderedDict((('a', np.array([5, 2.5, 5, 3], dtype=np.float64)),
                                          ('b', np.array([5, 2.5, 5, 3], dtype=np.float64)))),
                             Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_)))

        assert_dataframe_equal(actual, expected)

    def test_rename(self, df_small, data_f32, series_i64, data_str, index_i64):
        actual = df_small.rename({'a': 'd', 'd': 'nooo'})
        expected = DataFrame(OrderedDict((('d', data_f32),
                                          ('b', series_i64),
                                          ('c', data_str))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_drop_single(self, df_small, data_f32, series_i64, index_i64):
        actual = df_small.drop('c')
        expected = DataFrame(OrderedDict((('a', data_f32),
                                          ('b', series_i64))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_drop_multi(self, df_small, series_i64, index_i64):
        actual = df_small.drop(['a', 'c'])
        expected = DataFrame({'b': series_i64})

        assert_dataframe_equal(actual, expected)

    def test_reset_index(self, df_small, index_i64, data_f32, series_i64, data_str):
        actual = df_small.reset_index()
        expected = DataFrame(OrderedDict((('index', np.arange(5)),
                                          ('a', data_f32),
                                          ('b', series_i64),
                                          ('c', data_str))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize('names,expected_names', [
        (None, ['level_0', 'level_1']),
        (['i1', 'i2'], ['i1', 'i2'])
    ])
    def test_reset_multi_index(self, data_f32, data_i64, data_str, index_i64, names, expected_names):
        df = DataFrame(OrderedDict({'c': data_str}),
                       MultiIndex([data_f32, data_i64], names=names))

        actual = df.reset_index()
        expected = DataFrame(OrderedDict(((expected_names[0], data_f32),
                                          (expected_names[1], data_i64),
                                          ('c', data_str))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_set_index(self, df_small, data_f32, data_i64, data_str):
        actual = df_small.set_index('b')
        expected = DataFrame(OrderedDict((('a', data_f32),
                                          ('c', data_str))),
                             Index(data_i64, np.dtype(np.int64), 'b'))

        assert_dataframe_equal(actual, expected)

    def test_set_multi_index(self, df_small, data_f32, data_i64, data_str):
        actual = df_small.set_index(['b', 'c'])
        expected = DataFrame({'a': data_f32},
                             MultiIndex([Index(data_i64, np.dtype(np.int64), 'b'),
                                         Index(data_str, name='c')],
                                        ['b', 'c']))

        assert_dataframe_equal(actual, expected)

    def test_sort_index_index_ascending(self, data_f32, series_i64):
        data = [data_f32, series_i64]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(np.array([3, 1, 2, 5, 4])))

        actual = df.sort_index()

        expected_index = Index(np.arange(1, 6), np.dtype(np.int64))
        expected_data = [Series(np.array([2, 3, 1, 5, 4], dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([2, 3, 1, 5, 4], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)

    def test_sort_index_index_descending(self, data_f32, series_i64):
        data = [data_f32, series_i64]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(np.array([3, 1, 2, 5, 4])))

        actual = df.sort_index(ascending=False)

        expected_index = Index(np.arange(5, 0, -1), np.dtype(np.int64))
        expected_data = [Series(np.array([4, 5, 1, 3, 2], dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([4, 5, 1, 3, 2], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)

    # TODO: uncomment & refactor when implemented
    # def test_sort_index_multi_index_ascending(self):
    #     data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
    #     df = DataFrame(OrderedDict((('a', data[0]),
    #                                 ('b', data[1]))),
    #                    MultiIndex([np.array([2, 3, 3, 1, 1]), np.array([2, 2, 1, 0, 3], dtype=np.float64)]))
    #
    #     actual = df.sort_index()
    #     expected_index = MultiIndex([np.array([1, 1, 2, 3, 3]), np.array([0, 3, 2, 1, 2], dtype=np.float64)])
    #     expected_data = [Series(np.array([4, 5, 1, 3, 2], dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
    #                      Series(np.array([3, 4, 0, 2, 1], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
    #     expected = DataFrame(OrderedDict((('a', expected_data[0]),
    #                                       ('b', expected_data[1]))),
    #                          expected_index)
    #
    #     assert_dataframe_equal(actual, expected)

    def test_sort_values(self, data_f32, series_i64, data_i64):
        data = [np.array([3, 1, 2, 5, 4], dtype=np.float32), series_i64]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(data_i64))

        actual = df.sort_values('a')

        expected_index = Index(np.array([2, 3, 1, 5, 4]), np.dtype(np.int64))
        expected_data = [Series(data_f32, expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([2, 3, 1, 5, 4], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)

    def test_drop_duplicates_all(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]))),
                       index_i64)

        actual = df.drop_duplicates()
        expected = DataFrame(OrderedDict((('a', np.array([0, 1, 2, 3], dtype=np.float32)),
                                          ('b', [4, 5, 6, 6]))),
                             Index([0, 1, 3, 4]))

        assert_dataframe_equal(actual, expected, sort=True)

    def test_drop_duplicates_subset(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                    ('b', [4, 5, 5, 6, 6]))),
                       index_i64)

        actual = df.drop_duplicates(subset=['b'])
        expected = DataFrame(OrderedDict((('a', np.array([0, 1, 2], dtype=np.float32)),
                                          ('b', [4, 5, 6]))),
                             Index([0, 1, 3]))

        assert_dataframe_equal(actual, expected, sort=True)

    # TODO: update the following tests with fixture (also series and base)
    def test_isna(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, -999, 2, -999], dtype=np.float32)),
                                    ('b', [4, -999, -999, 6, 6]))),
                       index_i64)

        actual = df.isna()
        expected = DataFrame(OrderedDict((('a', np.array([False, False, True, False, True])),
                                          ('b', [False, True, True, False, False]))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_dropna(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, -999, 2, -999], dtype=np.float32)),
                                    ('b', [4, -999, -999, 6, 6]))),
                       index_i64)

        actual = df.dropna()
        expected = DataFrame(OrderedDict((('a', np.array([0, 2], dtype=np.float32)),
                                          ('b', [4, 6]))),
                             Index([0, 3]))

        assert_dataframe_equal(actual, expected)

    def test_dropna_subset(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, -999, 2, -999], dtype=np.float32)),
                                    ('b', [4, -999, -999, 6, 6]))),
                       index_i64)

        actual = df.dropna(subset=['a'])
        expected = DataFrame(OrderedDict((('a', np.array([0, 1, 2], dtype=np.float32)),
                                          ('b', [4, -999, 6]))),
                             Index([0, 1, 3]))

        assert_dataframe_equal(actual, expected)

    def test_fillna_scalar(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, -999, 2, -999], dtype=np.float32)),
                                    ('b', [4, -999, -999, 6, 6]))),
                       index_i64)

        actual = df.fillna(15)
        expected = DataFrame(OrderedDict((('a', np.array([0, 1, 15, 2, 15], dtype=np.float32)),
                                          ('b', [4, 15, 15, 6, 6]))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_fillna_dict(self, index_i64):
        df = DataFrame(OrderedDict((('a', np.array([0, 1, -999, 2, -999], dtype=np.float32)),
                                    ('b', [4, -999, -999, 6, 6]))),
                       index_i64)

        actual = df.fillna({'a': 15})
        expected = DataFrame(OrderedDict((('a', np.array([0, 1, 15, 2, 15], dtype=np.float32)),
                                          ('b', [4, -999, -999, 6, 6]))),
                             index_i64)

        assert_dataframe_equal(actual, expected)

    def test_astype_dtype(self, df1, data_i64, index_i64_2):
        actual = df1.astype(np.dtype(np.int64))
        expected = DataFrame(OrderedDict((('a', np.arange(5)),
                                          ('b', data_i64))),
                             index_i64_2)

        assert_dataframe_equal(actual, expected)

    def test_astype_dict(self, df1, data_f32, index_i64_2):
        actual = df1.astype({'a': np.dtype(np.float32)})
        expected = DataFrame(OrderedDict((('a', np.arange(5, dtype=np.float32)),
                                          ('b', data_f32))),
                             index_i64_2)

        assert_dataframe_equal(actual, expected)
