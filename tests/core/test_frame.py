from collections import OrderedDict

import numpy as np
import pytest

from baloo import DataFrame, Index, Series, RangeIndex, MultiIndex
from baloo.weld import create_placeholder_weld_object
from .indexes.utils import assert_indexes_equal
from .test_series import assert_series_equal


def assert_dataframe_equal(actual, expected):
    # assert before evaluating
    assert_indexes_equal(actual.index, expected.index)

    actual = actual.evaluate()
    expected = expected.evaluate()

    assert actual._length == expected._length
    assert len(actual) == len(expected)
    assert_series_equal(actual.dtypes, expected.dtypes)
    assert_indexes_equal(actual.index, expected.index)
    assert len(actual.data) == len(expected.data)
    assert actual.data.keys() == expected.data.keys()
    for column_name in actual:
        np.testing.assert_array_equal(actual[column_name].values, expected[column_name].values)


class TestDataFrame(object):
    def test_empty(self):
        df = DataFrame({})

        assert len(df) == 0
        # just testing if they don't crash
        repr(df)
        str(df)

    @pytest.mark.parametrize('size', [10, 1001])
    def test_evaluate(self, size):
        data_a = np.arange(size)
        data_b = np.random.choice(np.array(['abc', 'goosfraba', '   df   '], dtype=np.bytes_), size)
        actual = DataFrame({'a': data_a, 'b': data_b})

        assert len(actual) == size
        actual = actual.evaluate()
        assert len(actual) == size
        repr(actual)
        str(actual)

        expected = DataFrame({'a': data_a, 'b': data_b}, Index(np.arange(size)))

        assert_dataframe_equal(actual, expected)

    def test_len_lazy(self):
        size = 20
        data = np.arange(size)
        weld_obj = create_placeholder_weld_object(data)
        sr = Series(weld_obj, dtype=np.dtype(np.int64))
        df = DataFrame({'a': sr})

        assert df._length is None
        assert len(df) == size
        assert df._length == size

    def test_getitem_str(self):
        size = 10
        df = DataFrame({'a': np.arange(size)})

        actual = df['a']
        expected = Series(np.arange(size), RangeIndex(size), np.dtype(np.int64), 'a')

        assert_series_equal(actual, expected)

        with pytest.raises(KeyError):
            column = df['b']

    def test_getitem_list(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5)), np.arange(5)]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]), ('c', data[2]))))

        actual = df[['a', 'c']]
        expected = DataFrame(OrderedDict((('a', data[0]), ('c', data[2]))))

        assert_dataframe_equal(actual, expected)

        with pytest.raises(TypeError):
            result = df[[1]]

        with pytest.raises(KeyError):
            result = df[['d']]

    def test_comparison(self):
        df = DataFrame({'a': np.arange(0, 4),
                        'b': np.arange(4, 8)})

        actual = df < 3
        expected = DataFrame({'a': np.array([True, True, True, False]),
                              'b': np.array([False, False, False, False])},
                             RangeIndex(4))

        assert_dataframe_equal(actual, expected)

    def test_filter(self):
        df = DataFrame({'a': np.arange(0, 4),
                        'b': np.arange(4, 8)})

        actual = df[Series(np.array([False, True, True, False]))]
        expected = DataFrame({'a': np.array([1, 2]),
                              'b': np.array([5, 6])},
                             Index(np.array([1, 2])))

        assert_dataframe_equal(actual, expected)

    def test_slice(self):
        df = DataFrame({'a': np.arange(0, 4),
                        'b': np.arange(4, 8)})

        actual = df[1:3]
        expected = DataFrame({'a': np.array([1, 2]),
                              'b': np.array([5, 6])},
                             Index(np.array([1, 2])))

        assert_dataframe_equal(actual, expected)

    def test_setitem_new_col(self):
        df = DataFrame({'a': np.arange(0, 4)})

        df['b'] = np.arange(4, 8)
        actual = df
        expected = DataFrame({'a': np.arange(0, 4),
                              'b': np.arange(4, 8)})

        assert_dataframe_equal(actual, expected)

    def test_setitem_existing_col(self):
        df = DataFrame({'a': np.arange(0, 4),
                        'b': np.arange(8, 12)})

        df['b'] = np.arange(4, 8)
        actual = df
        expected = DataFrame({'a': np.arange(0, 4),
                              'b': np.arange(4, 8)})

        assert_dataframe_equal(actual, expected)

    def test_head(self):
        df = DataFrame({'a': np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        'b': Series(np.arange(5))})

        actual = df.head(2)
        expected = DataFrame({'a': np.array([1, 2], dtype=np.float32),
                              'b': np.array([0, 1])},
                             Index(np.array([0, 1])))

        assert_dataframe_equal(actual, expected)

    def test_tail(self):
        df = DataFrame({'a': np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        'b': Series(np.arange(5))})

        actual = df.tail(2)
        expected = DataFrame({'a': np.array([4, 5], dtype=np.float32),
                              'b': np.array([3, 4])},
                             Index(np.array([3, 4])))

        assert_dataframe_equal(actual, expected)

    def test_iloc_indices(self):
        df = DataFrame(OrderedDict((('a', np.array([1, 2, 3, 4, 5])),
                                    ('b', Series(np.arange(5))))))
        indices = Series(np.array([0, 2, 3]))

        actual = df.iloc[indices]
        expected = DataFrame(OrderedDict((('a', np.array([1, 3, 4])),
                                          ('b', Series(np.array([0, 2, 3]))))),
                             Index(np.array([0, 2, 3])))

        assert_dataframe_equal(actual, expected)

    def test_keys(self):
        df = DataFrame(OrderedDict((('a', np.array([1, 2, 3, 4, 5], dtype=np.float32)),
                                    ('b', Series(np.arange(5))))))

        actual = df.keys()
        expected = Index(np.array(['a', 'b'], dtype=np.bytes_))

        assert_indexes_equal(actual, expected)

    def test_op_array(self):
        df = DataFrame({'a': np.array([1, 2, 3, 4, 5]),
                        'b': Series(np.arange(5))})

        actual = df * Series(np.array([2] * 5))
        expected = DataFrame({'a': np.array([2, 4, 6, 8, 10]),
                              'b': Series(np.arange(0, 10, 2))})

        assert_dataframe_equal(actual, expected)

    def test_op_scalar(self):
        df = DataFrame({'a': np.array([1, 2, 3, 4, 5], dtype=np.float32),
                        'b': Series(np.arange(5))})

        actual = df * 2
        expected = DataFrame({'a': np.array([2, 4, 6, 8, 10], dtype=np.float32),
                              'b': Series(np.arange(0, 10, 2))})

        assert_dataframe_equal(actual, expected)

    @pytest.mark.parametrize('aggregation, expected', [
        ('min', Series(np.array([1, 2]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('max', Series(np.array([5, 6]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('sum', Series(np.array([15, 20]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('prod', Series(np.array([120, 720]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('count', Series(np.array([5, 5]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('var', Series(np.array([2.5, 2.5]), Index(np.array(['a', 'b'], dtype=np.bytes_)))),
        ('std', Series(np.array([1.581139, 1.581139]), Index(np.array(['a', 'b'], dtype=np.bytes_))))
    ])
    def test_aggregations(self, aggregation, expected):
        data = OrderedDict([('a', np.arange(1, 6)), ('b', np.arange(2, 7))])
        df = DataFrame(data)

        actual = getattr(df, aggregation)()

        assert_series_equal(actual, expected, 5)

    def test_aggregation_diff_types(self):
        data = OrderedDict([('a', np.arange(1, 6)), ('b', np.arange(2, 7, dtype=np.float32))])
        df = DataFrame(data)

        actual = df.min()
        expected = Series(np.array([1, 2], dtype=np.float64), Index(np.array(['a', 'b'], dtype=np.bytes_)))

        assert_series_equal(actual, expected)

    def test_agg(self):
        aggregations = ['max', 'var', 'count', 'mean']
        df = DataFrame(OrderedDict((('a', np.arange(1, 6, dtype=np.float32)),
                                    ('b', Series(np.arange(5))))))

        actual = df.agg(aggregations)

        expected = DataFrame(OrderedDict((('a', np.array([5, 2.5, 5, 3], dtype=np.float64)),
                                          ('b', np.array([4, 2.5, 5, 2], dtype=np.float64)))),
                             Index(np.array(aggregations, dtype=np.bytes_), np.dtype(np.bytes_)))

        assert_dataframe_equal(actual, expected)

    def test_rename(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]))))

        actual = df.rename({'a': 'c', 'd': 'nooo'})
        expected = DataFrame(OrderedDict((('c', data[0]), ('b', data[1]))))

        assert_dataframe_equal(actual, expected)

    def test_drop_single(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]))))

        actual = df.drop('b')
        expected = DataFrame({'a': data[0]})

        assert_dataframe_equal(actual, expected)

    def test_drop_multi(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5)), np.arange(5)]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]),
                                    ('c', data[2]))))

        actual = df.drop(['a', 'c'])
        expected = DataFrame({'b': data[1]})

        assert_dataframe_equal(actual, expected)

    def test_reset_index(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]))))

        actual = df.reset_index()
        expected = DataFrame(OrderedDict((('index', np.arange(5)),
                                          ('a', data[0]),
                                          ('b', data[1]))),
                             RangeIndex(5))

        assert_dataframe_equal(actual, expected)

    def test_reset_multi_index_unnamed(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        index_data = [np.arange(5, 10), np.arange(10, 15)]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]))),
                       MultiIndex(index_data))

        actual = df.reset_index()
        expected = DataFrame(OrderedDict((('level_0', index_data[0]),
                                          ('level_1', index_data[1]),
                                          ('a', data[0]),
                                          ('b', data[1]))),
                             RangeIndex(5))

        assert_dataframe_equal(actual, expected)

    def test_reset_multi_index_named(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        index_data = [np.arange(5, 10), np.arange(10, 15)]
        df = DataFrame(OrderedDict((('a', data[0]), ('b', data[1]))),
                       MultiIndex(index_data, ['i1', 'i2']))

        actual = df.reset_index()
        expected = DataFrame(OrderedDict((('i1', index_data[0]),
                                          ('i2', index_data[1]),
                                          ('a', data[0]),
                                          ('b', data[1]))),
                             RangeIndex(5))

        assert_dataframe_equal(actual, expected)

    def test_set_index(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5)), np.arange(5)]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]),
                                    ('c', data[2]))))

        actual = df.set_index('b')
        expected = DataFrame(OrderedDict((('a', data[0]),
                                          ('c', data[2]))),
                             Index(data[1].values, np.dtype(np.int64), 'b'))

        assert_dataframe_equal(actual, expected)

    def test_set_multi_index(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5)), np.arange(5)]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]),
                                    ('c', data[2]))))

        actual = df.set_index(['b', 'c'])
        expected = DataFrame({'a': data[0]},
                             MultiIndex([Index(data[1].values, np.dtype(np.int64), 'b'),
                                         Index(data[2], np.dtype(np.int64), 'c')],
                                        ['b', 'c']))

        assert_dataframe_equal(actual, expected)

    def test_sort_index_index_ascending(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(np.array([3, 1, 2, 5, 4])))

        actual = df.sort_index()

        expected_index = Index(np.arange(1, 6), np.dtype(np.int64), 'index')
        expected_data = [Series(np.array([2, 3, 1, 5, 4], dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([1, 2, 0, 4, 3], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)

    def test_sort_index_index_descending(self):
        data = [np.array([1, 2, 3, 4, 5], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(np.array([3, 1, 2, 5, 4])))

        actual = df.sort_index(ascending=False)

        expected_index = Index(np.arange(5, 0, -1), np.dtype(np.int64), 'index')
        expected_data = [Series(np.array([4, 5, 1, 3, 2], dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([3, 4, 0, 2, 1], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)

    # TODO: uncomment when implemented
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

    def test_sort_values(self):
        data = [np.array([3, 1, 2, 5, 4], dtype=np.float32), Series(np.arange(5))]
        df = DataFrame(OrderedDict((('a', data[0]),
                                    ('b', data[1]))),
                       Index(np.arange(1, 6)))

        actual = df.sort_values('a')

        expected_index = Index(np.array([2, 3, 1, 5, 4]), np.dtype(np.int64), 'index')
        expected_data = [Series(np.array(np.arange(1, 6), dtype=np.float32), expected_index, np.dtype(np.float32), 'a'),
                         Series(np.array([1, 2, 0, 4, 3], dtype=np.int64), expected_index, np.dtype(np.int64), 'b')]
        expected = DataFrame(OrderedDict((('a', expected_data[0]),
                                          ('b', expected_data[1]))),
                             expected_index)

        assert_dataframe_equal(actual, expected)
