import numpy as np
import pytest

from baloo import DataFrame, Index, Series, RangeIndex
from baloo.weld import create_placeholder_weld_object
from .test_series import assert_series_equal
from .indexes.utils import assert_indexes_equal


def assert_dataframe_equal(actual, expected):
    # assert before evaluating
    assert_indexes_equal(actual.index, expected.index)

    actual = actual.evaluate()
    expected = expected.evaluate()

    assert actual._length == expected._length
    assert len(actual) == len(expected)
    assert actual._dtypes == expected._dtypes
    assert_indexes_equal(actual.index, expected.index)
    assert len(actual.data) == len(expected.data)
    assert actual.data.keys() == expected.data.keys()
    for column_name in actual:
        np.testing.assert_array_equal(actual[column_name].data, expected[column_name].data)


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

        np.testing.assert_array_equal(df['a'].data, df.data['a'])
        actual = df['a']
        expected = Series(np.arange(size), RangeIndex(size), np.dtype(np.int64), 'a')

        assert_series_equal(actual, expected)

        with pytest.raises(KeyError):
            column = df['b']
