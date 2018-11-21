import pandas as pd

from baloo import Index, MultiIndex, Series, DataFrame
from .indexes.test_base import assert_index_equal
from .indexes.test_multi import assert_multiindex_equal
from .test_frame import assert_dataframe_equal
from .test_series import assert_series_equal


class TestPandasConversions(object):
    def test_from_pandas_index(self, index_i64):
        pandas_index = pd.Index([0, 1, 2, 3, 4])

        actual = Index.from_pandas(pandas_index)
        expected = index_i64

        assert_index_equal(actual, expected)

    def test_from_pandas_multiindex(self):
        pandas_index = pd.MultiIndex.from_product([[0, 1], [2., 3.]])

        actual = MultiIndex.from_pandas(pandas_index)
        expected = MultiIndex([[0, 0, 1, 1], [2., 3., 2., 3.]])

        assert_multiindex_equal(actual, expected)

    def test_from_pandas_series(self, data_i64, series_i64):
        pandas_series = pd.Series(data_i64)

        actual = Series.from_pandas(pandas_series)
        expected = series_i64

        assert_series_equal(actual, expected)

    def test_from_pandas_df(self, data_f32, df1):
        pandas_df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': data_f32}, pd.Index([2, 3, 4, 5, 6]))

        actual = DataFrame.from_pandas(pandas_df)
        expected = df1

        assert_dataframe_equal(actual, expected)

    def test_to_pandas_index(self, index_i64):
        actual = index_i64.to_pandas()
        expected = pd.Index([0, 1, 2, 3, 4])

        assert actual.equals(expected)

    def test_to_pandas_multiindex(self):
        data = [[0, 0, 1, 1], [2., 3., 2., 3.]]
        mi = MultiIndex(data)

        actual = mi.to_pandas()
        expected = pd.MultiIndex.from_arrays(data)

        assert actual.equals(expected)

    def test_to_pandas_series(self, data_f32, series_f32):
        actual = series_f32.to_pandas()
        expected = pd.Series(data_f32)

        assert actual.equals(expected)

    def test_to_pandas_df(self, df1, data_f32):
        actual = df1.to_pandas()
        expected = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': data_f32}, pd.Index([2, 3, 4, 5, 6]))

        assert actual.equals(expected)
