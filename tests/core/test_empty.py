import numpy as np
import pytest

from baloo import Index, Series
from .indexes.utils import assert_indexes_equal
from .test_series import assert_series_equal
from .test_frame import assert_dataframe_equal


class TestEmptyDataFrame(object):
    def test_aggregation_empty(self, df_empty):
        assert_series_equal(df_empty.min(), Series(np.empty(0, dtype=np.float64)))

    @pytest.mark.parametrize('op,expected', [
        ('df < 2', 'df'),
        ('df[2:]', 'df'),
        ('df * 2', 'df'),
        ('df.head()', 'df'),
        ('df.tail()', 'df'),
        ('df.sort_index()', 'df'),
        ('df.reset_index()', 'DataFrame({}, RangeIndex(0, 0, 1))')
    ])
    def test_empty_ops(self, df_empty, op, expected):
        df = df_empty
        assert_dataframe_equal(eval(op), eval(expected))

        with pytest.raises(ValueError):
            df.agg(['mean', 'var'])

    @pytest.mark.parametrize('op,exception', [
        ('df.agg(["mean", "var"])', ValueError),
        ('df["a"]', KeyError),
        ('df[["a", "b"]]', KeyError),
        ('df.drop("a")', KeyError),
        ('df.drop(["a", "b"])', KeyError)
    ])
    def test_empty_exceptions(self, df_empty, op, exception):
        df = df_empty

        with pytest.raises(exception):
            eval(op)

    def test_keys_empty(self, df_empty):
        assert_indexes_equal(df_empty.keys(), Index(np.empty(0)))
