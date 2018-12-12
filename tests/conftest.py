from collections import OrderedDict

import numpy as np
import pytest

from baloo import Series, Index, DataFrame, MultiIndex, RangeIndex
from baloo.weld import create_placeholder_weld_object


# TODO: perhaps figure out autouse
# TODO: add nulls to the data and make sure all ops still work
# TODO: could maybe test with names also?
@pytest.fixture(scope='module')
def data_f32():
    return np.arange(1, 6, dtype=np.float32)


@pytest.fixture(scope='module')
def data_i64():
    return np.arange(1, 6, dtype=np.int64)


@pytest.fixture(scope='module')
def data_i64_lazy(data_i64):
    return create_placeholder_weld_object(data_i64)


@pytest.fixture(scope='module')
def data_str():
    return np.array(['a', 'Abc', 'goosfraba', '   dC  ', 'secrETariat'], dtype=np.bytes_)


@pytest.fixture(scope='module')
def series_str(data_str, index_i64):
    return Series(data_str, index_i64, np.bytes_)


@pytest.fixture(scope='module')
def series_str_2(index_i64):
    return Series([b'abz', b'zabz', b'zab', b'  ab  ', b'a'], index_i64, np.bytes_)


@pytest.fixture(scope='module')
def index_i64():
    return Index(np.arange(5), np.dtype(np.int64))


@pytest.fixture(scope='module')
def range_index():
    return RangeIndex(0, 5, 1)


@pytest.fixture(scope='module')
def series_f32(data_f32, index_i64):
    return Series(data_f32, index_i64, np.dtype(np.float32))


@pytest.fixture(scope='module')
def series_i64(data_i64_lazy, index_i64):
    return Series(data_i64_lazy, index_i64, np.dtype(np.int64))


@pytest.fixture(scope='module')
def series_str(data_str, index_i64):
    return Series(data_str, index_i64, data_str.dtype)


@pytest.fixture(scope='module')
def op_array_other():
    return Series(np.array([2] * 5).astype(np.float32))


@pytest.fixture(scope='module')
def df_small(data_f32, series_i64, series_str, index_i64):
    return DataFrame(OrderedDict((('a', data_f32), ('b', series_i64), ('c', series_str))), index_i64)


@pytest.fixture(scope='module')
def df_small_columns():
    return Index(np.array(['a', 'b', 'c'], dtype=np.bytes_))


@pytest.fixture(scope='module')
def df_empty():
    return DataFrame()


@pytest.fixture(scope='module')
def index_i64_2():
    return Index(np.arange(2, 7), np.dtype(np.int64))


@pytest.fixture(scope='module')
def df1(data_f32, index_i64_2):
    return DataFrame(OrderedDict((('a', Series(np.arange(5))), ('b', data_f32))), index_i64_2)


@pytest.fixture(scope='module')
def df2():
    return DataFrame(OrderedDict((('b', np.arange(3, 6, dtype=np.float32)), ('c', np.arange(4, 7)))),
                     MultiIndex([np.array([1, 3, 5]),
                                 Index(np.array(['abc', 'def', 'efgh'], dtype=np.bytes_))],
                                ['a', 'd']))


@pytest.fixture(scope='module')
def df_dupl(series_i64, index_i64):
    return DataFrame(OrderedDict((('a', np.array([0, 1, 1, 2, 3], dtype=np.float32)),
                                  ('b', [4, 5, 5, 6, 6]),
                                  ('c', series_i64))),
                     index_i64)


@pytest.fixture(scope='module')
def df_dupl_exp_ind():
    return Index(np.array([4, 5, 6]), np.dtype(np.int64), 'b')


@pytest.fixture(scope='module')
def series_unsorted(index_i64):
    return Series(np.array([5, 2, 3, 1, 4], dtype=np.float32), index_i64, np.dtype(np.float32))
