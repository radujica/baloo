from collections import OrderedDict

import numpy as np
import pytest

from baloo import Series, Index, DataFrame, MultiIndex
from baloo.weld import create_placeholder_weld_object


# TODO: perhaps figure out autouse
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
    return np.array(['a', 'abc', 'goosfraba', '   dc  ', 'secretariat'], dtype=np.bytes_)


@pytest.fixture(scope='module')
def index_i64():
    return Index(np.arange(5), np.dtype(np.int64))


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
def df_small(data_f32, series_i64, series_str, index_i64):
    return DataFrame(OrderedDict((('a', data_f32), ('b', series_i64), ('c', series_str))), index_i64)


@pytest.fixture(scope='module')
def df_small_columns():
    return Index(np.array(['a', 'b', 'c'], dtype=np.bytes_))


# TODO: change this to empty constructor
@pytest.fixture(scope='module')
def df_empty():
    return DataFrame({})


@pytest.fixture(scope='module',
                params=[df_small(data_f32(),
                                 series_i64(data_i64_lazy(data_i64()), index_i64()),
                                 series_str(data_str(), index_i64()),
                                 index_i64()),
                        df_empty()])
def df(request):
    return request.param


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
