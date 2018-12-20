import numpy as np
import pytest

from baloo.weld.convertors import numpy_to_weld_type, weld_to_numpy_dtype
from baloo.weld.convertors.encoders import _numpy_to_weld_type_mapping
from baloo.weld.convertors.utils import to_weld_vec
from baloo.weld.lazy_result import LazyResult, LazyArrayResult
from baloo.weld.pyweld import *
from baloo.weld.weld_utils import weld_cast_scalar, weld_cast_array


class TestUtils(object):
    @pytest.mark.parametrize('ndim, expected', [
        (0, WeldLong()),
        (1, WeldVec(WeldLong())),
        (2, WeldVec(WeldVec(WeldLong())))
    ])
    def test_to_weld_vec(self, ndim, expected):
        result = to_weld_vec(WeldLong(), ndim)

        assert result == expected

    @pytest.mark.parametrize('np_dtype, weld_type',
                             list(_numpy_to_weld_type_mapping.items()))
    def test_numpy_to_weld_type(self, np_dtype, weld_type):
        result = numpy_to_weld_type(np_dtype)

        assert result == weld_type

    @pytest.mark.parametrize('np_dtype_str, weld_type', [
        ('S', WeldVec(WeldChar())),
        ('bytes_', WeldVec(WeldChar())),
        ('int16', WeldInt16()),
        ('int32', WeldInt()),
        ('int64', WeldLong()),
        ('float32', WeldFloat()),
        ('float64', WeldDouble()),
        ('bool', WeldBit())
    ])
    def test_numpy_to_weld_type_str(self, np_dtype_str, weld_type):
        result = numpy_to_weld_type(np_dtype_str)

        assert result == weld_type

    @pytest.mark.parametrize('scalar, weld_type', [
        (2.0, WeldInt16()),
        (2.0, WeldInt()),
        (2.0, WeldLong()),
        (2, WeldFloat()),
        (2, WeldDouble()),
        (1, WeldBit())
    ])
    def test_cast_scalar(self, scalar, weld_type):
        lazy_res = LazyResult(weld_cast_scalar(scalar, weld_type), weld_type, 0)

        assert scalar == lazy_res.evaluate()

    @pytest.mark.parametrize('scalar, weld_type', [
        (1, WeldVec(WeldBit())),
        (1, WeldChar()),
        ('str', WeldLong()),
        (b'bytes', WeldLong()),
        (True, WeldLong())
    ])
    def test_cast_scalar_not_supported(self, scalar, weld_type):
        with pytest.raises(TypeError):
            weld_cast_scalar(scalar, weld_type)

    @pytest.mark.parametrize('array_data, dtype, to_weld_type', [
        ([2.0, 3.0], np.float64, WeldInt16()),
        ([2.0, 3.0], np.float64, WeldInt()),
        ([2.0, 3.0], np.float64, WeldLong()),
        ([2, 3], np.int32, WeldFloat()),
        ([2, 3], np.int32, WeldDouble()),
        ([1, 0], np.int32, WeldBit())
    ])
    def test_cast_array(self, array_data, dtype, to_weld_type):
        data = np.array(array_data, dtype=dtype)

        actual = LazyArrayResult(weld_cast_array(data,
                                                 numpy_to_weld_type(np.dtype(dtype)),
                                                 to_weld_type),
                                 to_weld_type)
        expected = np.array(array_data, dtype=weld_to_numpy_dtype(to_weld_type))

        np.testing.assert_array_equal(actual.evaluate(), expected)

    @pytest.mark.parametrize('array_data, dtype, to_weld_type', [
        (['abc', 'def'], np.bytes_, WeldInt()),
        ([2.0, 3.0], np.float64, WeldChar())
    ])
    def test_cast_array_not_supported(self, array_data, dtype, to_weld_type):
        data = np.array(array_data, dtype=dtype)

        with pytest.raises(TypeError):
            weld_cast_array(data,
                            numpy_to_weld_type(np.dtype(dtype)),
                            to_weld_type)
