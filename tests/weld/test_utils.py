import pytest

from baloo.weld.convertors.utils import to_weld_vec
from baloo.weld.convertors import numpy_to_weld_type
from baloo.weld.convertors.encoders import _numpy_to_weld_type_mapping

from weld.types import *


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
        ('byte', WeldChar()),
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
