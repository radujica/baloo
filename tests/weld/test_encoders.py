import numpy as np
import pytest

from baloo.weld import LazyResult, NumPyEncoder, NumPyDecoder
from baloo.weld.pyweld import *


# TODO: should be restricted to the encoders, i.e. not through LazyResult/WeldObject
class TestNumPyEncoders(object):
    _encoder = NumPyEncoder()
    _decoder = NumPyDecoder()

    @pytest.mark.parametrize('data, weld_type', [
        (np.array([1, 2, 3], dtype=np.int16), WeldInt16()),
        (np.array([1, 2, 3], dtype=np.int32), WeldInt()),
        (np.array([1, 2, 3], dtype=np.int64), WeldLong()),
        (np.array([1, 2, 3], dtype=np.float32), WeldFloat()),
        (np.array([1, 2, 3], dtype=np.float64), WeldDouble()),
        (np.array([True, True, False], dtype=np.bool), WeldBit()),
        (np.array(['aaa', 'bb', 'c'], dtype=np.bytes_), WeldVec(WeldChar()))
    ])
    def test_array(self, data, weld_type):
        weld_obj = WeldObject(self._encoder, self._decoder)
        obj_id = weld_obj.update(data)
        weld_obj.weld_code = '{}'.format(obj_id)
        lazy_result = LazyResult(weld_obj, weld_type, 1)

        evaluated = lazy_result.evaluate()
        expected = data

        np.testing.assert_array_equal(evaluated, expected)

    @pytest.mark.parametrize('data, weld_type', [
        (np.array([1, np.nan, 3], dtype=np.float32), WeldFloat()),
        (np.array([1, np.nan, 3], dtype=np.float64), WeldDouble())
    ])
    def test_array_with_missing(self, data, weld_type):
        weld_obj = WeldObject(self._encoder, self._decoder)
        obj_id = weld_obj.update(data)
        weld_obj.weld_code = '{}'.format(obj_id)
        lazy_result = LazyResult(weld_obj, weld_type, 1)

        evaluated = lazy_result.evaluate()
        expected = data

        np.testing.assert_array_equal(evaluated, expected)

    @pytest.mark.parametrize('data, weld_type, expected', [
        (np.array([1, 2, 3], dtype=np.int16), WeldInt16(), np.int16(6)),
        (np.array([1, 2, 3], dtype=np.int32), WeldInt(), np.int32(6)),
        (np.array([1, 2, 3], dtype=np.int64), WeldLong(), np.int64(6)),
        (np.array([1, 2, 3], dtype=np.float32), WeldFloat(), np.float32(6)),
        (np.array([1, 2, 3], dtype=np.float64), WeldDouble(), np.float64(6))
    ])
    def test_scalar(self, data, weld_type, expected):
        weld_obj = WeldObject(self._encoder, self._decoder)
        obj_id = weld_obj.update(data)
        weld_obj.weld_code = 'result(for({}, merger[{}, +], |b, i, e| merge(b, e)))'.format(obj_id, str(weld_type))
        lazy_result = LazyResult(weld_obj, weld_type, 0)

        evaluated = lazy_result.evaluate()

        assert evaluated == expected

    def test_str(self):
        data = 'abc'
        weld_obj = WeldObject(self._encoder, self._decoder)
        obj_id = weld_obj.update(data)
        weld_obj.weld_code = '{}'.format(obj_id)
        res = LazyResult(weld_obj, WeldChar(), 1)

        actual = res.evaluate()
        expected = data

        assert actual == expected

    def test_struct(self):
        data1 = np.array([1, 2, 3], dtype=np.int64)
        data2 = np.array([2, 3, 4], dtype=np.int64)

        weld_obj = WeldObject(self._encoder, self._decoder)
        obj_id1 = weld_obj.update(data1)
        obj_id2 = weld_obj.update(data2)
        weld_obj.weld_code = '{{{}, {}}}'.format(obj_id1, obj_id2)
        lazy_result = LazyResult(weld_obj, WeldStruct([WeldVec(WeldLong()), WeldVec(WeldLong())]), 0)

        evaluated = lazy_result.evaluate()
        expected = (data1, data2)

        np.testing.assert_array_equal(evaluated[0], expected[0])
        np.testing.assert_array_equal(evaluated[1], expected[1])
