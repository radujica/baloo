import numpy as np

from baloo.weld import NumPyDecoder, NumPyEncoder, LazyResult, WeldLong, WeldObject


class TestLazyResult(object):
    def test_lazy_result_evaluate_raw(self):
        raw = np.array([1, 2, 3], dtype=np.int64)
        lazy_result = LazyResult(raw, WeldLong(), 1)

        evaluated = lazy_result.evaluate()
        expected = raw

        np.testing.assert_array_equal(evaluated, expected)

    def test_lazy_result_evaluate_lazy(self):
        raw = np.array([1, 2, 3], dtype=np.int64)
        weld_obj = WeldObject(NumPyEncoder(), NumPyDecoder())
        obj_id = weld_obj.update(raw)
        weld_obj.weld_code = '{}'.format(obj_id)
        lazy_result = LazyResult(weld_obj, WeldLong(), 1)

        evaluated = lazy_result.evaluate()
        expected = raw

        np.testing.assert_array_equal(evaluated, expected)
