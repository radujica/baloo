import numpy as np
from weld.types import WeldLong
from weld.weldobject import WeldObject

from baloo import RangeIndex, Index
from baloo.weld import create_placeholder_weld_object
from .test_base import assert_index_equal


def assert_range_equal(actual, expected):
    assert actual.start == expected.start
    if isinstance(actual.stop, WeldObject):
        actual.stop = actual.stop.evaluate(WeldLong(), verbose=False)
    if isinstance(expected.stop, WeldObject):
        expected.stop = expected.stop.evaluate(WeldLong(), verbose=False)
    assert actual.stop == expected.stop
    assert actual.step == expected.step

    actual = actual.evaluate()
    expected = expected.evaluate()

    assert_index_equal(actual, expected)


class TestRangeIndex(object):
    def test_init_single_arg(self):
        actual = RangeIndex(3)
        expected = RangeIndex(0, 3, 1)

        assert_range_equal(actual, expected)

    def test_init_single_arg_lazy(self):
        data = np.array([0, 1, 2], dtype=np.int64)
        weld_obj = create_placeholder_weld_object(data)
        weld_obj.weld_code = 'len({})'.format(weld_obj.weld_code)
        actual = RangeIndex(weld_obj)
        expected = RangeIndex(0, 3, 1)

        assert_range_equal(actual, expected)

    def test_evaluate(self):
        actual = RangeIndex(0, 3, 1).evaluate()

        data = np.array([0, 1, 2], dtype=np.int64)
        expected = Index(data, np.dtype(np.int64), None)

        assert_index_equal(actual, expected)

    def test_len_raw(self):
        ind = RangeIndex(0, 3, 1)

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_len_lazy(self):
        data = np.array([0, 1, 2], dtype=np.int64)
        weld_obj = create_placeholder_weld_object(data)
        weld_obj.weld_code = 'len({})'.format(weld_obj.weld_code)
        ind = RangeIndex(weld_obj)

        actual = len(ind)
        expected = 3

        assert actual == expected

    def test_slice(self):
        ind = RangeIndex(1, 6)

        actual = ind[1:3]
        expected = Index(np.array([2, 3]), np.dtype(np.int64))

        assert_index_equal(actual, expected)
