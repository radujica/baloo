import numpy as np
import pytest

from baloo import RangeIndex, Index
from baloo.weld import create_placeholder_weld_object, WeldObject, WeldLong
from .test_base import assert_index_equal


def assert_range_equal(actual, expected):
    assert actual.start == expected.start

    if isinstance(actual.stop, WeldObject):
        actual.stop = actual.stop.evaluate(WeldLong(), verbose=False)
    assert actual.stop == expected.stop

    if isinstance(expected.stop, WeldObject):
        expected.stop = expected.stop.evaluate(WeldLong(), verbose=False)
    assert actual.step == expected.step

    actual = actual.evaluate()
    expected = expected.evaluate()

    assert_index_equal(actual, expected)


class TestRangeIndex(object):
    def test_init_single_arg(self, range_index):
        assert_range_equal(RangeIndex(5), range_index)

    def test_init_single_arg_lazy(self, data_i64, range_index):
        weld_obj = create_placeholder_weld_object(data_i64)
        weld_obj.weld_code = 'len({})'.format(weld_obj.weld_code)
        actual = RangeIndex(weld_obj)

        assert_range_equal(actual, range_index)

    # TODO: change when implemented
    def test_init_negative_step(self):
        with pytest.raises(ValueError):
            RangeIndex(5, 0, -1)

    def test_evaluate(self, index_i64, range_index):
        actual = range_index.evaluate()
        expected = index_i64

        assert_index_equal(actual, expected)

    def test_len_raw(self, range_index):
        actual = len(range_index)
        expected = 5

        assert actual == expected

    def test_len_lazy(self, data_i64):
        weld_obj = create_placeholder_weld_object(data_i64)
        weld_obj.weld_code = 'len({})'.format(weld_obj.weld_code)
        ind = RangeIndex(weld_obj)

        actual = len(ind)
        expected = 5

        assert actual == expected

    def test_comparison(self, range_index):
        actual = range_index < 3
        expected = Index(np.array([True, True, True, False, False]))

        assert_index_equal(actual, expected)

        with pytest.raises(TypeError):
            range_index < 3.0

    def test_filter(self, range_index):
        actual = range_index[Index(np.array([False, True, True, False, False]))]
        expected = Index(np.array([1, 2]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_slice(self, range_index):
        actual = range_index[1:3]
        expected = Index(np.array([1, 2]), np.dtype(np.int64))

        assert_index_equal(actual, expected)

    def test_head(self, range_index):
        actual = range_index.head(2)
        expected = Index(np.array([0, 1]))

        assert_index_equal(actual, expected)

    def test_tail(self, range_index):
        actual = range_index.tail(2)
        expected = Index(np.array([3, 4]))

        assert_index_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.array(np.arange(2, 7))),
        ('-', np.array(np.arange(-2, 3))),
        ('*', np.array(np.arange(0, 9, 2))),
        ('/', np.array([0, 0, 1, 1, 2]))
    ])
    def test_op_array(self, operation, expected_data, range_index):
        other = Index(np.array([2] * 5))

        actual = eval('range_index {} other'.format(operation))
        expected = Index(expected_data, np.dtype(np.int64))

        assert_index_equal(actual, expected)

    @pytest.mark.parametrize('operation, expected_data', [
        ('+', np.array(np.arange(2, 7))),
        ('-', np.array(np.arange(-2, 3))),
        ('*', np.array(np.arange(0, 9, 2))),
        ('/', np.array([0, 0, 1, 1, 2]))
    ])
    def test_op_scalar(self, operation, expected_data, range_index):
        actual = eval('range_index {} 2'.format(operation))
        expected = Index(expected_data, np.dtype(np.int64))

        assert_index_equal(actual, expected)
