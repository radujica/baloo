import numpy as np

from ..weld import weld_count, WeldObject


def check_type(data, expected_types):
    if data is not None and not isinstance(data, expected_types):
        raise TypeError('Expected: {}'.format(str(expected_types)))

    return data


def infer_dtype(data, arg_dtype):
    if arg_dtype is not None:
        return arg_dtype
    else:
        if isinstance(data, np.ndarray):
            return data.dtype
        elif isinstance(data, WeldObject):
            # if WeldObject data then arg_dtype must have been passed as argument
            raise ValueError('Using WeldObject as data requires the dtype as argument')
        else:
            raise ValueError('Unsupported data type: {}'.format(str(type(data))))


def default_index(data):
    from .indexes import RangeIndex

    if isinstance(data, np.ndarray):
        return RangeIndex(len(data))
    elif isinstance(data, WeldObject):
        # must be WeldObject then
        return RangeIndex(weld_count(data))
    else:
        raise ValueError('Unsupported data type: {}'.format(str(type(data))))


def is_scalar(data):
    return isinstance(data, (int, float, str, bytes, bool))


def _is_int_or_none(value):
    return value is None or isinstance(value, int)


def valid_int_slice(slice_):
    return all([_is_int_or_none(v) for v in [slice_.start, slice_.stop, slice_.step]])
