import numpy as np

from ..weld import WeldObject, weld_count, WeldBit, WeldLong, LazyResult, supported_dtype_chars


def check_type(data, expected_types):
    if data is not None and not isinstance(data, expected_types):
        raise TypeError('Expected: {}'.format(str(expected_types)))

    return data


def check_dtype(data):
    # silently allow and convert bool
    if data == bool:
        return np.bool_

    if data is not None and not (isinstance(data, np.dtype) or data.__module__ == np.__name__):
        raise TypeError('Expected a valid NumPy dtype, received: {}'.format(str(data)))

    return data


def check_inner_types(data, expected_types):
    if data is not None:
        for value in data:
            check_type(value, expected_types)

    return data


def check_str_or_list_str(data):
    check_type(data, (list, str))
    if data is not None:
        check_inner_types(as_list(data), str)

    return data


def check_weld_bit_array(data):
    return check_type(data.weld_type, WeldBit)


def check_weld_long_array(data):
    return check_type(data.weld_type, WeldLong)


def infer_dtype(data, arg_dtype):
    if arg_dtype is not None:
        if not isinstance(arg_dtype, np.dtype):
            arg_dtype = np.dtype(arg_dtype)

        return arg_dtype
    else:
        if isinstance(data, np.ndarray):
            return data.dtype
        elif isinstance(data, WeldObject):
            # if WeldObject data then arg_dtype must have been passed as argument
            raise ValueError('Using WeldObject as data requires the dtype as argument')
        else:
            raise ValueError('Unsupported data type: {}'.format(str(type(data))))


def infer_length(data):
    if len(data) == 0:
        return 0
    else:
        for value in data:
            if isinstance(value, (np.ndarray, list)):
                return len(value)
            # Series then
            elif isinstance(value.values, np.ndarray):
                return len(value.values)

        return None


def default_index(data):
    from .indexes import RangeIndex

    if isinstance(data, int):
        return RangeIndex(data)
    elif isinstance(data, np.ndarray):
        return RangeIndex(len(data))
    elif isinstance(data, WeldObject):
        # must be WeldObject then
        return RangeIndex(weld_count(data))
    elif isinstance(data, LazyResult):
        return RangeIndex(weld_count(data.values))
    else:
        raise ValueError('Unsupported data type: {}'.format(str(type(data))))


def is_scalar(data):
    return isinstance(data, (int, float, str, bytes, bool))


def _is_int_or_none(value):
    return value is None or isinstance(value, int)


def _valid_int_slice(slice_):
    return all([_is_int_or_none(v) for v in [slice_.start, slice_.stop, slice_.step]])


def check_valid_int_slice(slice_):
    if not _valid_int_slice(slice_):
        raise ValueError('Can currently only slice with integers')


def shorten_data(data):
    if not isinstance(data, np.ndarray):
        raise TypeError('Cannot shorten unevaluated data. First call evaluate()')

    if len(data) > 50:
        return list(np.concatenate([data[:20], np.array(['...']), data[-20:]]))
    else:
        return data


def as_list(data):
    if isinstance(data, list):
        return data
    else:
        return [data]


def replace_if_none(value, default):
    return default if value is None else value


def replace_slice_defaults(slice_, default_start, default_stop, default_step):
    start = replace_if_none(slice_.start, default_start)
    stop = replace_if_none(slice_.stop, default_stop)
    step = replace_if_none(slice_.step, default_step)

    return slice(start, stop, step)


def convert_to_numpy(data):
    data = np.array(data)
    if data.dtype.char not in supported_dtype_chars:
        raise TypeError('dtype {} is not supported'.format(data.dtype))

    return data


def same_index(index1, index2):
    if not isinstance(index1, type(index2)):
        return False
    elif index1._gather_names() != index2._gather_names():
        return False
    elif index1._gather_weld_types() != index2._gather_weld_types():
        return False
    else:
        # should be Indexes
        data1 = index1._gather_data().values()
        data2 = index2._gather_data().values()

        for d1, d2 in zip(data1, data2):
            if not isinstance(d1.values, type(d2.values)):
                return False
            elif isinstance(d1.values, np.ndarray):
                if not np.array_equal(d1.values, d2.values):
                    return False
            # else they're WeldObjects so can only assume they're the same

    return True
