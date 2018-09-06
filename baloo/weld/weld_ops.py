from weld.types import *
from weld.weldobject import WeldObject

from .convertors import NumPyEncoder, NumPyDecoder

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


# Weld types can be inferred in many places however were included for performance reasons.


def _get_weld_obj_id(weld_obj, data):
    """Helper method to update WeldObject with some data.

    Parameters
    ----------
    weld_obj : WeldObject
        WeldObject to update.
    data : numpy.ndarray or WeldObject
        Data for which to get an id.

    Returns
    -------
    str
        The obj_id of the data.

    """
    obj_id = weld_obj.update(data)
    if isinstance(data, WeldObject):
        obj_id = data.obj_id
        weld_obj.dependencies[obj_id] = data

    return obj_id


def _create_weld_object(data):
    """Helper method to create a WeldObject and update with data.

    Parameters
    ----------
    data : numpy.ndarray or WeldObject
        Data to include in newly created object.

    Returns
    -------
    (str, WeldObject)
        Object id for the data to use in the Weld code and
        the WeldObject updated with the data.

    """
    weld_obj = WeldObject(_encoder, _decoder)
    obj_id = _get_weld_obj_id(weld_obj, data)

    return obj_id, weld_obj


def create_placeholder_weld_object(data):
    """Helper method that creates a WeldObject that evaluates to itself.

    Parameters
    ----------
    data : numpy.ndarray or WeldObject
        Data to wrap around.

    Returns
    -------
    WeldObject
        WeldObject wrapped around data.

    """
    weld_obj = WeldObject(_encoder, _decoder)
    obj_id = _get_weld_obj_id(weld_obj, data)
    weld_obj.weld_code = '{}'.format(str(obj_id))

    return weld_obj


# this method avoids expensive Weld casting
def _to_weld_literal(scalar, weld_type):
    """Return scalar formatted for Weld.

    Parameters
    ----------
    scalar : {int, float, str, bool, bytes}
        Scalar data to convert to weld literal.
    weld_type : WeldType
        Desired Weld type.

    Returns
    -------
    str
        String of the scalar to use in Weld code.

    Examples
    --------
    >>> _to_weld_literal(4, WeldLong())
    '4L'

    """
    if isinstance(weld_type, WeldInt16):
        return '{}si'.format(str(scalar))
    elif isinstance(weld_type, WeldLong):
        return '{}L'.format(str(scalar))
    elif isinstance(weld_type, WeldFloat):
        return '{}f'.format(str(scalar))
    elif isinstance(weld_type, WeldBit):
        return '{}'.format(str(scalar).lower())
    else:
        return '{}'.format(str(scalar))


def weld_count(array):
    """Returns the length of the array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    weld_template = 'len({array})'

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def weld_range(start, stop, step):
    """Create a vector for the range parameters above.

    Parameters
    ----------
    start : int
    stop : int or WeldObject
        Could be the lazily computed length of a WeldObject vec.
    step : int

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    if isinstance(stop, WeldObject):
        obj_id, weld_obj = _create_weld_object(stop)
        stop = obj_id
    else:
        weld_obj = WeldObject(_encoder, _decoder)

    weld_template = """result(
    for(
        rangeiter({start}L, {stop}, {step}L),
        appender[i64],
        |b: appender[i64], i: i64, e: i64| 
            merge(b, e)
    )
)"""

    stop = '{}L'.format(stop) if isinstance(stop, int) else stop

    weld_obj.weld_code = weld_template.format(start=start,
                                              stop=stop,
                                              step=step)

    return weld_obj


def weld_compare(array, scalar, operation, weld_type):
    """Applies comparison operation between each element in the array with scalar.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    scalar : {int, float, str, bool, bytes_}
        Value to compare with; must be same type as the values in the array. If not a str,
        it is casted to weld_type (allowing one to write e.g. native Python int).
    operation : str
        Operation to do out of: {<, <=, ==, !=, >=, >}.
    weld_type : WeldType
        Type of the elements in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    scalar = _to_weld_literal(scalar, weld_type)

    # TODO: there should be no casting! requires Weld fix
    weld_template = """map(
    {array},
    |a: {type}| 
        a {operation} {type}({scalar})
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              scalar=scalar,
                                              operation=operation,
                                              type=weld_type)

    return weld_obj


def weld_filter(array, weld_type, bool_array):
    """Returns a new array only with the elements with a corresponding True in bool_array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    weld_type : WeldType
        Type of the elements in the input array.
    bool_array : numpy.ndarray or WeldObject
        Array of bool with True for elements in array desired in the result array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)
    bool_obj_id = _get_weld_obj_id(weld_obj, bool_array)

    weld_template = """result(
    for(
        zip({array}, {bool_array}),
        appender[{type}],
        |b: appender[{type}], i: i64, e: {{{type}, bool}}| 
            if (e.$1, 
                merge(b, e.$0), 
                b)
    )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              bool_array=bool_obj_id,
                                              type=weld_type)

    return weld_obj


def _replace_slice_defaults(slice_, default_start, default_step):
    start = slice_.start
    stop = slice_.stop
    step = slice_.step

    if start is None:
        start = default_start

    # stop is required when making a slice, no need to replace

    if step is None:
        step = default_step

    return slice(start, stop, step)


def weld_slice(array, weld_type, slice_, default_start=0, default_step=1):
    """Returns a new array according to the given slice.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        1-dimensional array.
    weld_type : WeldType
        Type of the elements in the input array.
    slice_ : slice
        Subset to return. Assumed valid slice.
    default_start : int, optional
        Default value to slice start.
    default_step : int, optional
        Default value to slice step.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    slice_ = _replace_slice_defaults(slice_, default_start, default_step)
    obj_id, weld_obj = _create_weld_object(array)

    if slice_.step == 1:
        weld_template = """slice(
    {array},
    {slice_start},
    {slice_stop}
)"""
    else:
        weld_template = """result(
    for(
        iter({array}, {slice_start}, {slice_stop}, {slice_step}),
        appender[{type}],
        |b: appender[{type}], i: i64, e: {type}| 
            merge(b, n)
    )  
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              slice_start='{}L'.format(slice_.start),
                                              slice_stop='{}L'.format(slice_.stop - slice_.start),
                                              slice_step='{}L'.format(slice_.step))

    return weld_obj


# TODO: could generalize weld_slice to accept slice with possible WeldObjects
def weld_tail(array, length, n):
    """Return the last n elements.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Array to select from.
    length : int or WeldObject
        Length of the array. Int if already known can simplify the computation.
    n : int
        How many values.

    Returns
    -------
    WeldObject
        Representation of the computation.

    """
    obj_id, weld_obj = _create_weld_object(array)
    if isinstance(length, WeldObject):
        length = _get_weld_obj_id(weld_obj, length)

    weld_template = """slice(
    {array},
    {slice_start},
    {slice_stop}
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              slice_start='{}L - {}L'.format(length, n),
                                              slice_stop='{}L'.format(length))

    return weld_obj


def weld_array_op(array1, array2, result_type, operation):
    """Applies operation to each pair of elements in the arrays.

    Their lengths and types are assumed to be the same.
    TODO: what happens if not?

    Parameters
    ----------
    array1 : numpy.ndarray or WeldObject
        Input array.
    array2 : numpy.ndarray or WeldObject
        Second input array.
    result_type : WeldType
        Weld type of the result. Expected to be the same as both input arrays.
    operation : {'+', '-', '*', '/', '&&', '||', 'pow'}
        Which operation to apply. Note bitwise operations (not included) seem to be bugged at the LLVM level.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id1, weld_obj = _create_weld_object(array1)
    obj_id2 = _get_weld_obj_id(weld_obj, array2)

    if operation == 'pow':
        action = 'pow(e.$0, e.$1)'
    else:
        action = 'e.$0 {operation} e.$1'.format(operation=operation)

    weld_template = """result(
    for(zip({array1}, {array2}), 
        appender[{type}], 
        |b: appender[{type}], i: i64, e: {{{type}, {type}}}| 
            merge(b, {action})
    )
)"""

    weld_obj.weld_code = weld_template.format(array1=obj_id1,
                                              array2=obj_id2,
                                              type=result_type,
                                              action=action)

    return weld_obj


def weld_invert(array):
    """Inverts a bool array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data. Assumed to be bool data.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    weld_template = """result(
    for({array},
        appender[bool],
        |b: appender[bool], i: i64, e: bool|
            if(e, merge(b, false), merge(b, true))
    )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def weld_iloc_int(array, index):
    """Retrieves the value at index.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data. Assumed to be bool data.
    index : int
        The array index from which to retrieve value.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    weld_template = 'lookup({array}, {index}L)'

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              index=index)

    return weld_obj


def weld_element_wise_op(array, weld_type, scalar, operation):
    """Applies operation to each element in the array with scalar.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.
    weld_type : WeldType
        Type of each element in the input array.
    scalar : {int, float, str, bool, bytes_}
        Value to compare with; must be same type as the values in the array. If not a str,
        it is casted to weld_type (allowing one to write e.g. native Python int).
    operation : {+, -, *, /, pow}

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    scalar = _to_weld_literal(scalar, weld_type)

    if operation == 'pow':
        action = 'pow(e, {scalar})'.format(scalar=scalar)
    else:
        action = 'e {operation} {scalar}'.format(scalar=scalar,
                                                 operation=operation)

    weld_template = """result(
    for({array}, 
        appender[{type}], 
        |b: appender[{type}], i: i64, e: {type}| 
            merge(b, {action})
    )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              action=action)

    return weld_obj


def weld_aggregate(array, weld_type, operation):
    """Returns operation on the elements in the array.

    Arguments
    ---------
    array : WeldObject or numpy.ndarray
        Input array.
    weld_type : WeldType
        Weld type of each element in the input array.
    operation : {'+', '*', 'min', 'max'}
        Operation to apply.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    weld_template = """result(
    for(
        {array},
        merger[{type}, {operation}],
        |b: merger[{type}, {operation}], i: i64, e: {type}| 
            merge(b, e)
    )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id, type=weld_type, operation=operation)

    return weld_obj


def weld_combine_scalars(scalars, weld_type):
    """Combine column-wise aggregations (so resulting scalars) into a single array.

    Parameters
    ----------
    scalars : tuple of WeldObjects
        WeldObjects to combine.
    weld_type : WeldType
        The Weld type of the result. Currently expecting scalars to be of the same type.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj = WeldObject(_encoder, _decoder)
    obj_ids = (_get_weld_obj_id(weld_obj, scalar) for scalar in scalars)

    merges = '\n'.join(('let res = merge(res, {});'.format(obj_id) for obj_id in obj_ids))

    weld_template = """let res = appender[{type}];
{merges}
result(res)
"""

    weld_obj.weld_code = weld_template.format(type=weld_type,
                                              merges=merges)

    return weld_obj
