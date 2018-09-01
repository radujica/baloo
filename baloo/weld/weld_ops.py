from weld.types import *
from weld.weldobject import WeldObject

from .convertors import NumPyEncoder, NumPyDecoder

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


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
    scalar : {int, float, str, bool, bytes}
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

    weld_template = """map(
    {array},
    |a: {type}| 
        a {operation} {scalar}
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
