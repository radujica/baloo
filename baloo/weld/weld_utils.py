from weld.types import WeldInt16, WeldLong, WeldFloat, WeldBit, WeldDouble
from weld.weldobject import WeldObject

from .convertors import NumPyEncoder, NumPyDecoder

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def _create_empty_weld_object():
    return WeldObject(_encoder, _decoder)


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
    weld_obj = _create_empty_weld_object()
    obj_id = _get_weld_obj_id(weld_obj, data)

    return obj_id, weld_obj


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


# an attempt to avoid expensive casting
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
    elif isinstance(weld_type, WeldDouble) and isinstance(scalar, int):
        return '{}.0'.format(str(scalar))
    else:
        return '{}'.format(str(scalar))


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


def weld_cast_scalar(scalar, weld_type):
    """Returns the scalar casted to the request Weld type.

    Parameters
    ----------
    scalar : {int, float, str, bool, bytes, WeldObject}
        Input array.
    weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj = WeldObject(_encoder, _decoder)
    if isinstance(scalar, WeldObject):
        scalar = _get_weld_obj_id(weld_obj, scalar)

    weld_template = '{type}({scalar})'

    weld_obj.weld_code = weld_template.format(scalar=scalar,
                                              type=weld_type)

    return weld_obj


# this is fairly common so make separate method
def weld_cast_double(scalar):
    return weld_cast_scalar(scalar, WeldDouble())
