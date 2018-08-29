from weld.weldobject import WeldObject

from .convertors import NumPyEncoder, NumPyDecoder

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def _create_weld_object(data):
    weld_obj = WeldObject(_encoder, _decoder)

    obj_id = weld_obj.update(data)
    if isinstance(data, WeldObject):
        obj_id = data.obj_id
        weld_obj.dependencies[obj_id] = data

    return obj_id, weld_obj


def create_placeholder_weld_object(data):
    weld_obj = WeldObject(_encoder, _decoder)
    obj_id = weld_obj.update(data)
    weld_obj.weld_code = '{}'.format(str(obj_id))

    return weld_obj


def weld_count(array):
    """Returns the length of the array.

    Parameters
    ----------
    array : np.ndarray or WeldObject
        Input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = _create_weld_object(array)

    weld_template = 'len(%(array)s)'
    weld_obj.weld_code = weld_template % {'array': obj_id}

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

    # TODO: maybe use \n for cleaner template?
    weld_template = """result(
        for(
            rangeiter(%(start)sL, %(stop)s, %(step)sL),
            appender[i64],
            |b: appender[i64], i: i64, e: i64| 
                merge(b, e)
        )
    )"""

    weld_obj.weld_code = weld_template % {'start': start,
                                          'stop': '{}L'.format(stop) if isinstance(stop, int) else stop,
                                          'step': step}

    return weld_obj
