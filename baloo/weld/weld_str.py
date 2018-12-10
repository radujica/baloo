from weld.types import WeldLong, WeldChar, WeldVec

from .convertors import default_missing_data_literal
from .weld_utils import create_weld_object, to_weld_literal, get_weld_obj_id


# TODO: generalize if possible later
def weld_str_lower(array):
    """Convert values to lowercase.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)

    weld_template = """map(
    {array},
    |e: vec[i8]|
        result(
            for(e,
                appender[i8],
                |c: appender[i8], j: i64, f: i8|
                    if(f > 64c && f < 91c,
                        merge(c, f + 32c),
                        merge(c, f))
            )
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def weld_str_upper(array):
    """Convert values to uppercase.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)

    weld_template = """map(
    {array},
    |e: vec[i8]|
        result(
            for(e,
                appender[i8],
                |c: appender[i8], j: i64, f: i8|
                    if(f > 96c && f < 123c,
                        merge(c, f - 32c),
                        merge(c, f))
            )
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def weld_str_capitalize(array):
    """Capitalize first letter.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)

    weld_template = """map(
    {array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenString > 0L,
            let res = appender[i8];
            let firstChar = lookup(e, 0L);
            let res = if(firstChar > 96c && firstChar < 123c, merge(res, firstChar - 32c), merge(res, firstChar));
            result(
                for(slice(e, 1L, lenString - 1L),
                    res,
                    |c: appender[i8], j: i64, f: i8|
                        merge(c, f)
                )
            ),
            e)
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def weld_str_get(array, i):
    """Retrieve character at index i.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    i : int
        Index of character to retrieve. If greater than length of string, returns None.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    index_literal = to_weld_literal(i, WeldLong())
    missing_literal = default_missing_data_literal(WeldVec(WeldChar()))
    missing_literal_id = get_weld_obj_id(weld_obj, missing_literal)

    weld_template = """map(
    {array},
    |e: vec[i8]|
        let lenString = len(e);
        if({i} >= lenString,
            {missing},
            if({i} > 0L,
                result(merge(appender[i8], lookup(slice(e, 0L, lenString), {i}))),
                result(merge(appender[i8], lookup(slice(e, lenString, {i}), {i})))
            )
        )
)"""
    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              i=index_literal,
                                              missing=missing_literal_id)

    return weld_obj


def weld_str_strip(array):
    """Strip whitespace from start and end of elements.

    Note it currently only looks for whitespace (Ascii 32), not tabs or EOL.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)

    # +3L = +1 compensate start_i already +1'ed, +1 compensate end_i already -1'ed, +1 compensate for slice with size
    weld_template = """map(
    {array},
    |e: vec[i8]|
        let lenString = len(e);
        let res = appender[i8];
        let start_i = iterate(0L, |p| {{p + 1L, lookup(e, p) == 32c}});
        let end_i = iterate(lenString - 1L, |p| {{p - 1L, lookup(e, p) == 32c && p > 0L}});
        slice(e, start_i - 1L, end_i - start_i + 3L)
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


def _prepare_slice(i, default):
    if i is None:
        return default
    else:
        return to_weld_literal(i, WeldLong())


# TODO: allow negative step
def weld_str_slice(array, start=None, stop=None, step=None):
    """Slice each element.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    start : int, optional
    stop : int, optional
    step : int, optional

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    start = _prepare_slice(start, '0L')
    stop = _prepare_slice(stop, 'lenString')
    step = _prepare_slice(step, '1L')

    weld_template = """map(
    {array},
    |e: vec[i8]|
        let lenString = len(e);
        let stop = if({stop} > lenString, lenString, {stop});
        result(
            for(iter(e, {start}, stop, {step}),
                appender[i8],
                |c: appender[i8], j: i64, f: i8| 
                    merge(c, f)
            )
        ) 
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              start=start,
                                              stop=stop,
                                              step=step)

    return weld_obj
