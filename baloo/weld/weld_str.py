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
