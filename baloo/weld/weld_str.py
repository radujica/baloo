from .weld_utils import create_weld_object


def weld_str_lower(array):
    """Convert values to lowercase.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Data.

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
