from .weld_utils import _create_weld_object


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

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              operation=operation)

    return weld_obj


def weld_mean(array, weld_type):
    """Returns the mean of the array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.
    weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    # need to first get the sum, then create our weld_obj; this ensures the order of obj_ids is correct
    weld_obj_sum = weld_aggregate(array, weld_type, '+')

    obj_id, weld_obj = _create_weld_object(array)
    weld_obj.dependencies[weld_obj_sum.obj_id] = weld_obj_sum

    weld_template = 'f64({sum}) / f64(len({array}))'

    weld_obj.weld_code = weld_template.format(sum=weld_obj_sum.obj_id,
                                              array=obj_id)

    return weld_obj


def weld_variance(array, weld_type):
    """Returns the variance of the array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.
    weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    # need to first get the mean, then create our weld_obj; this ensures the order of obj_ids is correct
    weld_obj_mean = weld_mean(array, weld_type)

    obj_id, weld_obj = _create_weld_object(array)
    weld_obj.dependencies[weld_obj_mean.obj_id] = weld_obj_mean

    weld_template = """result(
    for(
        {array},
        merger[f64, +],
        |b: merger[f64, +], i: i64, n: {type}|
             merge(b, pow(f64(n) - {mean}, 2.0))
    )
) / f64(len({array}) - 1L)
"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              mean=weld_obj_mean.obj_id)

    return weld_obj


def weld_standard_deviation(array, weld_type):
    """Returns the *sample* standard deviation of the array.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.
    weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    # need to first get the variance, then create our weld_obj; this ensures the order of obj_ids is correct
    weld_obj_var = weld_variance(array, weld_type)

    obj_id, weld_obj = _create_weld_object(array)
    weld_obj.dependencies[weld_obj_var.obj_id] = weld_obj_var

    weld_template = 'sqrt({variance})'

    weld_obj.weld_code = weld_template.format(variance=weld_obj_var.obj_id)

    return weld_obj
