from .weld_utils import create_weld_object, get_weld_obj_id

# TODO: don't cast to f64 if data is already f64


_weld_count_code = 'len({array})'


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
    obj_id, weld_obj = create_weld_object(array)

    weld_template = _weld_count_code

    weld_obj.weld_code = weld_template.format(array=obj_id)

    return weld_obj


_weld_aggregate_code = """result(
    for(
        {array},
        merger[{type}, {operation}],
        |b: merger[{type}, {operation}], i: i64, e: {type}| 
            merge(b, e)
    )
)"""


_weld_aggregate_code_f64 = """result(
    for(
        {array},
        merger[f64, {operation}],
        |b: merger[f64, {operation}], i: i64, e: {type}| 
            merge(b, f64(e))
    )
)"""


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
    obj_id, weld_obj = create_weld_object(array)

    weld_template = _weld_aggregate_code

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              operation=operation)

    return weld_obj


_weld_mean_code = 'f64({sum}) / f64(len({array}))'


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
    weld_obj_sum = weld_aggregate(array, weld_type, '+')

    obj_id, weld_obj = create_weld_object(array)
    weld_obj_sum_id = get_weld_obj_id(weld_obj, weld_obj_sum)

    weld_template = _weld_mean_code

    weld_obj.weld_code = weld_template.format(sum=weld_obj_sum_id,
                                              array=obj_id)

    return weld_obj


_weld_variance_code = """result(
    for(
        {array},
        merger[f64, +],
        |b: merger[f64, +], i: i64, n: {type}|
             merge(b, pow(f64(n) - {mean}, 2.0))
    )
) / f64(len({array}) - 1L)
"""


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
    weld_obj_mean = weld_mean(array, weld_type)

    obj_id, weld_obj = create_weld_object(array)
    weld_obj_mean_id = get_weld_obj_id(weld_obj, weld_obj_mean)

    weld_template = _weld_variance_code

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              mean=weld_obj_mean_id)

    return weld_obj


_weld_std_code = 'sqrt({var})'


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
    weld_obj_var = weld_variance(array, weld_type)

    obj_id, weld_obj = create_weld_object(weld_obj_var)
    weld_obj_var_id = get_weld_obj_id(weld_obj, weld_obj_var)

    weld_template = _weld_std_code

    weld_obj.weld_code = weld_template.format(var=weld_obj_var_id)

    return weld_obj


# full dependencies of each aggregation
_agg_dependencies = {
    'min': set(),
    'max': set(),
    'count': set(),
    'sum': set(),
    'prod': set(),
    'mean': {'sum'},
    'var': {'sum', 'mean'},
    'std': {'sum', 'mean', 'var'}
}

# to order the aggregations; lower means it comes first
_agg_priorities = {
    'min': 1,
    'max': 1,
    'count': 1,
    'sum': 1,
    'prod': 1,
    'mean': 2,
    'var': 3,
    'std': 4
}

_agg_code = {
    'min': _weld_aggregate_code_f64.replace('{operation}', 'min'),
    'max': _weld_aggregate_code_f64.replace('{operation}', 'max'),
    'count': 'f64({})'.format(_weld_count_code),
    'sum': _weld_aggregate_code_f64.replace('{operation}', '+'),
    'prod': _weld_aggregate_code_f64.replace('{operation}', '*'),
    'mean': _weld_mean_code.replace('{sum}', 'agg_sum'),
    'var': _weld_variance_code.replace('{mean}', 'agg_mean'),
    'std': _weld_std_code.replace('{var}', 'agg_var')
}


# not using the methods above because we don't want duplicate code chunks;
# for example, asking for sum and mean would make 2 weldobjects both computing the sum
# (+ 3rd computing the mean and using one of the sum objects);
# this method avoids that
def weld_agg(array, weld_type, aggregations):
    """Multiple aggregations, optimized.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input array.
    weld_type : WeldType
        Type of each element in the input array.
    aggregations : list of str
        Which aggregations to compute.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    from functools import reduce

    obj_id, weld_obj = create_weld_object(array)

    # find which aggregation computations are actually needed
    to_compute = reduce(lambda x, y: x | y, ({agg} | _agg_dependencies[agg] for agg in aggregations))
    # get priorities and sort in the proper order of computation
    to_compute = sorted(((agg, _agg_priorities[agg]) for agg in to_compute), key=lambda x: x[1])
    # remove the priorities
    to_compute = (agg_pair[0] for agg_pair in to_compute)
    aggs = '\n'.join(('let agg_{} = {};'.format(agg, _agg_code[agg]) for agg in to_compute))

    # these are the aggregations requested
    merges = '\n'.join(('let res = merge(res, {});'.format('agg_{}'.format(agg)) for agg in aggregations))
    mergers = """let res = appender[f64];
{merges}
result(res)
"""
    mergers = mergers.format(merges=merges)

    weld_template = '{}\n{}'.format(aggs, mergers)

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type)

    return weld_obj
