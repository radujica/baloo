from .cache import Cache
from .convertors import default_missing_data_literal
from .lazy_result import LazyStructOfVecResult, LazyStructResult
from .pyweld import WeldLong, WeldStruct, WeldVec, WeldChar
from .weld_utils import weld_arrays_to_vec_of_struct, create_empty_weld_object, get_weld_obj_id, \
    extract_placeholder_weld_objects, extract_placeholder_weld_objects_from_index, weld_data_to_dict, struct_of


# e.g. n is the number of if statements;
# use recur placeholder for what is recursive, and n for iteration number
def _recurse_template_internal(text, n, orig_n, end):
    if n == 0:
        return end

    n_format = orig_n - n

    return text.format(n=n_format, recur=_recurse_template_internal(text, n - 1, orig_n, end), t='\t' * n_format)


def _recursive_template(text, n, end):
    return _recurse_template_internal(text, n, n, end)


_individual_merges = {
    'inner': 'p.$2, p.$3',
    'left': 'merge(p.$2, p.$0), merge(p.$3, none_elem2)',
    'right': 'merge(p.$2, none_elem1), merge(p.$3, p.$1)',
}

_merges_less = {
    'inner': _individual_merges['inner'],
    'left': _individual_merges['left'],
    'right': _individual_merges['inner'],
}
_merges_greater = {
    'inner': _individual_merges['inner'],
    'left': _individual_merges['inner'],
    'right': _individual_merges['right'],
}

_remaining_missing = {
    'inner': '',
    'left': """let res = if (res.$0 < len1, iterate(res,
        |p|
            {
                {p.$0 + 1L, p.$1, merge(p.$2, p.$0), merge(p.$3, none_elem2)},
                p.$0 + 1L < len1
            }
), res);""",
    'right': """let res = if (res.$1 < len2, iterate(res,
        |p|
            {
                {p.$0, p.$1 + 1L, merge(p.$2, none_elem1), merge(p.$3, p.$1)},
                p.$1 + 1L < len2
            }
), res);"""
}


def _generate_checks(how, separator_index):
    checks = """if(val1.${n} == val2.${n},
                {t}{recur},
                {t}if(val1.${n} < val2.${n},  
                    {t}{{p.$0 + 1L, p.$1, merge_less}},
                    {t}{{p.$0, p.$1 + 1L, merge_greater}}
                {t})
            {t})"""
    checks = checks.replace('merge_less', _merges_less[how], 1).replace('merge_greater', _merges_greater[how], 1)

    end = '{p.$0 + 1L, p.$1 + 1L, merge(p.$2, p.$0), merge(p.$3, p.$1)}'
    checks = _recursive_template(checks, separator_index, end)

    return checks


def _weld_merge_join(vec_of_struct_self, vec_of_struct_other, separator_index, how, is_on_unique):
    weld_obj = create_empty_weld_object()
    weld_obj_id_self = get_weld_obj_id(weld_obj, vec_of_struct_self)
    weld_obj_id_other = get_weld_obj_id(weld_obj, vec_of_struct_other)

    checks = _generate_checks(how, separator_index)

    weld_template = """let len1 = len({self});
let len2 = len({other});
let none_elem1 = len1 + 1L;
let none_elem2 = len2 + 1L;
let res = iterate({{0L, 0L, appender[i64], appender[i64]}},
    |p|
        let val1 = lookup({self}, p.$0);
        let val2 = lookup({other}, p.$1);
        let iter_output = 
            {checks};
        {{
            iter_output,
            iter_output.$0 < len1 && 
            iter_output.$1 < len2
        }}
);
{remaining}
{{result(res.$2), result(res.$3)}}"""

    weld_obj.weld_code = weld_template.format(self=weld_obj_id_self,
                                              other=weld_obj_id_other,
                                              checks=checks,
                                              remaining=_remaining_missing[how])

    return weld_obj


def weld_merge_join(arrays_self, weld_types_self, arrays_other, weld_types_other,
                    how, is_on_sorted, is_on_unique, readable_text):
    """Applies merge-join on the arrays returning indices from each to keep in the resulting

    Parameters
    ----------
    arrays_self : list of (numpy.ndarray or WeldObject)
        Columns from the self DataFrame on which to join.
    weld_types_self : list of WeldType
        Corresponding Weld types.
    arrays_other : list of (numpy.ndarray or WeldObject)
        Columns from the other DataFrame on which to join.
    weld_types_other : list of WeldType
        Corresponding Weld types.
    how : {'inner', 'left', 'right'}
        Which kind of join to do.
    is_on_sorted : bool
        If we know that the on columns are already sorted, can employ faster algorithm.
    is_on_unique : bool
        If we know that the values are unique, can employ faster algorithm.
    readable_text : str
        Explanatory string to add in the Weld placeholder.

    Returns
    -------
    tuple of WeldObject
        Two columns of indices from the input arrays, indices of the rows from self and other that should be
        available in the resulting joined DataFrame.

    """
    assert is_on_unique

    weld_obj_vec_of_struct_self = weld_arrays_to_vec_of_struct(arrays_self, weld_types_self)
    weld_obj_vec_of_struct_other = weld_arrays_to_vec_of_struct(arrays_other, weld_types_other)

    weld_obj_join = _weld_merge_join(weld_obj_vec_of_struct_self,
                                     weld_obj_vec_of_struct_other,
                                     len(arrays_self),
                                     how,
                                     is_on_unique)

    intermediate_result = LazyStructOfVecResult(weld_obj_join, [WeldLong(), WeldLong()])
    dependency_name = Cache.cache_intermediate_result(intermediate_result, readable_text)

    weld_objects = extract_placeholder_weld_objects(dependency_name, 2, readable_text)

    return weld_objects


def _weld_merge_outer_join(vec_of_struct_self, vec_of_struct_other, weld_types,
                           separator_index, is_on_unique):
    weld_obj = create_empty_weld_object()
    weld_obj_id_self = get_weld_obj_id(weld_obj, vec_of_struct_self)
    weld_obj_id_other = get_weld_obj_id(weld_obj, vec_of_struct_other)

    new_index_appenders = struct_of('appender[{e}]', weld_types)
    new_index_results = struct_of('result(res.$4.${i})', weld_types)

    to_merge = struct_of('merge(p.$4.${i}, val.${i})', weld_types)
    to_merge_less = '{}, {}'.format(_merges_less['left'], to_merge.replace('val', 'val1', 1))
    to_merge_greater = '{}, {}'.format(_merges_greater['right'], to_merge.replace('val', 'val2', 1))

    checks_to_merge_less = '{}, {{{}}}'.format(_merges_less['left'], to_merge.replace('val', 'val1', 1))
    checks_to_merge_greater = '{}, {{{}}}'.format(_merges_greater['right'], to_merge.replace('val', 'val2', 1))
    checks = """if(val1.${n} == val2.${n},
                {t}{recur},
                {t}if(val1.${n} < val2.${n},  
                    {t}{{p.$0 + 1L, p.$1, to_merge_less}},
                    {t}{{p.$0, p.$1 + 1L, to_merge_greater}}
                {t})
            {t})"""
    checks = checks.replace('to_merge_less', checks_to_merge_less, 1)\
        .replace('to_merge_greater', checks_to_merge_greater, 1)
    end = '{{p.$0 + 1L, p.$1 + 1L, merge(p.$2, p.$0), merge(p.$3, p.$1), {}}}'\
        .format(to_merge.replace('val', 'val1', 1))
    checks = _recursive_template(checks, separator_index, end)

    weld_template = """let len1 = len({self});
let len2 = len({other});
let none_elem1 = len1 + 1L;
let none_elem2 = len2 + 1L;
let res = iterate({{0L, 0L, appender[i64], appender[i64], {new_index_appenders}}},
    |p|
        let val1 = lookup({self}, p.$0);
        let val2 = lookup({other}, p.$1);
        let iter_output = 
            {checks};
        {{
            iter_output,
            iter_output.$0 < len1 && 
            iter_output.$1 < len2
        }}
);
let res = if (res.$0 < len1, iterate(res,
        |p|
            let val1 = lookup({self}, p.$0);
            {{
                {{p.$0 + 1L, p.$1, {to_merge_less}}},
                p.$0 + 1L < len1
            }}
), res);
let res = if (res.$1 < len2, iterate(res,
        |p|
            let val2 = lookup({other}, p.$1);
            {{
                {{p.$0, p.$1 + 1L, {to_merge_greater}}},
                p.$1 + 1L < len2
            }}
), res);
{{result(res.$2), result(res.$3), {new_index_results}}}"""

    weld_obj.weld_code = weld_template.format(self=weld_obj_id_self,
                                              other=weld_obj_id_other,
                                              checks=checks,
                                              to_merge_less=to_merge_less,
                                              to_merge_greater=to_merge_greater,
                                              new_index_appenders=new_index_appenders,
                                              new_index_results=new_index_results)

    return weld_obj


def weld_merge_outer_join(arrays_self, weld_types_self, arrays_other, weld_types_other,
                          how, is_on_sorted, is_on_unique, readable_text):
    """Applies merge-join on the arrays returning indices from each to keep in the resulting

    Parameters
    ----------
    arrays_self : list of (numpy.ndarray or WeldObject)
        Columns from the self DataFrame on which to join.
    weld_types_self : list of WeldType
        Corresponding Weld types.
    arrays_other : list of (numpy.ndarray or WeldObject)
        Columns from the other DataFrame on which to join.
    weld_types_other : list of WeldType
        Corresponding Weld types.
    how : {'outer'}
        Here it is not used but kept to maintain same method signature as weld_merge_join
    is_on_sorted : bool
        If we know that the on columns are already sorted, can employ faster algorithm.
    is_on_unique : bool
        If we know that the values are unique, can employ faster algorithm.
    readable_text : str
        Explanatory string to add in the Weld placeholder.

    Returns
    -------
    tuple of WeldObject
        Three objects: first 2 are indices from the input arrays, indices of the rows from self and other that should be
        available in the resulting joined DataFrame. The last object will be a tuple containing the new index with
        actual values, not indices to other rows.

    """
    assert is_on_unique

    weld_obj_vec_of_struct_self = weld_arrays_to_vec_of_struct(arrays_self, weld_types_self)
    weld_obj_vec_of_struct_other = weld_arrays_to_vec_of_struct(arrays_other, weld_types_other)

    weld_obj_join = _weld_merge_outer_join(weld_obj_vec_of_struct_self,
                                           weld_obj_vec_of_struct_other,
                                           weld_types_self,
                                           len(arrays_self),
                                           is_on_unique)

    intermediate_result = LazyStructResult(weld_obj_join, [WeldVec(WeldLong()),
                                                           WeldVec(WeldLong()),
                                                           WeldStruct([WeldVec(weld_type)
                                                                       for weld_type in weld_types_self])])
    dependency_name = Cache.cache_intermediate_result(intermediate_result, readable_text)

    weld_objects_indexes = extract_placeholder_weld_objects(dependency_name, 2, readable_text)
    weld_objects_new_index = extract_placeholder_weld_objects_from_index(dependency_name,
                                                                         len(weld_types_self),
                                                                         readable_text,
                                                                         2)

    return weld_objects_indexes + [weld_objects_new_index]


def weld_align(df_index_arrays, df_index_weld_types,
               series_index_arrays, series_index_weld_types,
               series_data, series_weld_type):
    """Returns the data from the Series aligned to the DataFrame index.

    Parameters
    ----------
    df_index_arrays : list of (numpy.ndarray or WeldObject)
        The index columns as a list.
    df_index_weld_types : list of WeldType
    series_index_arrays : numpy.ndarray or WeldObject
        The index of the Series.
    series_index_weld_types : list of WeldType
    series_data : numpy.ndarray or WeldObject
        The data of the Series.
    series_weld_type : WeldType

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj_index_df = weld_arrays_to_vec_of_struct(df_index_arrays, df_index_weld_types)
    weld_obj_series_dict = weld_data_to_dict(series_index_arrays,
                                             series_index_weld_types,
                                             series_data,
                                             series_weld_type)

    weld_obj = create_empty_weld_object()
    df_index_obj_id = get_weld_obj_id(weld_obj, weld_obj_index_df)
    series_dict_obj_id = get_weld_obj_id(weld_obj, weld_obj_series_dict)

    index_type = struct_of('{e}', df_index_weld_types)
    missing_literal = default_missing_data_literal(series_weld_type)
    if series_weld_type == WeldVec(WeldChar()):
        missing_literal = get_weld_obj_id(weld_obj, missing_literal)

    weld_template = """result(
    for({df_index},
        appender[{data_type}],
        |b: appender[{data_type}], i: i64, e: {index_type}|
            if(keyexists({series_dict}, e),
                merge(b, lookup({series_dict}, e)),
                merge(b, {missing})
            )
    )
)"""

    weld_obj.weld_code = weld_template.format(series_dict=series_dict_obj_id,
                                              df_index=df_index_obj_id,
                                              index_type=index_type,
                                              data_type=series_weld_type,
                                              missing=missing_literal)

    return weld_obj
