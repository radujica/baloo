from .cache import Cache
from .lazy_result import LazyStructOfVecResult, WeldLong
from .weld_utils import weld_arrays_to_vec_of_struct, create_empty_weld_object, get_weld_obj_id, \
    extract_placeholder_weld_objects


# e.g. n is the number of if statements;
# use recur placeholder for what is recursive, and n for iteration number
def _recurse_template_internal(text, n, orig_n, end):
    if n == 0:
        return end

    n_format = orig_n - n

    return text.format(n=n_format, recur=_recurse_template_internal(text, n - 1, orig_n, end), t='\t' * n_format)


def _recursive_template(text, n, end):
    return _recurse_template_internal(text, n, n, end)


def _weld_merge_join_inner(vec_of_struct_self, vec_of_struct_other, separator_index, is_on_unique):
    weld_obj = create_empty_weld_object()
    weld_obj_id_self = get_weld_obj_id(weld_obj, vec_of_struct_self)
    weld_obj_id_other = get_weld_obj_id(weld_obj, vec_of_struct_other)

    checks = """if(val1.${n} == val2.${n},
    {t}{recur},
    {t}if(val1.${n} < val2.${n},  
        {t}{{p.$0 + 1L, p.$1, p.$2, p.$3}},
        {t}{{p.$0, p.$1 + 1L, p.$2, p.$3}}
    {t})
{t})"""

    end = '{p.$0 + 1L, p.$1 + 1L, merge(p.$2, p.$0), merge(p.$3, p.$1)}'
    checks = _recursive_template(checks, separator_index, end)

    weld_template = """let len1 = len({self});
let len2 = len({other});
let res = iterate({{0L, 0L, appender[i64], appender[i64]}},
    |p|
        let val1 = lookup({self}, p.$0);
        let val2 = lookup({other}, p.$1);
        let iter_output = 
            {checks}
        {{
            iter_output,
            iter_output.$0 < len1 && 
            iter_output.$1 < len2
        }}
);
{{result(res.$2), result(res.$3)}}"""

    weld_obj.weld_code = weld_template.format(self=weld_obj_id_self,
                                              other=weld_obj_id_other,
                                              checks=checks + ';')

    return weld_obj


def weld_merge_join(arrays_self, weld_types_self, arrays_other, weld_types_other,
                    how, is_on_sorted, is_on_unique, readable_text):
    """Applies merge-join on the arrays returning indices from each to keep in the resulting

    Parameters
    ----------
    arrays_self : list of numpy.ndarray or WeldObject
        Columns from the self DataFrame on which to join.
    weld_types_self : list of WeldType
        Corresponding Weld types.
    arrays_other : list of numpy.ndarray or WeldObject
        Columns from the other DataFrame on which to join.
    weld_types_other : list of WeldType
        Corresponding Weld types.
    how : {'inner', 'left', 'right', 'outer'}
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
    assert how == 'inner'
    assert is_on_sorted
    assert is_on_unique

    weld_obj_vec_of_struct_self = weld_arrays_to_vec_of_struct(arrays_self, weld_types_self)
    weld_obj_vec_of_struct_other = weld_arrays_to_vec_of_struct(arrays_other, weld_types_other)

    if how == 'inner':
        weld_obj_join = _weld_merge_join_inner(weld_obj_vec_of_struct_self,
                                               weld_obj_vec_of_struct_other,
                                               len(arrays_self),
                                               is_on_unique)
    else:
        raise NotImplementedError()

    intermediate_result = LazyStructOfVecResult(weld_obj_join, [WeldLong(), WeldLong()])
    dependency_name = Cache.cache_intermediate_result(intermediate_result, readable_text)

    weld_objects = extract_placeholder_weld_objects(dependency_name, 2, readable_text)

    return weld_objects
