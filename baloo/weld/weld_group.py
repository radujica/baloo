from .weld_utils import weld_arrays_to_vec_of_struct, create_weld_object


def weld_groupby(arrays: list, weld_types: list, by_indices):
    """Groups by the columns in by.

    Parameters
    ----------
    arrays : list of (numpy.ndarray or WeldObject)
        Entire DataFrame data.
    weld_types : list of WeldType
        Corresponding to data.
    by_indices : list of int
        Indices of which arrays to group by.

    Returns
    -------
    WeldObject
        Representation of the computation.

    """
    weld_struct = weld_arrays_to_vec_of_struct(arrays, weld_types)

    obj_id, weld_obj = create_weld_object(weld_struct)

    all_indices = list(range(len(arrays)))
    column_indices = list(filter(lambda x: x not in by_indices, all_indices))
    by_weld_types = [weld_types[i] for i in by_indices]
    column_weld_types = [weld_types[i] for i in column_indices]

    by_types = '{{{}}}'.format(', '.join(str(type_) for type_ in by_weld_types))
    column_types = '{{{}}}'.format(', '.join(str(type_) for type_ in column_weld_types))
    all_types = '{{{}}}'.format(', '.join(str(type_) for type_ in weld_types))
    by_select = '{{{}}}'.format(', '.join('e.${}'.format(str(i)) for i in by_indices))
    column_select = '{{{}}}'.format(', '.join('e.${}'.format(str(i)) for i in column_indices))

    weld_template = """tovec(
    result(
        for(
            {arrays},
            groupmerger[{by_types}, {column_types}],
            |b: groupmerger[{by_types}, {column_types}], i: i64, e: {all_types}|
                merge(b, {{{by_select}, {column_select}}})
        )
    )
)"""

    weld_obj.weld_code = weld_template.format(arrays=obj_id,
                                              by_types=by_types,
                                              column_types=column_types,
                                              all_types=all_types,
                                              by_select=by_select,
                                              column_select=column_select)

    return weld_obj


_merger_ops = {
    '+': 'merge(c.${}, f.${})',
    '*': 'merge(c.${}, f.${})',
    'min': 'merge(c.${}, f.${})',
    'max': 'merge(c.${}, f.${})',
    'mean': 'merge(c.${}, f64(f.${}) / f64(len(e.$1)))'
}

_dictmerger_operations = {'+', '*', 'min', 'max'}


def _deduce_operation(aggregation):
    if aggregation in _dictmerger_operations:
        return aggregation
    else:
        return '+'


# TODO: make it work without replace
def _assemble_computation(aggregation, column_weld_types, new_column_weld_types, operation):
    if aggregation in _merger_ops:
        template = """let group_res = for(
                        e.$1,
                        {mergers},
                        |c: {mergers}, j: i64, f: {column_types}|
                            {merger_ops}
                    );
                    merge(b, {{e.$0, {merger_res}}})"""
        mergers = '{{{}}}'.format(', '.join('merger[{}, {}]'.format(type_, operation) for type_ in new_column_weld_types))
        merger_ops = '{{{}}}'.format(', '.join(_merger_ops[aggregation].format(i, i) for i in range(len(column_weld_types))))
        merger_res = '{{{}}}'.format(', '.join('result(group_res.${})'.format(i) for i in range(len(column_weld_types))))

        return template.replace('mergers', mergers, 2)\
            .replace('merger_ops', merger_ops, 1)\
            .replace('merger_res', merger_res, 1)
    elif aggregation == 'size':
        template = 'merge(b, {{e.$0, {lengths}}})'
        lengths = '{{{}}}'.format(', '.join('len(e.$1)' for _ in range(len(column_weld_types))))

        return template.replace('lengths', lengths, 1)
    else:
        raise NotImplementedError('Oops')


def weld_groupby_aggregate(grouped_df, weld_types: list, by_indices, aggregation, result_type=None):
    """Perform aggregation on grouped data.

    Parameters
    ----------
    grouped_df : WeldObject
        DataFrame which has been grouped through weld_groupby.
    weld_types : list of WeldType
        Corresponding to data.
    by_indices : list of int
        Indices of which arrays to group by.
    aggregation : {'+', '*', 'min', 'max', 'mean'}
        What operation to apply to grouped rows.
    result_type : WeldType, optional
        Whether the result shall be (casted to) some specific type.

    Returns
    -------
    (list of WeldType, WeldObject)
        Tuple of newly ordered Weld types and the WeldObject representation of the computation.

    """
    obj_id, weld_obj = create_weld_object(grouped_df)

    operation = _deduce_operation(aggregation)

    all_indices = list(range(len(weld_types)))
    column_indices = list(filter(lambda x: x not in by_indices, all_indices))
    by_weld_types = [weld_types[i] for i in by_indices]
    column_weld_types = [weld_types[i] for i in column_indices]
    new_column_weld_types = column_weld_types if result_type is None else [result_type for _ in column_weld_types]

    # TODO: generalize this ', '.join stuff
    by_types = '{{{}}}'.format(', '.join(str(type_) for type_ in by_weld_types))
    column_types = '{{{}}}'.format(', '.join(str(type_) for type_ in column_weld_types))
    new_column_types = '{{{}}}'.format(', '.join(str(type_) for type_ in new_column_weld_types))
    grouped_df_types = '{{{}, {}}}'.format(by_types, column_types)
    res = '{{{}}}'.format(', '.join(['e.$0.${}'.format(i) for i in range(len(by_weld_types))] +
                                    ['e.$1.${}'.format(i) for i in range(len(column_weld_types))]))

    weld_template = """map(
    tovec(
        result(
            for(
                {grouped_df},
                dictmerger[{by_types}, {new_column_types}, {operation}],
                |b: dictmerger[{by_types}, {new_column_types}, {operation}], i: i64, e: {{{by_types}, vec[{column_types}]}}|
                    {computation}
            )
        )
    ),
    |e: {{{by_types}, {new_column_types}}}|
        {res}
)"""

    weld_template = weld_template.replace('{computation}',
                                          _assemble_computation(aggregation,
                                                                column_weld_types,
                                                                new_column_weld_types,
                                                                operation),
                                          1)

    weld_obj.weld_code = weld_template.format(grouped_df=obj_id,
                                              by_types=by_types,
                                              column_types=column_types,
                                              new_column_types=new_column_types,
                                              operation=operation,
                                              grouped_df_types=grouped_df_types,
                                              res=res)

    return by_weld_types + new_column_weld_types, weld_obj
