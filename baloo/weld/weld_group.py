from .weld_utils import weld_arrays_to_vec_of_struct, create_weld_object


def weld_groupby_aggregate_dictmerger(arrays: list, weld_types: list, by_indices: list, operation):
    """Groups by the columns in by.

    Parameters
    ----------
    arrays : list of (numpy.ndarray or WeldObject)
        Entire DataFrame data.
    weld_types : list of WeldType
        Corresponding to data.
    by_indices : list of int
        Indices of which arrays to group by.
    operation : {'+', '*', 'min', 'max'}
        Aggregation.

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
    res = '{{{}}}'.format(', '.join(['e.$0.${}'.format(i) for i in range(len(by_weld_types))] +
                                    ['e.$1.${}'.format(i) for i in range(len(column_weld_types))]))

    weld_template = """map(
    tovec(
        result(
            for(
                {arrays},
                dictmerger[{by_types}, {column_types}, {operation}],
                |b: dictmerger[{by_types}, {column_types}, {operation}], i: i64, e: {all_types}|
                    merge(b, {{{by_select}, {column_select}}})
            )
        )
    ),
    |e: {{{by_types}, {column_types}}}|
        {res}
)"""

    weld_obj.weld_code = weld_template.format(arrays=obj_id,
                                              by_types=by_types,
                                              column_types=column_types,
                                              all_types=all_types,
                                              by_select=by_select,
                                              column_select=column_select,
                                              operation=operation,
                                              res=res)

    return by_weld_types + column_weld_types, weld_obj


def weld_groupby(arrays: list, weld_types: list, by_indices: list):
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


_dictmerger_ops = {'+', '*', 'min', 'max'}


def _deduce_operation(aggregation):
    if aggregation in _dictmerger_ops:
        return aggregation
    else:
        return '+'


# all _assemble_aggregation_* end with let group = {<scalars>};
def _assemble_aggregation_simple(column_weld_types, aggregation):
    template = """let sums = for(
                        e.$1,
                        {mergers},
                        |c: {mergers}, j: i64, f: {column_types}|
                            {merger_ops}
                    );
                    let group = {sums_res};"""
    mergers = '{{{}}}'.format(', '.join('merger[{}, {}]'.format(type_, aggregation) for type_ in column_weld_types))
    merger_ops = '{{{}}}'.format(', '.join('merge(c.${i}, f.${i})'.format(i=i) for i in range(len(column_weld_types))))
    sums_res = '{{{}}}'.format(', '.join('result(sums.${})'.format(i) for i in range(len(column_weld_types))))

    return template.replace('mergers', mergers, 2) \
        .replace('merger_ops', merger_ops, 1) \
        .replace('sums_res', sums_res, 1)


def _assemble_aggregation_size(column_weld_types):
    template = """let group = {lengths};"""
    lengths = '{{{}}}'.format(', '.join('len(e.$1)' for _ in range(len(column_weld_types))))

    return template.replace('lengths', lengths, 1)


def _assemble_aggregation_mean(column_weld_types, new_column_weld_types):
    template = """sums
                    let length = len(e.$1);
                    let group = {means_res};"""
    sums = _assemble_aggregation('+', column_weld_types, new_column_weld_types)
    means_res = '{{{}}}'.format(', '.join('f64(group.${}) / f64(length)'.format(i) for i in range(len(column_weld_types))))

    return template.replace('sums', sums, 2) \
        .replace('means_res', means_res, 1)


def _assemble_aggregation_var(column_weld_types, new_column_weld_types):
    template = """means
                    let sqdevs = for(
                        e.$1,
                        {mergers},
                        |c: {mergers}, j: i64, f: {column_types}|
                            {merger_ops}
                    );
                    let group = {sqdevs_res};"""
    means = _assemble_aggregation('mean', column_weld_types, new_column_weld_types)
    mergers = '{{{}}}'.format(', '.join('merger[{}, +]'.format(type_) for type_ in new_column_weld_types))
    merger_ops = '{{{}}}'.format(', '.join('merge(c.${i}, pow(f64(f.${i}) - group.${i}, 2.0))'.format(i=i) for i in range(len(column_weld_types))))
    sqdevs_res = '{{{}}}'.format(', '.join('result(sqdevs.${})'.format(i) for i in range(len(column_weld_types))))

    return template.replace('means', means, 1) \
        .replace('mergers', mergers, 2) \
        .replace('merger_ops', merger_ops, 1) \
        .replace('sqdevs_res', sqdevs_res, 1)


def _assemble_aggregation_std(column_weld_types, new_column_weld_types):
    template = """vars
                    let group = {vars_res};"""
    vars_ = _assemble_aggregation('var', column_weld_types, new_column_weld_types)
    vars_res = '{{{}}}'.format(', '.join('sqrt(group.${})'.format(i) for i in range(len(column_weld_types))))

    return template.replace('vars', vars_, 1)\
        .replace('vars_res', vars_res, 1)


# TODO: this could be a dict if all functions accepted the same params
def _assemble_aggregation(aggregation, column_weld_types, new_column_weld_types):
    if aggregation in _dictmerger_ops:
        return _assemble_aggregation_simple(column_weld_types, aggregation)
    elif aggregation == 'size':
        return _assemble_aggregation_size(column_weld_types)
    elif aggregation == 'mean':
        return _assemble_aggregation_mean(column_weld_types, new_column_weld_types)
    elif aggregation == 'var':
        return _assemble_aggregation_var(column_weld_types, new_column_weld_types)
    elif aggregation == 'std':
        return _assemble_aggregation_std(column_weld_types, new_column_weld_types)
    else:
        raise NotImplementedError('Oops')


# TODO: make it work without replace
def _assemble_computation(aggregation, column_weld_types, new_column_weld_types):
    template = """aggregation
                    merge(b, {{e.$0, {group_res}}})"""

    aggregation_template = _assemble_aggregation(aggregation, column_weld_types, new_column_weld_types)
    group_res = '{{{}}}'.format(', '.join('group.${}'.format(i) for i in range(len(column_weld_types))))

    return template.replace('aggregation', aggregation_template, 1)\
        .replace('group_res', group_res, 1)


def weld_groupby_aggregate(grouped_df, weld_types: list, by_indices: list, aggregation, result_type=None):
    """Perform aggregation on grouped data.

    Parameters
    ----------
    grouped_df : WeldObject
        DataFrame which has been grouped through weld_groupby.
    weld_types : list of WeldType
        Corresponding to data.
    by_indices : list of int
        Indices of which arrays to group by.
    aggregation : {'+', '*', 'min', 'max', 'mean', 'var', 'std'}
        What operation to apply to grouped rows.
    result_type : WeldType, optional
        Whether the result shall be (casted to) some specific type.

    Returns
    -------
    (list of WeldType, WeldObject)
        Tuple of newly ordered Weld types and the WeldObject representation of the computation.

    """
    obj_id, weld_obj = create_weld_object(grouped_df)

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
                dictmerger[{by_types}, {new_column_types}, +],
                |b: dictmerger[{by_types}, {new_column_types}, +], i: i64, e: {{{by_types}, vec[{column_types}]}}|
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
                                                                new_column_weld_types),
                                          1)

    weld_obj.weld_code = weld_template.format(grouped_df=obj_id,
                                              by_types=by_types,
                                              column_types=column_types,
                                              new_column_types=new_column_types,
                                              grouped_df_types=grouped_df_types,
                                              res=res)

    return by_weld_types + new_column_weld_types, weld_obj
