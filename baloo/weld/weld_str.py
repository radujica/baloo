from .convertors import default_missing_data_literal
from .pyweld import WeldLong, WeldVec, WeldChar
from .weld_utils import create_weld_object, to_weld_literal, get_weld_obj_id


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
                        if(f > 64c && f < 91c,
                            merge(c, f + 32c),
                            merge(c, f)
                        )
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


# TODO: check & allow negative step ~ requires Weld fix
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


def weld_str_contains(array, pat):
    """Check which elements contain pat.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    pat : str
        To check for.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    pat_id = get_weld_obj_id(weld_obj, pat)

    weld_template = """let lenPat = len({pat});
map({array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenPat > lenString,
            false,
            # start by assuming pat is not found, until proven it is
            let words_iter_res = iterate({{0L, false}}, 
                |p| 
                    let e_i = p.$0;
                    let pat_i = 0L;
                    # start by assuming the substring and pat are the same, until proven otherwise
                    let word_check_res = iterate({{e_i, pat_i, true}}, 
                        |q| 
                            let found = lookup(e, q.$0) == lookup({pat}, q.$1);
                            {{
                                {{q.$0 + 1L, q.$1 + 1L, found}}, 
                                q.$1 + 1L < lenPat &&
                                found == true
                            }}
                    ).$2;
                    {{
                        {{p.$0 + 1L, word_check_res}}, 
                        p.$0 + lenPat < lenString &&
                        word_check_res == false
                    }}
            ).$1;
            words_iter_res
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              pat=pat_id)

    return weld_obj


def weld_str_startswith(array, pat):
    """Check which elements start with pattern.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    pat : str
        To check for.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    pat_id = get_weld_obj_id(weld_obj, pat)

    """alternative implementation for reference
    let res = result(
        for(zip(slice(e, 0L, lenPat), {pat}),
            merger[i64, +],
            |b: merger[i64, +], i: i64, e: {{i8, i8}}|
                if(e.$0 == e.$1, 
                    merge(b, 1L), 
                    merge(b, 0L)
                )
        )
    );
    res == lenPat
    """

    weld_template = """let lenPat = len({pat});
map({array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenPat > lenString,
            false,
            iterate({{0L, true}}, 
                |q| 
                    let found = lookup(e, q.$0) == lookup({pat}, q.$0);
                    {{
                        {{q.$0 + 1L, found}}, 
                        q.$0 + 1L < lenPat &&
                        found == true
                    }}
            ).$1
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              pat=pat_id)

    return weld_obj


def weld_str_endswith(array, pat):
    """Check which elements end with pattern.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    pat : str
        To check for.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    pat_id = get_weld_obj_id(weld_obj, pat)

    weld_template = """let lenPat = len({pat});
map({array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenPat > lenString,
            false,
            iterate({{lenString - lenPat, 0L, true}}, 
                |q| 
                    let found = lookup(e, q.$0) == lookup({pat}, q.$1);
                    {{
                        {{q.$0 + 1L, q.$1 + 1L, found}}, 
                        q.$1 + 1L < lenPat &&
                        found == true
                    }}
            ).$2
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              pat=pat_id)

    return weld_obj


def weld_str_find(array, sub, start, end):
    """Return index of sub in elements if found, else -1.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    sub : str
        To check for.
    start : int
        Start index for searching.
    end : int or None
        Stop index for searching.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    sub_id = get_weld_obj_id(weld_obj, sub)

    if end is None:
        end = 'len(e)'
    else:
        end = to_weld_literal(end, WeldLong())

    start = to_weld_literal(start, WeldLong())

    # TODO: maybe be more friendly and fix end >= len(e) to be len(e) - 1?
    weld_template = """let lenSub = len({sub});
map({array},
    |e: vec[i8]|
        let start = {start};
        let size = {end} - start;
        let string = slice(e, start, size);
        let lenString = len(string);
        if(lenSub > lenString,
            -1L,
            # start by assuming sub is not found, until proven it is
            let words_iter_res = iterate({{0L, false}}, 
                |p| 
                    let e_i = p.$0;
                    let pat_i = 0L;
                    # start by assuming the substring and sub are the same, until proven otherwise
                    let word_check_res = iterate({{e_i, pat_i, true}}, 
                        |q| 
                            let found = lookup(string, q.$0) == lookup({sub}, q.$1);
                            {{
                                {{q.$0 + 1L, q.$1 + 1L, found}}, 
                                q.$1 + 1L < lenSub &&
                                found == true
                            }}
                    ).$2;
                    {{
                        {{p.$0 + 1L, word_check_res}}, 
                        p.$0 + lenSub < lenString &&
                        word_check_res == false
                    }}
            );
            if(words_iter_res.$1 == true,
                words_iter_res.$0 - 1L + start,
                -1L
            )
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              sub=sub_id,
                                              start=start,
                                              end=end)

    return weld_obj


def weld_str_replace(array, pat, rep):
    """Replace first occurrence of pat with rep.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    pat : str
        To find.
    rep : str
        To replace with.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    pat_id = get_weld_obj_id(weld_obj, pat)
    rep_id = get_weld_obj_id(weld_obj, rep)

    weld_template = """let lenPat = len({pat});
map({array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenPat > lenString,
            e,
            # start by assuming sub is not found, until proven it is
            let words_iter_res = iterate({{0L, false}}, 
                |p| 
                    let e_i = p.$0;
                    let pat_i = 0L;
                    # start by assuming the substring and sub are the same, until proven otherwise
                    let word_check_res = iterate({{e_i, pat_i, true}}, 
                        |q| 
                            let found = lookup(e, q.$0) == lookup({pat}, q.$1);
                            {{
                                {{q.$0 + 1L, q.$1 + 1L, found}}, 
                                q.$1 + 1L < lenPat &&
                                found == true
                            }}
                    ).$2;
                    {{
                        {{p.$0 + 1L, word_check_res}}, 
                        p.$0 + lenPat < lenString &&
                        word_check_res == false
                    }}
            );
            if(words_iter_res.$1 == true,
                let rep_from = words_iter_res.$0 - 1L;
                let rep_to = rep_from + lenPat;
                let res = appender[i8];
                let res = for(slice(e, 0L, rep_from),
                    res,
                    |c: appender[i8], j: i64, f: i8|
                        merge(c, f)                    
                );
                let res = for({rep},
                    res,
                    |c: appender[i8], j: i64, f: i8|
                        merge(c, f)                    
                );
                let res = for(slice(e, rep_to, lenString),
                    res,
                    |c: appender[i8], j: i64, f: i8|
                        merge(c, f)                    
                );
                result(res),
                e
            )
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              pat=pat_id,
                                              rep=rep_id)

    return weld_obj


def weld_str_split(array, pat, side):
    """Split on pat and return side.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    pat : str
        To find.
    side : {0, 1}
        Which side to return, with 0 being left and 1 being right

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(array)
    pat_id = get_weld_obj_id(weld_obj, pat)

    left_side_template = """let pat_start_index = words_iter_res.$0 - 1L;
result(
    for(slice(e, 0L, pat_start_index),
        appender[i8],
        |c: appender[i8], j: i64, f: i8|
            merge(c, f)   
    )                 
)"""
    right_side_template = """let start_index = words_iter_res.$0 - 1L + lenPat;
result(
    for(slice(e, start_index, lenString),
        appender[i8],
        |c: appender[i8], j: i64, f: i8|
            merge(c, f)   
    )                 
)"""

    weld_template = """let lenPat = len({pat});
map({array},
    |e: vec[i8]|
        let lenString = len(e);
        if(lenPat > lenString,
            e,
            # start by assuming sub is not found, until proven it is
            let words_iter_res = iterate({{0L, false}}, 
                |p| 
                    let e_i = p.$0;
                    let pat_i = 0L;
                    # start by assuming the substring and sub are the same, until proven otherwise
                    let word_check_res = iterate({{e_i, pat_i, true}}, 
                        |q| 
                            let found = lookup(e, q.$0) == lookup({pat}, q.$1);
                            {{
                                {{q.$0 + 1L, q.$1 + 1L, found}}, 
                                q.$1 + 1L < lenPat &&
                                found == true
                            }}
                    ).$2;
                    {{
                        {{p.$0 + 1L, word_check_res}}, 
                        p.$0 + lenPat < lenString &&
                        word_check_res == false
                    }}
            );
            if(words_iter_res.$1 == true,
                {side},
                e
            )
        )
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              pat=pat_id,
                                              side=left_side_template if side == 0 else right_side_template)

    return weld_obj
