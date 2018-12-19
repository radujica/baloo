from .cache import Cache
from .convertors import NumPyEncoder, NumPyDecoder
from .pyweld.types import WeldInt16, WeldLong, WeldFloat, WeldBit, WeldDouble, WeldInt
from .pyweld.weldobject import WeldObject

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def create_empty_weld_object():
    return WeldObject(_encoder, _decoder)


def get_weld_obj_id(weld_obj, data):
    """Helper method to update WeldObject with some data.

    Parameters
    ----------
    weld_obj : WeldObject
        WeldObject to update.
    data : numpy.ndarray or WeldObject or str
        Data for which to get an id. If str, it is a placeholder or 'str' literal.

    Returns
    -------
    str
        The id of the data, e.g. _inp0 for raw data, obj101 for WeldObject

    """
    obj_id = weld_obj.update(data)
    if isinstance(data, WeldObject):
        obj_id = data.obj_id
        weld_obj.dependencies[obj_id] = data

    return obj_id


def create_weld_object(data):
    """Helper method to create a WeldObject and update with data.

    Parameters
    ----------
    data : numpy.ndarray or WeldObject or str
        Data to include in newly created object. If str, it is a placeholder or 'str' literal.

    Returns
    -------
    (str, WeldObject)
        Object id for the data to use in the Weld code and
        the WeldObject updated with the data.

    """
    weld_obj = create_empty_weld_object()
    obj_id = get_weld_obj_id(weld_obj, data)

    return obj_id, weld_obj


def create_placeholder_weld_object(data):
    """Helper method that creates a WeldObject that evaluates to itself.

    Parameters
    ----------
    data : numpy.ndarray or WeldObject or str
        Data to wrap around. If str, it is a placeholder or 'str' literal.

    Returns
    -------
    WeldObject
        WeldObject wrapped around data.

    """
    obj_id, weld_obj = create_weld_object(data)
    weld_obj.weld_code = '{}'.format(str(obj_id))

    return weld_obj


def _extract_placeholder_weld_objects_at_index(dependency_name, length, readable_text, index):
    """Helper method that creates a WeldObject for each component of dependency.

    Parameters
    ----------
    dependency_name : str
        The name of the dependency evaluating to a tuple.
    length : int
        Number of components to create WeldObjects for
    readable_text : str
        Used when creating the placeholders in WeldObject.context.
    index : str
        Representing a tuple of ints used to select from the struct.

    Returns
    -------
    list of WeldObject

    """
    weld_objects = []

    for i in range(length):
        fake_weld_input = Cache.create_fake_array_input(dependency_name, readable_text + '_' + str(i), eval(index))
        obj_id, weld_obj = create_weld_object(fake_weld_input)
        weld_obj.weld_code = '{}'.format(obj_id)
        weld_objects.append(weld_obj)

        Cache.cache_fake_input(obj_id, fake_weld_input)

    return weld_objects


def extract_placeholder_weld_objects(dependency_name, length, readable_text):
    return _extract_placeholder_weld_objects_at_index(dependency_name, length, readable_text, '(i, )')


def extract_placeholder_weld_objects_from_index(dependency_name, length, readable_text, index):
    return _extract_placeholder_weld_objects_at_index(dependency_name, length, readable_text, '({}, i)'.format(index))


def struct_of(text, array):
    return '{{{}}}'.format(', '.join(text.format(i=i, e=elem) for i, elem in enumerate(array)))


# an attempt to avoid expensive casting
def to_weld_literal(scalar, weld_type):
    """Return scalar formatted for Weld.

    Parameters
    ----------
    scalar : {int, float, str, bool}
        Scalar data to convert to weld literal.
    weld_type : WeldType
        Desired Weld type.

    Returns
    -------
    str
        String of the scalar to use in Weld code.

    Examples
    --------
    >>> to_weld_literal(4, WeldLong())
    '4L'

    """
    try:
        if isinstance(scalar, int):
            if isinstance(weld_type, WeldInt16):
                return '{}si'.format(str(scalar))
            elif isinstance(weld_type, WeldInt):
                return str(scalar)
            elif isinstance(weld_type, WeldLong):
                return '{}L'.format(str(scalar))
            elif isinstance(weld_type, WeldFloat):
                return '{}f'.format(str(scalar))
            elif isinstance(weld_type, WeldDouble):
                return '{}.0'.format(str(scalar))
            else:
                raise TypeError()
        elif isinstance(scalar, float):
            if isinstance(weld_type, WeldFloat):
                return '{}f'.format(str(scalar))
            elif isinstance(weld_type, WeldDouble):
                return str(scalar)
            else:
                raise TypeError()
        elif isinstance(scalar, str):
            if str(weld_type) == 'vec[i8]':
                return str(scalar)
            else:
                raise TypeError()
        elif isinstance(scalar, bool):
            if isinstance(weld_type, WeldBit):
                return str(scalar).lower()
            else:
                raise TypeError()
        else:
            raise TypeError()
    except TypeError:
        raise TypeError('Cannot convert scalar:{} to type:{}'.format(scalar, weld_type))


def weld_combine_scalars(scalars, weld_type):
    """Combine column-wise aggregations (so resulting scalars) into a single array.

    Parameters
    ----------
    scalars : tuple of WeldObjects
        WeldObjects to combine.
    weld_type : WeldType
        The Weld type of the result. Currently expecting scalars to be of the same type.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj = create_empty_weld_object()
    obj_ids = (get_weld_obj_id(weld_obj, scalar) for scalar in scalars)

    merges = '\n'.join(('let res = merge(res, {});'.format(obj_id) for obj_id in obj_ids))

    weld_template = """let res = appender[{type}];
{merges}
result(res)
"""

    weld_obj.weld_code = weld_template.format(type=weld_type,
                                              merges=merges)

    return weld_obj


_numeric_types = {WeldInt16(), WeldInt(), WeldLong(), WeldFloat(), WeldDouble(), WeldBit()}


def is_numeric(weld_type):
    """Checks whether the WeldType is a numeric type.

    Parameters
    ----------
    weld_type : WeldType
        To check.

    Returns
    -------
    bool
        Whether the Weld type is numeric or not.

    """
    return weld_type in _numeric_types


def _not_possible_to_cast(scalar, to_weld_type):
    return not isinstance(scalar, (int, float, WeldObject)) or \
           isinstance(scalar, bool) or \
           not is_numeric(to_weld_type)


def weld_cast_scalar(scalar, to_weld_type):
    """Returns the scalar casted to the request Weld type.

    Parameters
    ----------
    scalar : {int, float, WeldObject}
        Input array.
    to_weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    if _not_possible_to_cast(scalar, to_weld_type):
        raise TypeError('Cannot cast scalar of type={} to type={}'.format(type(scalar), to_weld_type))

    weld_obj = create_empty_weld_object()
    if isinstance(scalar, WeldObject):
        scalar = get_weld_obj_id(weld_obj, scalar)

    weld_template = '{type}({scalar})'

    weld_obj.weld_code = weld_template.format(scalar=scalar,
                                              type=to_weld_type)

    return weld_obj


# this is fairly common so make separate method
def weld_cast_double(scalar):
    return weld_cast_scalar(scalar, WeldDouble())


def weld_cast_array(array, weld_type, to_weld_type):
    """Cast array to a different type.

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Input data.
    weld_type : WeldType
        Type of each element in the input array.
    to_weld_type : WeldType
        Desired type.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    if not is_numeric(weld_type) or not is_numeric(to_weld_type):
        raise TypeError('Cannot cast array of type={} to type={}'.format(weld_type, to_weld_type))

    obj_id, weld_obj = create_weld_object(array)

    weld_template = """map(
    {array},
    |e: {type}|
        {to}(e)
)"""

    weld_obj.weld_code = weld_template.format(array=obj_id,
                                              type=weld_type,
                                              to=to_weld_type)

    return weld_obj


# essentially switching from columns to rows ~ axis 0 to 1
def weld_arrays_to_vec_of_struct(arrays, weld_types):
    """Create a vector of structs from multiple vectors.

    Parameters
    ----------
    arrays : list of (numpy.ndarray or WeldObject)
        Arrays to put in a struct.
    weld_types : list of WeldType
        The Weld types of the arrays in the same order.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj = create_empty_weld_object()
    obj_ids = [get_weld_obj_id(weld_obj, array) for array in arrays]

    arrays = 'zip({})'.format(', '.join(obj_ids)) if len(obj_ids) > 1 else '{}'.format(obj_ids[0])
    input_types = struct_of('{e}', weld_types) if len(obj_ids) > 1 else '{}'.format(weld_types[0])
    res_types = struct_of('{e}', weld_types)
    to_merge = 'e' if len(obj_ids) > 1 else '{e}'

    weld_template = """result(
    for({arrays},
        appender[{res_types}],
        |b: appender[{res_types}], i: i64, e: {input_types}|
            merge(b, {to_merge})
    )    
)"""

    weld_obj.weld_code = weld_template.format(arrays=arrays,
                                              input_types=input_types,
                                              res_types=res_types,
                                              to_merge=to_merge)

    return weld_obj


# essentially switching from rows to columns ~ axis 1 to 0
def weld_vec_of_struct_to_struct_of_vec(vec_of_structs, weld_types):
    """Create a struct of vectors.

    Parameters
    ----------
    vec_of_structs : WeldObject
        Encoding a vector of structs.
    weld_types : list of WeldType
        The Weld types of the arrays in the same order.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(vec_of_structs)

    appenders = struct_of('appender[{e}]', weld_types)
    types = struct_of('{e}', weld_types)
    merges = struct_of('merge(b.${i}, e.${i})', weld_types)
    result = struct_of('result(vecs.${i})', weld_types)

    weld_template = """let vecs = for({vec_of_struct},
    {appenders},
    |b: {appenders}, i: i64, e: {types}|
        {merges}
);
{result}
"""

    weld_obj.weld_code = weld_template.format(vec_of_struct=obj_id,
                                              appenders=appenders,
                                              types=types,
                                              merges=merges,
                                              result=result)

    return weld_obj


def weld_select_from_struct(struct_of_vec, index_to_select):
    """Select a single vector from the struct of vectors.

    Parameters
    ----------
    struct_of_vec : WeldObject
        Encoding a struct of vectors.
    index_to_select : int
        Which vec to select from the struct.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    obj_id, weld_obj = create_weld_object(struct_of_vec)

    weld_template = '{struct}.${index}'

    weld_obj.weld_code = weld_template.format(struct=obj_id,
                                              index=index_to_select)

    return weld_obj


# TODO: support multiple values
def weld_data_to_dict(keys, keys_weld_types, values, values_weld_types):
    """Adds the key-value pairs in a dictionary. Overlapping keys are max-ed.

    Note this cannot be evaluated!

    Parameters
    ----------
    keys : list of (numpy.ndarray or WeldObject)
    keys_weld_types : list of WeldType
    values : numpy.ndarray or WeldObject
    values_weld_types : WeldType

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj_keys = weld_arrays_to_vec_of_struct(keys, keys_weld_types)

    weld_obj = create_empty_weld_object()
    keys_obj_id = get_weld_obj_id(weld_obj, weld_obj_keys)
    values_obj_id = get_weld_obj_id(weld_obj, values)

    keys_types = struct_of('{e}', keys_weld_types)
    values_types = values_weld_types

    weld_template = """result(
    for(
        zip({keys}, {values}),
        dictmerger[{keys_types}, {values_types}, max],
        |b: dictmerger[{keys_types}, {values_types}, max], i: i64, e: {{{keys_types}, {values_types}}}|
            merge(b, e)
    )
)"""

    weld_obj.weld_code = weld_template.format(keys=keys_obj_id,
                                              values=values_obj_id,
                                              keys_types=keys_types,
                                              values_types=values_types)

    return weld_obj
