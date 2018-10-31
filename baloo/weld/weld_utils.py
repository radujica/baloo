from weld.types import WeldInt16, WeldLong, WeldFloat, WeldBit, WeldDouble
from weld.weldobject import WeldObject

from .cache import Cache
from .convertors import NumPyEncoder, NumPyDecoder

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
    data : numpy.ndarray or WeldObject
        Data for which to get an id.

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
        Data to include in newly created object. If str, it is a placeholder.

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
    data : numpy.ndarray or WeldObject
        Data to wrap around.

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


# an attempt to avoid expensive casting
def to_weld_literal(scalar, weld_type):
    """Return scalar formatted for Weld.

    Parameters
    ----------
    scalar : {int, float, str, bool, bytes}
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
    if isinstance(weld_type, WeldInt16):
        return '{}si'.format(str(scalar))
    elif isinstance(weld_type, WeldLong):
        return '{}L'.format(str(scalar))
    elif isinstance(weld_type, WeldFloat):
        return '{}f'.format(str(scalar))
    elif isinstance(weld_type, WeldBit):
        return '{}'.format(str(scalar).lower())
    elif isinstance(weld_type, WeldDouble) and isinstance(scalar, int):
        return '{}.0'.format(str(scalar))
    else:
        return '{}'.format(str(scalar))


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


def weld_cast_scalar(scalar, weld_type):
    """Returns the scalar casted to the request Weld type.

    Parameters
    ----------
    scalar : {int, float, str, bool, bytes, WeldObject}
        Input array.
    weld_type : WeldType
        Type of each element in the input array.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    weld_obj = create_empty_weld_object()
    if isinstance(scalar, WeldObject):
        scalar = get_weld_obj_id(weld_obj, scalar)

    weld_template = '{type}({scalar})'

    weld_obj.weld_code = weld_template.format(scalar=scalar,
                                              type=weld_type)

    return weld_obj


# this is fairly common so make separate method
def weld_cast_double(scalar):
    return weld_cast_scalar(scalar, WeldDouble())


# essentially switching from columns to rows ~ axis 0 to 1
def weld_arrays_to_vec_of_struct(arrays, weld_types):
    """Create a vector of structs from multiple vectors.

    Parameters
    ----------
    arrays : list of numpy.ndarray or WeldObject
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
    input_types = '{{{}}}'.format(', '.join((str(weld_type) for weld_type in weld_types))) \
        if len(obj_ids) > 1 else '{}'.format(weld_types[0])
    res_types = '{{{}}}'.format(', '.join((str(weld_type) for weld_type in weld_types)))
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

    appenders = '{{{}}}'.format(', '.join(('appender[{}]'.format(str(weld_type)) for weld_type in weld_types)))
    types = '{{{}}}'.format(', '.join((str(weld_type) for weld_type in weld_types)))
    merges = '{{{}}}'.format(', '.join(('merge(b.${i}, e.${i})'.format(i=i) for i in range(len(weld_types)))))
    result = '{{{}}}'.format(', '.join(('result(vecs.${})'.format(i) for i in range(len(weld_types)))))

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
