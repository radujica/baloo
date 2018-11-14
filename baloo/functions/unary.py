from ..weld import create_weld_object, WeldDouble, WeldFloat


def _weld_unary(array, weld_type, operation):
    """Apply operation on each element in the array.

    As mentioned by Weld, the operations follow the behavior of the equivalent C functions from math.h

    Parameters
    ----------
    array : numpy.ndarray or WeldObject
        Data
    weld_type : WeldType
        Of the data
    operation : {'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'erf'}
        Which unary operation to apply.

    Returns
    -------
    WeldObject
        Representation of this computation.

    """
    if weld_type not in {WeldFloat(), WeldDouble()}:
        raise TypeError('Unary operation supported only on scalar f32 or f64')

    obj_id, weld_obj = create_weld_object(array)
    weld_template = 'map({array}, |e: {type}| {op}(e))'
    weld_obj.weld_code = weld_template.format(array=obj_id, type=weld_type, op=operation)

    return weld_obj


def exp(array, weld_type):
    return _weld_unary(array, weld_type, 'exp')


def log(array, weld_type):
    return _weld_unary(array, weld_type, 'log')


def sqrt(array, weld_type):
    return _weld_unary(array, weld_type, 'sqrt')


def sin(array, weld_type):
    return _weld_unary(array, weld_type, 'sin')


def cos(array, weld_type):
    return _weld_unary(array, weld_type, 'cos')


def tan(array, weld_type):
    return _weld_unary(array, weld_type, 'tan')


def asin(array, weld_type):
    return _weld_unary(array, weld_type, 'asin')


def acos(array, weld_type):
    return _weld_unary(array, weld_type, 'acos')


def atan(array, weld_type):
    return _weld_unary(array, weld_type, 'atan')


def sinh(array, weld_type):
    return _weld_unary(array, weld_type, 'sinh')


def cosh(array, weld_type):
    return _weld_unary(array, weld_type, 'cosh')


def tanh(array, weld_type):
    return _weld_unary(array, weld_type, 'tanh')


def erf(array, weld_type):
    return _weld_unary(array, weld_type, 'erf')
