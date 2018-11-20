from ctypes import CDLL
from functools import wraps
from os import RTLD_GLOBAL

from weld.weldobject import WeldObject


def load_cudf(path_to_so):
    """Dynamically load a C UDF.

    Parameters
    ----------
    path_to_so : str
        Absolute path to so.

    Returns
    -------

    """
    CDLL(path_to_so, mode=RTLD_GLOBAL)


def raw(func, **func_args):
    if len(func_args) == 0:
        @wraps(func)
        def wrapper(array, **kwargs):
            if isinstance(array, WeldObject):
                raise TypeError('Can only perform operation on raw data')
            # need to not pass weld_type to whatever function
            if 'weld_type' in kwargs:
                del kwargs['weld_type']
            return func(array, **kwargs)
        return wrapper
    else:
        # here kwargs is only kept s.t. Series can still pass the weld_type
        @wraps(func)
        def wrapper(array, **kwargs):
            if isinstance(array, WeldObject):
                raise TypeError('Can only perform operation on raw data')
            return func(array, **func_args)
        return wrapper
