from ctypes import CDLL
from functools import wraps
from os import RTLD_GLOBAL

from ..weld import WeldObject


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
    """Decorator for eager functions checking input array
    and stripping away the weld_type.

    Stripping the weld_type is required to keep the same code in Series.apply and because
    Numpy functions don't (all) have kwargs. Passing weld_type to NumPy functions is unexpected
    and raises ValueError.

    Parameters
    ----------
    func : function
        Function to execute eagerly over raw data.
    func_args : kwargs
        Arguments to pass to func, if any.

    Returns
    -------
    function

    """
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
