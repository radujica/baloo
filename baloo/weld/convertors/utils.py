import sys

from ..pyweld.types import WeldVec


def to_weld_vec(weld_type, ndim):
    """Convert multi-dimensional data to WeldVec types.

    Parameters
    ----------
    weld_type : WeldType
        WeldType of data.
    ndim : int
        Number of dimensions.

    Returns
    -------
    WeldVec
        WeldVec of 1 or more dimensions.

    """
    for i in range(ndim):
        weld_type = WeldVec(weld_type)
    return weld_type


def to_shared_lib(name):
    """Return library name depending on platform.

    Parameters
    ----------
    name : str
        Name of library.

    Returns
    -------
    str
        Name of library with extension.

    """
    if sys.platform.startswith('linux'):
        return name + '.so'
    elif sys.platform.startswith('darwin'):
        return name + '.dylib'
    elif sys.platform.startswith('win'):
        return name + '.dll'
    else:
        sys.exit(1)
