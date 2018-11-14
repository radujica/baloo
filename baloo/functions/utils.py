from ctypes import CDLL
from os import RTLD_GLOBAL


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
