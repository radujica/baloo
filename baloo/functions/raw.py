import numpy as np

from .utils import raw


@raw
def sort(array, **kwargs):
    return np.sort(array, **kwargs)
