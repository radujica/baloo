from .core import *

# TODO: handle missing values properly ~ auto conversion to np.float64?
"""
Expected values for missing data:
float32, float64: np.nan
int16, int32, int64: not supported; should be converted to float64 for np.nan
bool: False
S/bytes_: special value chosen by user, e.g. b'None'
"""
