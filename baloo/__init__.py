from .core import *

# TODO: handle missing values properly ~ auto conversion to np.float64?
"""
Expected values for missing data:
float32, float64: np.nan
int16, int32, int64: not supported; should be converted to float64 for np.nan
bool: False
S/bytes_: special value chosen by user, e.g. b'None'
"""

""" All baloo objects shall conform to the following:
- `__repr__` shall show the class info without any data or weld_expr
- `.data` shall contain the underlying data, be it np.ndarray or weld_expr, or dict for DataFrame
- `evaluate()` shall return a new object of the same type but with raw evaluated data within
- `__str__` shall pretty print the data with no guarantee if raw or weld_expr; e.g. for `Series` it is just the `str()`
but for `DataFrame` it's a tabulate pretty print
"""
