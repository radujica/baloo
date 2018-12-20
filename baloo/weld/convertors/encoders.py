import ctypes

import numpy as np

from .utils import to_weld_vec
from ..pyweld.types import *
from ..pyweld.weldobject import WeldObjectDecoder, WeldObjectEncoder, cweld
from ...config import ENCODERS_PATH

# Python3: str _is_ unicode -> 'Bürgermeister'.encode() => b'B\xc3\xbcrgermeister'
# Python2: str is ascii -> 'Bürgermeister' does not exist; u'Bürgermeister'.encode() => 'B\xc3\xbcrgermeister'

supported_dtype_chars = {'h', 'i', 'l', 'f', 'd', '?', 'S'}

# TODO: datetime support
_numpy_to_weld_type_mapping = {
    'S': WeldVec(WeldChar()),
    'h': WeldInt16(),
    'i': WeldInt(),
    'l': WeldLong(),
    'f': WeldFloat(),
    'd': WeldDouble(),
    '?': WeldBit()
}


def numpy_to_weld_type(np_dtype):
    """Convert from NumPy dtype to Weld type.

    Note that support for strings is intended to be only for
    Python 2 str and Python 3 bytes. No unicode.

    Parameters
    ----------
    np_dtype : numpy.dtype or str
        NumPy dtype.

    Returns
    -------
    WeldType
        Corresponding WeldType.

    Examples
    --------
    >>> import numpy as np
    >>> from baloo.weld import numpy_to_weld_type
    >>> str(numpy_to_weld_type(np.dtype(np.int64)))
    'i64'
    >>> str(numpy_to_weld_type('?'))
    'bool'

    """
    if not isinstance(np_dtype, (str, bytes, np.dtype, type)):
        raise TypeError('Can only convert np.dtype or str')

    if isinstance(np_dtype, (str, bytes, type)):
        np_dtype = np.dtype(np_dtype)

    return _numpy_to_weld_type_mapping[np_dtype.char]


_weld_to_numpy_type_mapping = {
    WeldVec(WeldChar()): 'S',
    WeldInt16(): 'h',
    WeldInt(): 'i',
    WeldLong(): 'l',
    WeldFloat(): 'f',
    WeldDouble(): 'd',
    WeldBit(): '?'
}


def weld_to_numpy_dtype(weld_type):
    """Convert from Weld type to NumPy dtype.

    Note that support for strings is intended to be only for
    Python 2 str and Python 3 bytes. No unicode.

    Parameters
    ----------
    weld_type : WeldType
        Weld type.

    Returns
    -------
    numpy.dtype
        Corresponding Numpy dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from baloo.weld import weld_to_numpy_dtype, WeldFloat
    >>> weld_to_numpy_dtype(WeldFloat())
    dtype('float32')

    """
    return np.dtype(_weld_to_numpy_type_mapping[weld_type])


# TODO: make np.nan work?
_default_missing_mapping = {
    WeldVec(WeldChar()): 'None',
    WeldInt16(): '-999si',
    WeldInt(): '-999',
    WeldLong(): '-999L',
    WeldFloat(): '-999f',
    WeldDouble(): '-999.0',
    WeldBit(): 'false'
}


def default_missing_data_literal(weld_type):
    """Convert from Weld type to missing literal placeholder.

    Parameters
    ----------
    weld_type : WeldType
        Weld type.

    Returns
    -------
    str
        Literal for missing data.

    Examples
    --------
    >>> import numpy as np
    >>> from baloo.weld import default_missing_data_literal, WeldDouble
    >>> default_missing_data_literal(WeldDouble())
    '-999.0'

    """
    return _default_missing_mapping[weld_type]


class NumPyEncoder(WeldObjectEncoder):
    def __init__(self):
        self.utils = ctypes.PyDLL(ENCODERS_PATH)

    def py_to_weld_type(self, obj):
        if isinstance(obj, np.ndarray):
            base = numpy_to_weld_type(obj.dtype)
            base = to_weld_vec(base, obj.ndim)

            return base
        elif isinstance(obj, str):
            return WeldVec(WeldChar())
        else:
            raise TypeError('Unable to infer weld type from obj of type={}'.format(str(type(obj))))

    def _numpy_to_weld_func(self, obj):
        if obj.ndim == 1:
            if obj.dtype == 'int16':
                numpy_to_weld = self.utils.numpy_to_weld_int16_arr
            elif obj.dtype == 'int32':
                numpy_to_weld = self.utils.numpy_to_weld_int_arr
            elif obj.dtype == 'int64':
                numpy_to_weld = self.utils.numpy_to_weld_long_arr
            elif obj.dtype == 'float32':
                numpy_to_weld = self.utils.numpy_to_weld_float_arr
            elif obj.dtype == 'float64':
                numpy_to_weld = self.utils.numpy_to_weld_double_arr
            elif obj.dtype == 'bool':
                numpy_to_weld = self.utils.numpy_to_weld_bool_arr
            elif obj.dtype.char == 'S':
                numpy_to_weld = self.utils.numpy_to_weld_char_arr_arr
            else:
                raise TypeError('Unable to encode np.ndarray of 1 dimension with dtype={}'.format(str(obj.dtype)))
        elif obj.ndim == 2:
            if obj.dtype == 'int16':
                numpy_to_weld = self.utils.numpy_to_weld_int16_arr_arr
            elif obj.dtype == 'int32':
                numpy_to_weld = self.utils.numpy_to_weld_int_arr_arr
            elif obj.dtype == 'int64':
                numpy_to_weld = self.utils.numpy_to_weld_long_arr_arr
            elif obj.dtype == 'float32':
                numpy_to_weld = self.utils.numpy_to_weld_float_arr_arr
            elif obj.dtype == 'float64':
                numpy_to_weld = self.utils.numpy_to_weld_double_arr_arr
            elif obj.dtype == 'bool':
                numpy_to_weld = self.utils.numpy_to_weld_bool_arr_arr
            else:
                raise TypeError('Unable to encode np.ndarray of 2 dimensions with dtype={}'.format(str(obj.dtype)))
        else:
            raise ValueError('Can only encode np.ndarray of 1 or 2 dimensions')

        return numpy_to_weld

    def encode(self, obj):
        if isinstance(obj, np.ndarray):
            numpy_to_weld = self._numpy_to_weld_func(obj)
            numpy_to_weld.restype = self.py_to_weld_type(obj).ctype_class
            numpy_to_weld.argtypes = [py_object]

            return numpy_to_weld(obj)
        elif isinstance(obj, str):
            numpy_to_weld = self.utils.str_to_weld_char_arr
            numpy_to_weld.restype = WeldVec(WeldChar()).ctype_class
            numpy_to_weld.argtypes = [py_object]

            return numpy_to_weld(obj.encode('ascii'))
        else:
            raise TypeError('Unable to encode obj of type={}'.format(str(type(obj))))


class NumPyDecoder(WeldObjectDecoder):
    def __init__(self):
        self.utils = ctypes.PyDLL(ENCODERS_PATH)

    def _try_decode_scalar(self, data, restype):
        if restype == WeldInt16():
            result = ctypes.cast(data, ctypes.POINTER(c_int16)).contents.value
            return np.int16(result)
        elif restype == WeldInt():
            result = ctypes.cast(data, ctypes.POINTER(c_int)).contents.value
            return np.int32(result)
        elif restype == WeldLong():
            result = ctypes.cast(data, ctypes.POINTER(c_long)).contents.value
            return np.int64(result)
        elif restype == WeldFloat():
            result = ctypes.cast(data, ctypes.POINTER(c_float)).contents.value
            return np.float32(result)
        elif restype == WeldDouble():
            result = ctypes.cast(data, ctypes.POINTER(c_double)).contents.value
            return np.float64(result)
        elif restype == WeldBit():
            result = ctypes.cast(data, ctypes.POINTER(c_bool)).contents.value
            return np.bool(result)
        elif restype == WeldVec(WeldChar()):
            weld_to_numpy = self.utils.weld_to_str
            weld_to_numpy.restype = py_object
            weld_to_numpy.argtypes = [restype.ctype_class]
            result = ctypes.cast(data, ctypes.POINTER(restype.ctype_class)).contents

            return weld_to_numpy(result).decode('ascii')
        else:
            return None

    def _weld_to_numpy_func(self, restype):
        if restype == WeldVec(WeldInt16()):
            return self.utils.weld_to_numpy_int16_arr
        elif restype == WeldVec(WeldInt()):
            return self.utils.weld_to_numpy_int_arr
        elif restype == WeldVec(WeldLong()):
            return self.utils.weld_to_numpy_long_arr
        elif restype == WeldVec(WeldFloat()):
            return self.utils.weld_to_numpy_float_arr
        elif restype == WeldVec(WeldDouble()):
            return self.utils.weld_to_numpy_double_arr
        elif restype == WeldVec(WeldBit()):
            return self.utils.weld_to_numpy_bool_arr
        elif restype == WeldVec(WeldVec(WeldChar())):
            return self.utils.weld_to_numpy_char_arr_arr
        elif restype == WeldVec(WeldVec(WeldInt16())):
            return self.utils.weld_to_numpy_int16_arr_arr
        elif restype == WeldVec(WeldVec(WeldInt())):
            return self.utils.weld_to_numpy_int_arr_arr
        elif restype == WeldVec(WeldVec(WeldLong())):
            return self.utils.weld_to_numpy_long_arr_arr
        elif restype == WeldVec(WeldVec(WeldFloat())):
            return self.utils.weld_to_numpy_float_arr_arr
        elif restype == WeldVec(WeldVec(WeldDouble())):
            return self.utils.weld_to_numpy_double_arr_arr
        elif restype == WeldVec(WeldVec(WeldBit())):
            return self.utils.weld_to_numpy_bool_arr_arr
        else:
            return None

    def _try_decode_array(self, data, restype):
        weld_to_numpy = self._weld_to_numpy_func(restype)

        if weld_to_numpy is not None:
            weld_to_numpy.restype = py_object
            weld_to_numpy.argtypes = [restype.ctype_class]
            result = ctypes.cast(data, ctypes.POINTER(restype.ctype_class)).contents

            res = weld_to_numpy(result)
            # TODO: this might be a bug waiting to happen; the dtype is |S0 despite actually being e.g. |S9
            if restype == WeldVec(WeldVec(WeldChar())):
                res = res.astype(np.bytes_)

            return res
        else:
            return None

    def _try_decode_struct(self, data, restype):
        if isinstance(restype, WeldStruct):
            results = []
            # Iterate through all fields in the struct and recursively decode
            for field_type in restype.field_types:
                result = self.decode(data, field_type, raw_ptr=True)
                data += sizeof(field_type.ctype_class())
                results.append(result)

            return tuple(results)
        else:
            return None

    def decode(self, obj, restype, raw_ptr=False):
        if raw_ptr:
            data = obj
        else:
            data = cweld.WeldValue(obj).data()

        decoded_scalar = self._try_decode_scalar(data, restype)
        if decoded_scalar is not None:
            return decoded_scalar

        decoded_array = self._try_decode_array(data, restype)
        if decoded_array is not None:
            return decoded_array

        decoded_struct = self._try_decode_struct(data, restype)
        if decoded_struct is not None:
            return decoded_struct
        else:
            raise TypeError('Unable to decode obj with restype={}'.format(str(restype)))
