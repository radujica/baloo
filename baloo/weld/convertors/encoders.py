import numpy as np
import pkg_resources
from weld.weldobject import *

from .utils import to_shared_lib, to_weld_vec

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
    """Convert from np.dtype to WeldType.

    Note that support for strings is intended to be only for
    Python 2 str and Python 3 bytes. No unicode.

    Parameters
    ----------
    np_dtype : np.dtype or str
        NumPy dtype.

    Returns
    -------
    WeldType
        Corresponding WeldType.
    """
    if not isinstance(np_dtype, (str, np.dtype)):
        raise TypeError('Can only convert np.dtype or str')

    if isinstance(np_dtype, str):
        np_dtype = np.dtype(np_dtype)

    return _numpy_to_weld_type_mapping[np_dtype.char]


class NumPyEncoder(WeldObjectEncoder):
    def __init__(self):
        lib = to_shared_lib('numpy_weld_convertor')
        lib_file = pkg_resources.resource_filename(__name__, lib)
        self.utils = ctypes.PyDLL(lib_file)

    def py_to_weld_type(self, obj):
        if isinstance(obj, np.ndarray):
            base = numpy_to_weld_type(obj.dtype)
            base = to_weld_vec(base, obj.ndim)

            return base
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
                raise TypeError('Unable to decode np.ndarray of 1 dimension with dtype={}'.format(str(obj.dtype)))
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
                raise TypeError('Unable to decode np.ndarray of 2 dimensions with dtype={}'.format(str(obj.dtype)))
        else:
            raise ValueError('Can only encode np.ndarray of 1 or 2 dimensions')

        return numpy_to_weld

    def encode(self, obj):
        if isinstance(obj, np.ndarray):
            numpy_to_weld = self._numpy_to_weld_func(obj)
            numpy_to_weld.restype = self.py_to_weld_type(obj).ctype_class
            numpy_to_weld.argtypes = [py_object]

            return numpy_to_weld(obj)
        else:
            raise TypeError('Unable to encode obj of type={}'.format(str(type(obj))))


class NumPyDecoder(WeldObjectDecoder):
    def __init__(self):
        lib = to_shared_lib('numpy_weld_convertor')
        lib_file = pkg_resources.resource_filename(__name__, lib)
        self.utils = ctypes.PyDLL(lib_file)

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

            return weld_to_numpy(result)
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
