import numpy as np

from ..weld import weld_count


class Descriptor(object):
    def __init__(self, name=None, **kwargs):
        self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


# by default allows None
class Typed(Descriptor):
    def __init__(self, expected_types=None, **kwargs):
        self.expected_types = expected_types

        super().__init__(**kwargs)

    def __set__(self, instance, value):
        if value is not None and not isinstance(value, self.expected_types):
            raise TypeError('Expected: {}'.format(str(self.expected_types)))

        super(Typed, self).__set__(instance, value)


def check_attributes(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key

            setattr(cls, key, value)
        return cls
    return decorate


def infer_dtype(data, arg_dtype):
    if arg_dtype is not None:
        return arg_dtype
    else:
        if isinstance(data, np.ndarray):
            return data.dtype
        else:
            # if WeldObject data then arg_dtype must have been passed as argument
            raise ValueError('Using WeldObject as data requires the dtype as argument')


def default_index(data):
    from .indexes import RangeIndex

    if isinstance(data, np.ndarray):
        return RangeIndex(len(data))
    else:
        # must be WeldObject then
        return RangeIndex(weld_count(data))
