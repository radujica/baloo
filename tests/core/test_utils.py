import pytest

from baloo.core.utils import Typed, check_attributes


class TestDescriptors(object):
    def test_typed_invalid(self):
        class T(object):
            x = Typed(int)

        t = T()

        with pytest.raises(TypeError):
            t.x = 'abc'

    def test_typed_none(self):
        class T(object):
            x = Typed(int)

        t = T()
        t.x = None

    def test_check_attributes(self):
        @check_attributes(a=Typed(int))
        class T(object):
            def __init__(self, a=None, b=None):
                self.a = a
                self.b = b

        t = T()
        assert t.a is None
        assert t.b is None

        t = T(1)
        assert t.a == 1
        assert t.b is None

        t = T(1, 2)
        assert t.a == 1
        assert t.b == 2

        t = T(None, 1)
        assert t.a is None
        assert t.b == 1

        t = T(1, 'abc')
        assert t.a == 1
        assert t.b == 'abc'

        with pytest.raises(TypeError):
            t = T('abc')
