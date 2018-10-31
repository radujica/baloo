import numpy as np

from baloo.weld import LazyArrayResult, WeldLong, Cache, LazyStructOfVecResult
from baloo.weld.cache import _FakeArray, _FakeStructMember
from baloo.weld.weld_utils import create_placeholder_weld_object, create_weld_object


class TestCache(object):
    def test_cache_intermediate_result(self):
        data = np.arange(5)
        weld_obj = create_placeholder_weld_object(data)
        intermediate_result = LazyArrayResult(weld_obj, WeldLong())

        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'test')

        assert dependency_name in Cache._intermediate_results
        assert Cache._intermediate_results[dependency_name] == intermediate_result
        assert dependency_name.startswith('_interm_') and dependency_name.endswith('_test')

    def test_create_fake_array_input(self):
        data = np.arange(5)
        weld_obj = create_placeholder_weld_object(data)
        intermediate_result = LazyArrayResult(weld_obj, WeldLong())
        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'test')
        fake_weld_input = Cache.create_fake_array_input(dependency_name, 'test_array')

        assert isinstance(fake_weld_input, _FakeArray)
        assert fake_weld_input.dependency == dependency_name
        assert fake_weld_input.name.startswith('_interm_') and fake_weld_input.name.endswith('_test_array')

    def test_create_fake_array_input_tuple(self):
        data = np.arange(5)
        obj_id, weld_obj = create_weld_object(data)
        weld_obj.weld_code = '{{{obj_id}, {obj_id}}}'.format(obj_id=obj_id)
        intermediate_result = LazyStructOfVecResult(weld_obj, [WeldLong(), WeldLong()])
        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'test')
        fake_weld_input1 = Cache.create_fake_array_input(dependency_name, 'test_struct_1', (0, ))
        fake_weld_input2 = Cache.create_fake_array_input(dependency_name, 'test_struct_2', (1, ))

        assert isinstance(fake_weld_input1, _FakeStructMember)
        assert isinstance(fake_weld_input2, _FakeStructMember)
        assert fake_weld_input1.dependency == dependency_name
        assert fake_weld_input2.dependency == dependency_name
        assert fake_weld_input1.name.startswith('_interm_') and fake_weld_input1.name.endswith('_test_struct_1')
        assert fake_weld_input2.name.startswith('_interm_') and fake_weld_input2.name.endswith('_test_struct_2')
        assert fake_weld_input1.index == (0, )
        assert fake_weld_input2.index == (1, )

    def test_integration_array(self):
        data = np.arange(5)
        weld_obj = create_placeholder_weld_object(data)
        intermediate_result = LazyArrayResult(weld_obj, WeldLong())
        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'test')
        fake_weld_input = Cache.create_fake_array_input(dependency_name, 'test_array')

        obj_id, weld_obj = create_weld_object(fake_weld_input)
        Cache.cache_fake_input(obj_id, fake_weld_input)

        assert Cache.contains(obj_id)

        weld_template = 'result(for({interm}, appender, |b, i, e| merge(b, e + 1L)))'
        weld_obj.weld_code = weld_template.format(interm=obj_id)
        actual = LazyArrayResult(weld_obj, WeldLong())

        actual = actual.evaluate()
        expected = np.arange(1, 6)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(Cache._cache[obj_id], data)
        np.testing.assert_array_equal(Cache._intermediate_results[dependency_name], data)

    def test_integration_tuple(self):
        data = np.arange(5)
        obj_id, weld_obj = create_weld_object(data)
        weld_obj.weld_code = '{{{obj_id}, {obj_id}}}'.format(obj_id=obj_id)
        intermediate_result = LazyStructOfVecResult(weld_obj, [WeldLong(), WeldLong()])
        dependency_name = Cache.cache_intermediate_result(intermediate_result, 'test')
        fake_weld_input1 = Cache.create_fake_array_input(dependency_name, 'test_struct_1', (0, ))
        # fake_weld_input2 = Cache.create_fake_array_input(dependency_name, 'test_struct_2', 1)

        obj_id, weld_obj = create_weld_object(fake_weld_input1)
        Cache.cache_fake_input(obj_id, fake_weld_input1)

        assert Cache.contains(obj_id)

        weld_template = 'result(for({interm}, appender, |b, i, e| merge(b, e + 1L)))'
        weld_obj.weld_code = weld_template.format(interm=obj_id)
        actual = LazyArrayResult(weld_obj, WeldLong())

        actual = actual.evaluate()
        expected = np.arange(1, 6)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(Cache._cache[obj_id], data)
        np.testing.assert_array_equal(Cache._intermediate_results[dependency_name], [data, data])
