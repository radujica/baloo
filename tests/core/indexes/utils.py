from baloo import RangeIndex, Index


def assert_indexes_equal(actual, expected):
    if actual == expected:
        return

    if type(actual) != type(expected):
        raise AssertionError('Expected indexes of the same type')

    from .test_base import assert_index_equal
    from .test_range import assert_range_equal

    if isinstance(actual, RangeIndex):
        assert_range_equal(actual, expected)
    elif isinstance(actual, Index):
        assert_index_equal(actual, expected)
