from baloo import RangeIndex, Index, MultiIndex


def assert_indexes_equal(actual, expected, sort=False):
    if type(actual) != type(expected):
        raise AssertionError('Expected indexes of the same type')

    from .test_base import assert_index_equal
    from .test_range import assert_range_equal
    from .test_multi import assert_multiindex_equal

    if isinstance(actual, RangeIndex):
        assert_range_equal(actual, expected)
    elif isinstance(actual, Index):
        assert_index_equal(actual, expected, sort=sort)
    elif isinstance(actual, MultiIndex):
        assert_multiindex_equal(actual, expected, sort=sort)
