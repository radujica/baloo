from .lazy_result import LazyResult, LazyScalarResult, LazyLongResult
from .weld_aggs import weld_aggregate, weld_count


class LazyArrayResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyArrayResult, self).__init__(weld_expr, weld_type, 1)

    def _aggregate(self, operation):
        return LazyScalarResult(weld_aggregate(self.weld_expr,
                                               self.weld_type,
                                               operation),
                                self.weld_type)

    def min(self):
        """Returns the minimum value.

        Returns
        -------
        LazyScalarResult
            The minimum value.

        """
        return self._aggregate('min')

    def max(self):
        """Returns the maximum value.

        Returns
        -------
        LazyScalarResult
            The maximum value.

        """
        return self._aggregate('max')

    def _lazy_len(self):
        return LazyLongResult(weld_count(self.weld_expr))

    def __len__(self):
        """Eagerly get the length.

        Note that if the length is unknown (such as for a WeldObject stop),
        it will be eagerly computed by evaluating the data!

        Returns
        -------
        int
            Length.

        """
        if self._length is None:
            self._length = self._lazy_len().evaluate()

        return self._length