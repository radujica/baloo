from weld.weldobject import *

from .weld_aggs import weld_count, weld_aggregate
from .convertors.utils import to_weld_vec


class LazyResult(object):
    """Wrapper class around a yet un-evaluated Weld result.

    Attributes
    ----------
    weld_expr : WeldObject or numpy.ndarray
        Expression that needs to be evaluated.
    weld_type : WeldType
        Type of the output.
    ndim : int
        Dimensionality of the output.

    """
    def __init__(self, weld_expr, weld_type, ndim):
        self.weld_expr = weld_expr
        self.weld_type = weld_type
        self.ndim = ndim

    def __repr__(self):
        return "{}(weld_type={}, ndim={})".format(self.__class__.__name__,
                                                  self.weld_type,
                                                  self.ndim)

    def __str__(self):
        return str(self.weld_expr)

    @property
    def values(self):
        """The internal data representation.

        Returns
        -------
        numpy.ndarray or WeldObject
            The internal data representation.

        """
        return self.weld_expr

    def is_raw(self):
        return not isinstance(self.weld_expr, WeldObject)

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=True):
        """Evaluate the stored expression.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print output for each Weld compilation step.
        decode : bool, optional
            Whether to decode the result
        passes : list, optional
            Which Weld optimization passes to apply
        num_threads : int, optional
            On how many threads to run Weld
        apply_experimental_transforms : bool
            Whether to apply the experimental Weld transforms.

        Returns
        -------
        numpy.ndarray
            Output of the evaluated expression.

        """
        if isinstance(self.weld_expr, WeldObject):
            return self.weld_expr.evaluate(to_weld_vec(self.weld_type,
                                                       self.ndim),
                                           verbose,
                                           decode,
                                           passes,
                                           num_threads,
                                           apply_experimental_transforms)
        else:
            return self.weld_expr


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


# TODO: could make all subclasses but seems rather unnecessary atm
class LazyScalarResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyScalarResult, self).__init__(weld_expr, weld_type, 0)


class LazyLongResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldLong(), 0)


class LazyDoubleResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldDouble(), 0)
