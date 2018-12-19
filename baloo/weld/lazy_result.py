from .cache import Cache
from .convertors import numpy_to_weld_type
from .convertors.utils import to_weld_vec
from .pyweld.types import WeldStruct, WeldVec, WeldLong, WeldDouble
from .pyweld.weldobject import WeldObject
from .weld_aggs import weld_aggregate, weld_count
from .weld_utils import weld_cast_array


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
    _cache = Cache()

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
            old_context = dict(self.weld_expr.context)

            for key in self.weld_expr.context.keys():
                if LazyResult._cache.contains(key):
                    self.weld_expr.context[key] = LazyResult._cache.get(key)

            evaluated = self.weld_expr.evaluate(to_weld_vec(self.weld_type,
                                                            self.ndim),
                                                verbose,
                                                decode,
                                                passes,
                                                num_threads,
                                                apply_experimental_transforms)

            self.weld_expr.context = old_context

            return evaluated
        else:
            return self.weld_expr


# TODO: not really happy having functionality here; maybe have e.g. LazyArray(LazyArrayResult) adding the functionality?
class LazyArrayResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyArrayResult, self).__init__(weld_expr, weld_type, 1)

    @property
    def empty(self):
        if self.is_raw():
            return len(self.weld_expr) == 0
        else:
            return False

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

    def _astype(self, dtype):
        return weld_cast_array(self.values,
                               self.weld_type,
                               numpy_to_weld_type(dtype))


# could make all subclasses but seems rather unnecessary atm
class LazyScalarResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyScalarResult, self).__init__(weld_expr, weld_type, 0)


class LazyLongResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldLong(), 0)


class LazyDoubleResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldDouble(), 0)


class LazyStructResult(LazyResult):
    # weld_types should be a list of the Weld types in the struct
    def __init__(self, weld_expr, weld_types):
        super(LazyStructResult, self).__init__(weld_expr, WeldStruct(weld_types), 0)


class LazyStructOfVecResult(LazyStructResult):
    # weld_types should be a list of the Weld types in the struct
    def __init__(self, weld_expr, weld_types):
        weld_vec_types = [WeldVec(weld_type) for weld_type in weld_types]

        super(LazyStructOfVecResult, self).__init__(weld_expr, weld_vec_types)
