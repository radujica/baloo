import numpy as np
from weld.weldobject import WeldObject, WeldLong, WeldBit

from ...core.utils import check_type, infer_dtype, valid_int_slice
from ...weld import LazyResult, numpy_to_weld_type, weld_count, weld_filter, weld_slice


class Index(LazyResult):
    """Weld-ed Pandas Index.

    Attributes
    ----------
    dtype : np.dtype
        Numpy dtype of the elements.
    name : str
        Name of the series.

    See Also
    --------
    pandas.Index

    """
    def __init__(self, data, dtype=None, name=None):
        """Initialize an Index object.

        Parameters
        ----------
        data : np.ndarray or WeldObject
            Raw data or Weld expression.
        dtype : np.dtype, optional
            Numpy dtype of the elements. Inferred from `data` by default.
        name : str, optional
            Name of the Index.

        """
        data = check_type(data, (np.ndarray, WeldObject))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Index, self).__init__(data, numpy_to_weld_type(self.dtype), 1)

    @property
    def name(self):
        """The name of the Index.

        Returns
        -------
        str
            The name of the Index.

        """
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def values(self):
        """The internal data representation.

        Returns
        -------
        numpy.ndarray or WeldObject
            The internal data representation.

        """
        return self.weld_expr

    def __len__(self):
        """Eagerly get the length of the Index.

        Note that if the length is unknown (such as for a WeldObject),
        it will be eagerly computed.

        Returns
        -------
        int
            Length of the Index.

        """
        if self._length is not None:
            return self._length
        else:
            return LazyResult(weld_count(self.values), WeldLong(), 0).evaluate()

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        return str(self.values)

    def __getitem__(self, item):
        """Select from the Index. Currently used internally through DataFrame and Series.

        Supported functionality:

        - Filter: ind[ind <comparison> <scalar>]
        - Slice: ind[<start>:<stop>:<step>]

        """
        if isinstance(item, LazyResult):
            if item.weld_type != WeldBit():
                raise ValueError('Expected LazyResult of bool data to filter values')

            return Index(weld_filter(self.weld_expr,
                                     self.weld_type,
                                     item.weld_expr),
                         self.dtype,
                         self.name)
        elif isinstance(item, slice):
            if not valid_int_slice(item):
                raise ValueError('Can currently only slice with integers')

            return Index(weld_slice(self.weld_expr,
                                    self.weld_type,
                                    item),
                         self.dtype,
                         self.name)
        else:
            raise TypeError('Expected a LazyResult or a slice')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1, apply_experimental=True):
        """Evaluates by creating an Index containing evaluated data.

        See `LazyResult`

        Returns
        -------
        Index
            Index with evaluated data.

        """
        evaluated_data = super(Index, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(evaluated_data, self.dtype, self.name)
