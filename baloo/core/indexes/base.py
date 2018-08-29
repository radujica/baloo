import numpy as np
from weld.weldobject import WeldObject, WeldLong

from ...core.utils import check_attributes, Typed, infer_dtype
from ...weld import LazyResult, numpy_to_weld_type, weld_count


@check_attributes(data=Typed((np.ndarray, WeldObject)),
                  dtype=Typed(np.dtype),
                  name=Typed(str))
class Index(LazyResult):
    """Weld-ed Pandas Index.

    Attributes
    ----------
    dtype : np.dtype
        Numpy dtype of the elements.
    name : str
        Name of the series.

    See also
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
        self.data = data
        self.dtype = infer_dtype(data, dtype)
        self.name = name
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Index, self).__init__(data, numpy_to_weld_type(self.dtype), 1)

    @property
    def data(self):
        """Get the data inside this Index.

        Returns
        -------
        np.ndarray or WeldObject
            Data within this Index.

        """
        return self.weld_expr

    # to actually allow @property to work after Typed descriptors
    @data.setter
    def data(self, value):
        self.data = value

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
            return LazyResult(weld_count(self.data), WeldLong(), 0).evaluate()

    def __repr__(self):
        return "{}(name={}, dtype={})".format(self.__class__.__name__,
                                              self.name,
                                              self.dtype)

    def __str__(self):
        return str(self.data)

    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1, apply_experimental=False):
        data = super(Index, self).evaluate(verbose, decode, passes, num_threads, apply_experimental)

        return Index(data, self.dtype, self.name)
