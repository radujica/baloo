import numpy as np
from weld.weldobject import WeldObject, WeldLong

from ...core.utils import check_type, infer_dtype
from ...weld import LazyResult, numpy_to_weld_type, weld_count


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
        self.data = check_type(data, (np.ndarray, WeldObject))
        self.dtype = infer_dtype(data, check_type(dtype, np.dtype))
        self.name = check_type(name, str)
        self._length = len(data) if isinstance(data, np.ndarray) else None

        super(Index, self).__init__(data, numpy_to_weld_type(self.dtype), 1)

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

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
