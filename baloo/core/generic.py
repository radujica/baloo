import abc


# To enforce the implementation of these methods such that convention is maintained.
# Note: inherit from this AFTER any other class that might implement desired default behavior.
class BalooCommon(abc.ABC):
    @property
    @abc.abstractmethod
    def values(self):
        """The internal data representation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def empty(self):
        """Check whether the data structure is empty.

        Returns
        -------
        bool

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        # eager operation returning the length of the internal data
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        # lazy repr without any actual raw/lazy data
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        # eager representation including data
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self):
        """Evaluate by returning object of the same type but now containing raw data."""
        raise NotImplementedError


class BinaryOps(abc.ABC):
    @abc.abstractmethod
    def _comparison(self, other, comparison):
        raise NotImplementedError

    def __lt__(self, other):
        return self._comparison(other, '<')

    def __le__(self, other):
        return self._comparison(other, '<=')

    def __eq__(self, other):
        return self._comparison(other, '==')

    def __ne__(self, other):
        return self._comparison(other, '!=')

    def __ge__(self, other):
        return self._comparison(other, '>=')

    def __gt__(self, other):
        return self._comparison(other, '>')

    def isna(self):
        return self._comparison(None, '==')

    def notna(self):
        return self._comparison(None, '!=')

    @abc.abstractmethod
    def _element_wise_operation(self, other, operation):
        raise NotImplementedError

    def __add__(self, other):
        return self._element_wise_operation(other, '+')

    def __sub__(self, other):
        return self._element_wise_operation(other, '-')

    def __mul__(self, other):
        return self._element_wise_operation(other, '*')

    def __truediv__(self, other):
        return self._element_wise_operation(other, '/')

    def __pow__(self, other):
        return self._element_wise_operation(other, 'pow')


class BitOps(abc.ABC):
    @abc.abstractmethod
    def _bitwise_operation(self, other, operation):
        raise NotImplementedError

    def __and__(self, other):
        return self._bitwise_operation(other, '&&')

    def __or__(self, other):
        return self._bitwise_operation(other, '||')


class IndexCommon(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        """Name of the Index.

        Returns
        -------
        str
            name

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _iloc_indices(self, indices):
        """Filter based on indices.

        Parameters
        ----------
        indices : numpy.ndarray or WeldObject

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _iloc_indices_with_missing(self, indices):
        """Filter based on indices, where an index > length signifies missing data.

        Parameters
        ----------
        indices : numpy.ndarray or WeldObject

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _gather_names(self):
        # returns the names of the index columns as a list replacing None's with default values
        raise NotImplementedError

    @abc.abstractmethod
    def _gather_data_for_weld(self):
        # returns the raw/WeldObjects in a list s.t. can be passed directly to weld_* methods
        raise NotImplementedError

    @abc.abstractmethod
    def _gather_data(self):
        # returns a dict of names to Indexes, not to raw data for Weld
        raise NotImplementedError

    @abc.abstractmethod
    def _gather_weld_types(self):
        # returns the raw/WeldObjects in a list s.t. can be passed directly to weld_* methods
        raise NotImplementedError
