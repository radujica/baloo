import abc

# TODO: maybe write generic class to enforce values, len, repr, str, evaluate?


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


class IlocIndex(abc.ABC):
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
