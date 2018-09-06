import abc


class BinaryOps(abc.ABC):
    # not abstractmethod because implementation in DataFrame is ambiguous atm
    def _bitwise_operation(self, other, operation):
        raise NotImplementedError

    def __and__(self, other):
        return self._bitwise_operation(other, '&&')

    def __or__(self, other):
        return self._bitwise_operation(other, '||')

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
