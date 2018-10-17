import abc


# TODO: implement a flush-like method which clears both this cache and the values it added to WeldObject._registry?
# garbage collection should then kick in and remove intermediate results from memory; though probably need to implement
# `del df/sr/etc` to make this possible
# TODO: this functionality would be nice if it could be a 'cache()' operator which is detected in weld_code at evaluate;
# at the Python side (not weld/rust) since caching violates the Weld stateless principle...
class Cache(object):
    # TODO: rewrite these
    """workflow:
    1. cache intermediate result -> returns it's placeholder name; adds placeholder -> intermresult in _interm_result : cache_intermediate_result
    2. use placeholder to create fake inputs -> returns fake_input; no sideeffect : create_fake_array_input
    3. use fake inputs as input array to weld objects -> weld_input_name obtained for the fake input;
        fake input is implicitly added to WeldObject._registry[fake_input] -> weld_input_name : WeldObject(); weld_obj.update(fake_input)
    4. link this weld_input_name to the fake input for evaluate to find; adds weld_input_name -> fake input to _cache : cache_fake_input
    5. use the weld_input_name in the weld_code as desired


    Cache for intermediate results. Acts as a singleton.

    The idea is to track weld_input_ids -> IntermediateResults for intermediate data which only gets computed
    upon evaluation. For this to work, we treat intermediate results as new inputs to Weld. This means that
    intermediate results can now exist within WeldObject.context as lazy inputs/data represented with a placeholder.

    For example, a context could be: {'_inp0': np.array([1, 2, 3]), '_inp1': 'inter_result_join'}.

    Typically, the cache shall be used to cache an entire DataFrame which would actually be tuples of columns.
    To be able to lazily encode this into future WeldObjects which would hence depend on it,
    we either need to be able to encode a tuple, or provide further functionality in the Cache. Second option is
    chosen by having the IntermediateResult contain a list of dependencies on other intermediate results.

    The placeholder is merely used to be informative to the user. On evaluation, the cache is checked for
    intermediate results and if found, the context is replaced with the actual, i.e. raw, data. If it's the first
    time the intermediate result is required, it is evaluated and replaced in the cache such that future uses
    can retrieve the already computed data.

    For example, the context could now be: {'_inp0': np.array([1, 2, 3]), '_inp1': np.array([4., 5., 6.])}

    With a now valid context, the LazyResult can now be evaluated.

    Lastly, the context of the WeldObject is reverted to its original state to avoid polluting the context
    unnecessarily. If not doing this, this and future dependencies on the WeldObject will have outdated data
    in the context (while maintaining an unnecessary reference to the data, working against the garbage collector
    ~ future work)

    Currently contains a few sanity checks, aka asserts. Could remove later.

    """
    _counter = 0
    # actual intermediate results, weld_input_name -> LazyResult
    _intermediate_results = {}
    # the fake inputs used during evaluation, weld_input_name -> FakeWeldInput
    _cache = {}

    @staticmethod
    def _generate_placeholder(readable_text=None):
        """Generate a placeholder name to use while updating WeldObject.

        Parameters
        ----------
        readable_text : str, optional
            Appended to the name for a more understandable placeholder.

        Returns
        -------
        str
            Placeholder.

        """
        name = '_interm_' + str(Cache._counter)
        Cache._counter += 1

        if readable_text is not None:
            assert isinstance(readable_text, str)
            name += '_' + readable_text

        return name

    # returns the evaluated intermediate result
    @classmethod
    def _get_intermediate_result(cls, weld_input_name):
        from .lazy_result import LazyResult

        intermediate_result = Cache._intermediate_results[weld_input_name]
        # if not yet evaluated, do so and cache the result
        if isinstance(intermediate_result, LazyResult):
            intermediate_result = intermediate_result.evaluate()
            Cache._intermediate_results[weld_input_name] = intermediate_result

        return intermediate_result

    @classmethod
    def contains(cls, item):
        assert isinstance(item, str)

        return item in cls._cache

    @classmethod
    def cache_intermediate_result(cls, result, readable_name=None):
        """Add result to the cached data.

        Parameters
        ----------
        result : LazyResult
            Data to cache.
        readable_name : str
            Will be used when generating a name for this intermediate result.

        Returns
        -------
        str
            A generated placeholder name uniquely identifying this intermediate result.

        """
        from .lazy_result import LazyResult
        assert isinstance(result, LazyResult)

        dependency_name = Cache._generate_placeholder(readable_name)
        Cache._intermediate_results[dependency_name] = result

        return dependency_name

    @classmethod
    def create_fake_array_input(cls, dependency, readable_name, index=None):
        """Create fake Weld inputs to be used in future WeldObjects.

        Parameters
        ----------
        dependency : str
            The Weld input name of the actual intermediate result, obtained from cache_intermediate_result.
        readable_name : str
            User-friendly string that will be used to generate a unique placeholder name. This placeholder
            will be seen in the WeldObject.context.
        index : int, optional
            If passed, it means the intermediate result dependency is in fact a struct/tuple, so this fake input
            shall be able to select the required array from the struct.

        Returns
        -------
        FakeWeldInput
            An instance representing the fake Weld input to be used when generating the WeldObjects relying on it.

        """
        assert dependency in Cache._intermediate_results
        assert isinstance(readable_name, str)

        name = Cache._generate_placeholder(readable_name)

        if index is None:
            fake_weld_input = _FakeArray(dependency, name)
        else:
            fake_weld_input = _FakeStructMember(dependency, index, name)

        return fake_weld_input

    @classmethod
    def cache_fake_input(cls, weld_input_id, fake_weld_input):
        """Cache the fake Weld input to be seen by LazyResult.evaluate

        Parameters
        ----------
        weld_input_id : str
            Generated when registering the fake_weld_input in WeldObject.update.
        fake_weld_input : _FakeWeldInput
            The fake Weld input previously generated by create_fake_array_input.

        """
        assert isinstance(weld_input_id, str)
        assert isinstance(fake_weld_input, _FakeWeldInput)

        Cache._cache[weld_input_id] = fake_weld_input

    @classmethod
    def get(cls, key):
        """Retrieve a fake Weld input. Evaluate its intermediate result dependency if not yet done.

        Parameters
        ----------
        key : str
            Weld input name previously obtained through create_fake_array_input.

        Returns
        -------
        numpy.ndarray or tuple
            The corresponding data.

        """
        cls._cache[key] = cls._cache[key].retrieve()

        return cls._cache[key]


class _FakeWeldInput(abc.ABC):
    def __init__(self, dependency, readable_name):
        self.dependency = dependency
        self.name = readable_name

    def __repr__(self):
        return '_FakeWeldInput(dependency={}, name={})'.format(self.dependency, self.name)

    def __str__(self):
        return self.name

    def _evaluate_dependency(self):
        return Cache._get_intermediate_result(self.dependency)

    @abc.abstractmethod
    def retrieve(self):
        raise NotImplementedError


class _FakeArray(_FakeWeldInput):
    def __init__(self, dependency, readable_name):
        super(_FakeArray, self).__init__(dependency, readable_name)

    def retrieve(self):
        return self._evaluate_dependency()


class _FakeStructMember(_FakeWeldInput):
    def __init__(self, dependency, index, readable_name):
        assert isinstance(index, int)

        self.index = index

        super(_FakeStructMember, self).__init__(dependency, readable_name)

    def retrieve(self):
        return self._evaluate_dependency()[self.index]
