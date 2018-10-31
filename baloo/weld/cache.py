import abc


# TODO 1: implement a flush-like method which clears both this cache and the values it added to WeldObject._registry?
# TODO garbage collection should then kick in and remove intermediate results from memory;
# TODO though probably need to implement `del df/sr/etc` to make this possible
# TODO 2: would be nice if there could be a 'cache()' operator which is detected in weld_code at evaluate;
# TODO though at the Python side (not weld/rust) since caching violates the Weld stateless principle...
# TODO 3: re-using the WeldObject.dependencies could be more natural instead of fake inputs
class Cache(object):
    """Cache for intermediate results. Acts as a singleton.

    The idea is to track weld_input_ids -> intermediate results for intermediate data which only gets computed
    upon evaluation. For this to work, we treat intermediate results as new 'fake' inputs to Weld. This means that
    intermediate results can now exist within WeldObject.context as lazy inputs/data represented with a placeholder.

    For example, a context could be: {'_inp0': np.array([1, 2, 3]), '_inp1': 'inter_result_join'}.

    Typically, the cache shall be used to cache an entire DataFrame which would actually be tuples of columns.
    To be able to lazily encode this into future WeldObjects which would hence depend on it,
    we either need to be able to encode a tuple, or provide further functionality in the Cache. Second option is
    chosen by having the _FakeWeldInputs contain a dependency on another intermediate results, here the entire
    DataFrame as a tuple of columns.

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

    tl;dr How to use:

    1. Cache.cache_intermediate_result(LazyResult) => returns placeholder; cache this LazyResult.
        Internally, add placeholder -> LazyResult to _intermediate_results.
    2. Cache.create_fake_array_input(placeholder, index=Optional) => returns a _FakeWeldInput.
    3. WeldObject().update(_FakeWeldInput) => returns weld id for this fake input as _inpX (raw data);
        _FakeWeldInput acts as a regular input to any further WeldObjects. Through update, the
        input internally becomes registered in WeldObject._registry as _FakeWeldInput -> _inpX
    4. Cache.cache_fake_input(id, _FakeWeldInput); the _FakeWeldInput (corresponding to a _inpX) is now cached
        and can be seen by LazyResult.evaluate(). Internally, add _inpX -> _FakeWeldInput to _cache. On evaluate,
        LazyResult replaced the placeholder in the WeldObject.context with the actual data, evaluating if necessary.
    5. Use the weld id in weld_code as desired.

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
        index : tuple, optional
            If passed, it means the intermediate result dependency is in fact a struct/tuple, so this fake input
            shall be able to select the required array from the tuple. Note that it can work through multiple levels,
            i.e. passing (1, 0) would essentially select 2 from (1, (2, 3)).

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
            assert isinstance(index, tuple)
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
        assert isinstance(key, str)

        data = cls._cache[key]
        if isinstance(data, _FakeWeldInput):
            data = data.retrieve()

        cls._cache[key] = data

        return data


class _FakeWeldInput(abc.ABC):
    def __init__(self, dependency, readable_name):
        assert isinstance(dependency, str)
        assert isinstance(readable_name, str)

        self.dependency = dependency
        self.name = readable_name

    def __repr__(self):
        return '_FakeWeldInput(dependency={}, name={})'.format(self.dependency, self.name)

    # the str representation is added to WeldObject._registry
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
        assert isinstance(index, tuple)

        self.index = index

        super(_FakeStructMember, self).__init__(dependency, readable_name)

    def __repr__(self):
        return '_FakeStructMember(dependency={}, index={}, name={})'.format(self.dependency, self.index, self.name)

    def retrieve(self):
        result = self._evaluate_dependency()
        for i in self.index:
            result = result[i]

        return result
