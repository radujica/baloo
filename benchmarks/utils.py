import contextlib
import io
import sys
import time
from collections import OrderedDict
from functools import wraps

import numpy as np
import pandas as pd
from memory_profiler import profile

import baloo as bl


def generate_data(scale=1):
    np.random.seed(42)
    n = 1000

    print('Generating data...')

    data = OrderedDict((
        ('col1', np.random.randn(n * scale) * 17),
        ('col2', np.random.randn(n * scale) * 29),
        ('col3', np.random.randint(100, size=n*scale, dtype=np.int64)),
        ('col4', np.random.randint(200, size=n*scale, dtype=np.int32))
    ))
    size = sum(arr.nbytes for arr in data.values())

    print('Data size in MB: {}'.format(size / 1000000))

    return data


# decorator to time a function
def timer(runs=5, file=sys.stdout):
    def function_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return_values = []
            runtimes = []
            for i in range(runs):
                start = time.time()
                return_value = func(*args, **kwargs)
                end = time.time()
                runtimes.append(float(end) - start)
                return_values.append(return_value)

            msg = '{func}: {time:.8f} seconds'
            print(msg.format(func=func.__name__,
                             time=sum(runtimes) / runs,
                             runs=runs),
                  file=file)

            return return_values
        return wrapper
    return function_timer


# summarizes the time spend in each Weld step
def _process_weld_output(weld_output):
    df = pd.DataFrame(weld_output.split('\n'), columns=['output'])
    df['op'], df['time'] = df['output'].str.split(':', 1).str
    df = df.drop('output', axis=1).dropna()
    df['time'] = df['time'].astype(np.float64)
    df = df.groupby('op').sum()

    return df


# evaluates with verbose=true to print time spent in each compilation step; further processed by process_weld_output
def _record_verbose_weld_output(df):
    f = io.StringIO()
    original = sys.stdout
    sys.stdout = f
    df = df.evaluate(verbose=True)
    sys.stdout = original
    verbose_output = f.getvalue()
    f.close()

    return df, verbose_output


# averages the Weld compilation times output of multiple runs and prints it to stdout
def _average_weld_output(dfs):
    weld_output = dfs[0]
    for out in dfs[1:]:
        weld_output = weld_output.append(out)
    weld_output = weld_output.groupby('op').mean()

    print(weld_output)


# executes the operation on baloo while gathering verbose statistics
def verbose_baloo(operation, scale=1, runs=5):
    assert runs > 0

    print('Running verbose benchmark on: {}'.format(operation))
    print('Averaging over {} runs'.format((str(runs))))
    data = generate_data(scale=scale)

    @timer()
    def run():
        df = bl.DataFrame(data)
        exec(operation)
        df = df.sum()

        df, verbose_output = _record_verbose_weld_output(df)
        with contextlib.redirect_stdout(None):
            print(df.values)

        return verbose_output

    verbose_outputs = run()
    _average_weld_output([_process_weld_output(out) for out in verbose_outputs])


# only checks if series are equal, which makes sense given the sum aggregation
def check_correctness(operation, scale=1):
    print('Checking correctness of: {}'.format(operation))
    generated_data = generate_data(scale=scale)

    def pandas(op, data):
        df = pd.DataFrame(data)
        exec(op)
        df = df.sum()

        return df.values

    def baloo(op, data):
        df = bl.DataFrame(data)
        # temp workaround
        op = op.replace('np', 'bl')
        exec(op)
        df = df.sum()

        return df.evaluate().values

    result_pandas = pandas(operation, generated_data)
    result_baloo = baloo(operation, generated_data)

    np.testing.assert_allclose(result_pandas, result_baloo)

    print('All good!\n')


def run_correctness_checks(operations, scale=1):
    for operation in operations:
        check_correctness(operation, scale)


# runs the operation on both pandas and baloo while profiling memory usage
def profile_memory_usage(operation, scale=1):
    print('Running memory profiling on: {}'.format(operation))
    generated_data = generate_data(scale=scale)

    @profile
    def pandas(op, data):
        df = pd.DataFrame(data)
        exec(op)
        df = df.sum()

        with contextlib.redirect_stdout(None):
            print(df.values)

    @profile
    def baloo(op, data):
        df = bl.DataFrame(data)
        # temp workaround
        op = op.replace('np', 'bl')
        exec(op)
        df = df.sum()

        with contextlib.redirect_stdout(None):
            print(df.evaluate().values)

    print('pandas:')
    pandas(operation, generated_data)
    print('baloo:')
    baloo(operation, generated_data)


# evaluation is forced by performing a sum aggregation;
# pretty-print overhead (mostly) avoided by using .values;
# note that data caching has little effect on these numbers (as manually tested);
# data in memory anyway and CPU cache hard to manipulate
def benchmark(operation, scale=1, runs=5, file=sys.stdout):
    assert runs > 0

    print('Running benchmark on: {}'.format(operation))
    print('Averaging over {} runs'.format((str(runs))))
    generated_data = generate_data(scale=scale)

    @timer(runs=runs, file=file)
    def pandas(op, data):
        df = pd.DataFrame(data)
        exec(op)
        df = df.sum()

        with contextlib.redirect_stdout(None):
            print(df.values)

    @timer(runs=runs, file=file)
    def baloo(op, data):
        df = bl.DataFrame(data)
        # temp workaround
        op = op.replace('np', 'bl')
        exec(op)
        df = df.sum()

        with contextlib.redirect_stdout(None):
            print(df.evaluate().values)

    pandas(operation, generated_data)
    baloo(operation, generated_data)

    print('Done')


def run_benchmarks(operations, scale=1, runs=5, file=sys.stdout):
    for operation in operations:
        benchmark(operation, scale, runs, file)
