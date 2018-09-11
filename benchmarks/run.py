# Weld is expected to outperform Pandas on 1GB+ of data.
# My VM does not have enough memory for numbers that large.
# Main time hog in Weld is compilation.
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.utils import run_benchmarks, benchmark, run_correctness_checks, profile_memory_usage

operations = [
    "df = df[(df['col1'] > 0) & (df['col2'] >= 10) & (df['col3'] < 30)]",
    "df = df.max()",
    "df = df.agg(['min', 'prod', 'mean', 'std'])",
    "df['col4'] = df['col1'] * 2",
    "df['col4'] = df['col1'] * 2 + 1 - 23"
]


def plot_scalability():
    scales = [1, 10, 100, 1000, 10000, 20000]
    sizes = [0.028, 0.28, 2.8, 28., 280., 560.]  # modify this if changing the generated data!
    operation = operations[0]
    output_file = io.StringIO()
    for scale in scales:
        benchmark(operation, scale, 5, output_file)

    output = output_file.getvalue()
    df = pd.DataFrame(output.split('\n'), columns=['output'])
    df['op'], df['time'] = df['output'].str.split(':', 1).str
    df = df.drop('output', axis=1).dropna()
    df['time'] = df['time'].map(lambda x: x.split()[0]).astype(np.float64)
    df = df.groupby('op')
    data = {group: data.reset_index().drop('index', axis=1) for group, data in df}

    fig, ax = plt.subplots()
    scatter1 = ax.plot(data['pandas'].index.values, data['pandas']['time'], color='r')
    scatter2 = ax.plot(data['baloo'].index.values, data['baloo']['time'], color='g')

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Scale (MB)')
    ax.set_xticks(data['pandas'].index.values)
    ax.set_xticklabels(sizes)
    ax.set_title('Average execution time of filter operation')
    ax.legend((scatter1[0], scatter2[0]), ('pandas', 'baloo'))

    # plt.show()
    plt.savefig('scalability.png')


def plot_benchmarks():
    output_file = io.StringIO()
    scale = 10000
    run_benchmarks(operations, scale=scale, runs=1, file=output_file)
    output = output_file.getvalue()
    df = pd.DataFrame(output.split('\n'), columns=['output'])
    df['op'], df['time'] = df['output'].str.split(':', 1).str
    df = df.drop('output', axis=1).dropna()
    df['time'] = df['time'].map(lambda x: x.split()[0]).astype(np.float64)
    df = df.groupby('op')       # groupby maintains the order of operations
    data = {group: data.reset_index().drop('index', axis=1) for group, data in df}

    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(data['pandas'].index, data['pandas']['time'], width, color='r')
    rects2 = ax.bar(data['baloo'].index + width, data['baloo']['time'], width, color='g')

    ax.set_ylabel('Time (s)')
    ax.set_title('Average execution time for specific operations')
    ax.set_xticks(data['pandas'].index + width / 2)
    ax.set_xticklabels(('filter', 'max', '4x agg', 'op', '3x op'))

    ax.legend((rects1[0], rects2[0]), ('pandas', 'baloo'))

    # plt.show()
    plt.savefig('benchmarks.png')


run_correctness_checks(operations, scale=10000)
plot_benchmarks()
plot_scalability()
profile_memory_usage(operations[0], scale=10000)
