from numbers import Number

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame, Series, Interval


def _float_interval_to_str(i: Interval):
    return f'({i.left},{i.right}]'


def _int_interval_to_str(i: Interval):
    return f'({int(np.floor(i.left))},{int(np.floor(i.right))}]'


def visualize_scatter(ax: Axes, x: Series, y: Series, alpha=1):
    ax.set_title(f'{y.name} vs {x.name}')
    ax.set_xlabel(f'{x.name}')
    ax.set_ylabel(f'{y.name}')
    ax.scatter(x.to_numpy(), y.to_numpy(), marker='.', alpha=alpha)


def visualize_distribution(ax: Axes, series: Series, categorical, bins=10, integer=True):
    ax.set_title(f'Distribution of {series.name}')
    ax.set_xlabel(f'{series.name}')
    ax.set_ylabel(f'Occurrences')

    if not categorical:
        if isinstance(bins, Number):
            bins = np.linspace(series.min(), series.max(), bins + 1)
        if integer:
            bins = np.rint(bins).astype(np.int64)
        interval_mapper = _int_interval_to_str if integer else _float_interval_to_str
        series = pd.cut(series, bins, include_lowest=True).map(interval_mapper)

    count_df = series.value_counts(sort=False).sort_index().reset_index()
    col_values, col_occurrences = 'values', 'occurrences'
    count_df.columns = col_values, col_occurrences

    if categorical:
        count_df.sort_values(by=col_occurrences, ascending=False, inplace=True)

    x = count_df[col_values]
    y = count_df[col_occurrences]
    x_numeric = np.arange(x.size)
    ax.bar(x_numeric, y)
    ax.set_xticks(x_numeric, x, rotation=90, ha='center')


def visualize_correlation_matrix(ax: Axes, corr: DataFrame):
    ax.set_title('Correlation matrix')
    ax.matshow(corr, cmap='BrBG')
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center')


def visualize_confusion_matrix(ax: Axes, conf: np.ndarray):
    ax.matshow(conf, cmap='PuBu')
    ax.set_xlabel('Estimate')
    ax.set_ylabel('Target')
    for (i, j), z in np.ndenumerate(conf):
        if z != 0:
            ax.text(j, i, '{:.0f}'.format(z), ha='center', va='center')
