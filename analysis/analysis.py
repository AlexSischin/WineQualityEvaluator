from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import visualize_correlation_matrix, visualize_distribution
from columns import *

white_wine_file = Path('../dataset/winequality-white.csv')
red_wine_file = Path('../dataset/winequality-red.csv')


def analyze_white_wine():
    df = pd.read_csv(white_wine_file, sep=';')

    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)

    plt.show()


def analyze_red_wine():
    df = pd.read_csv(red_wine_file, sep=';')

    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)

    plt.show()


if __name__ == '__main__':
    analyze_white_wine()
    analyze_red_wine()
