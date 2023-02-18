import matplotlib.pyplot as plt
from pandas import DataFrame

from analysis_utils import visualize_correlation_matrix, visualize_distribution
from columns import *


def analyze_white_wine(df: DataFrame):
    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)

    plt.show()


def analyze_red_wine(df: DataFrame):
    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)

    plt.show()
