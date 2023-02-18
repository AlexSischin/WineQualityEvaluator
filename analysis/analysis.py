import matplotlib.pyplot as plt
from pandas import DataFrame

from analisys_utils import visualize_correlation_matrix, visualize_distribution, visualize_scatter
from columns import *


def analyze_white_wine(df: DataFrame):
    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)
    _, (res_sugar_vs_quality_ax) = plt.subplots(1, 1)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)
    visualize_scatter(res_sugar_vs_quality_ax, df[COL_RESIDUAL_SIGAR], target, alpha=0.1)

    plt.show()


def analyze_red_wine(df: DataFrame):
    df.info()

    target = df[COL_QUALITY]
    corr = df.corr(numeric_only=True)

    _, (distr_ax, corr_ax) = plt.subplots(1, 2)

    visualize_distribution(distr_ax, target, categorical=True)
    visualize_correlation_matrix(corr_ax, corr)

    plt.show()
