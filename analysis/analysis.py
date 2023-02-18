import matplotlib.pyplot as plt
from pandas import DataFrame

from analisys_utils import visualize_correlation_matrix, visualize_distribution
from columns import *


def analyze(df: DataFrame):
    df.info()

    target = df[COL_QUALITY]
    visualize_distribution(target, categorical=True)

    corr = df.corr(numeric_only=True)
    fig, ax = visualize_correlation_matrix(corr)
    fig.subplots_adjust(bottom=0.03, top=0.75)

    plt.show()
