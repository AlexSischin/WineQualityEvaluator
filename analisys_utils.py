import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


def visualize_correlation_matrix(corr: DataFrame):
    fig, ax = plt.subplots()
    ax.set_title('Correlation matrix')
    ax.matshow(corr, cmap='BrBG')
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center')
    return fig, ax
