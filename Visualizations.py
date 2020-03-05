import os

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from DataProcessing.Preprocessing import get_big_dataframe
from Utils import PROJECT_ROOT


def show_scatter_plot_matrix(df):
    pd.plotting.scatter_matrix(df)
    plt.show()


def show_correlation_matrix(df: pd.DataFrame):
    print("Correlation matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)
    return corr_matrix


# def show_histogram(df, column):



if __name__ == "__main__":
    df = get_big_dataframe(os.path.join(PROJECT_ROOT, 'data'))
    show_scatter_plot_matrix(df)
