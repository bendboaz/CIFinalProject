import os

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from DataProcessing.Preprocessing import get_big_dataframe
from Utils import PROJECT_ROOT, trim_for_overlap
from DataProcessing.Features import get_engineered_dataframe
from EffectComputations import calculate_propensity


def show_correlation_matrix(df: pd.DataFrame, do_abs=False, save_path=None):
    print("Correlation matrix:")
    corr_matrix = df.corr()
    if do_abs:
        corr_matrix = np.abs(corr_matrix)
    plt.matshow(corr_matrix)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return corr_matrix


def show_histogram(df: pd.DataFrame, column):
    df.hist(column=column)
    plt.show()


def show_obesity_cutoff_graphs(data_path, big_df, outcome_column, treatment_column, tolerance):
    """
    Shows state histogram before/after PS trimming, for obesity cutoffs at 25%/50%/75%
    :param data_path:
    :param outcome_column:
    :param treatment_column:
    :param tolerance:
    :return:
    """
    df_025 = get_engineered_dataframe(data_path, 'engineered_025.csv', big_df, outcome_column,
                                      cols_to_keep=[outcome_column, treatment_column],
                                      cols_to_convert=[(treatment_column, 0.25)])
    df_05 = get_engineered_dataframe(data_path, 'engineered_05.csv', big_df, outcome_column,
                                     cols_to_keep=[outcome_column, treatment_column],
                                     cols_to_convert=[(treatment_column, 0.5)])
    df_075 = get_engineered_dataframe(data_path, 'engineered_075.csv', big_df, outcome_column,
                                      cols_to_keep=[outcome_column, treatment_column],
                                      cols_to_convert=[(treatment_column, 0.75)])

    df_025 = calculate_propensity(df_025, treatment_column, outcome_column, LogisticRegression,
                                  random_state=346, solver='lbfgs')
    df_025 = trim_for_overlap(df_025, treatment_column, show_graphs=True, tolerance=tolerance)

    df_05 = calculate_propensity(df_05, treatment_column, outcome_column, LogisticRegression,
                                 random_state=346, solver='lbfgs')
    df_05 = trim_for_overlap(df_05, treatment_column, show_graphs=True, tolerance=tolerance)

    df_075 = calculate_propensity(df_075, treatment_column, outcome_column, LogisticRegression,
                                  random_state=346, solver='lbfgs')
    df_075 = trim_for_overlap(df_075, treatment_column, show_graphs=True, tolerance=tolerance)


if __name__ == "__main__":
    data_path = os.path.join(PROJECT_ROOT, 'data')
    big_df = get_big_dataframe(data_path)
    outcome_column = 'totalScore'
    treatment_column = 'obesity_percentage'

    # show_correlation_matrix(big_df.drop([outcome_column, treatment_column], axis=1), do_abs=True,
    #                         save_path=os.path.join(data_path, 'results', 'abs_corr_matrix.jpg'))
    show_obesity_cutoff_graphs(data_path, big_df, outcome_column, treatment_column, 0.05)
