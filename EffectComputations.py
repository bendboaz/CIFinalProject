import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from DataProcessing.Features import get_engineered_dataframe
from Utils import PROJECT_ROOT, trim_for_overlap


def calculate_propensity(dataset: pd.DataFrame, treatment_col, outcome_col, classifier_cls, to_remove=None, 
                         **classifier_params):
    if to_remove is not None:
        dataset = dataset.drop(to_remove, axis=1)
    
    predictor = classifier_cls(**classifier_params)
    samples = dataset.drop([treatment_col, outcome_col], axis=1)
    labels = dataset[treatment_col]
    predictor.fit(samples, labels)
    propensity_scores = predictor.predict_proba(samples.to_numpy())
    dataset["PS"] = [score for _, score in propensity_scores]
    return dataset


def compute_odds(number):
    return number / (1 - number)


def ate_ipw(df: pd.DataFrame, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    n_samples = len(df.index)
    sigma_t = sum(treated[outcome_col] / treated['PS'])
    sigma_c = sum(control[outcome_col] / (1 - control['PS']))
    return (1 / n_samples) * (sigma_t - sigma_c)


def ate_s(df, treatment_col, outcome_col):
    samples = df.drop(["PS", outcome_col], axis=1)
    labels = df[outcome_col]
    predictor = Ridge(alpha=0.08, normalize=True, random_state=321)
    predictor.fit(samples, labels)

    samples_y1 = samples.copy()
    samples_y1[treatment_col] = np.ones(len(samples_y1.index))
    predicted_y1 = predictor.predict(samples_y1)

    samples_y0 = samples.copy()
    samples_y0[treatment_col] = np.zeros(len(samples_y0.index))
    predicted_y0 = predictor.predict(samples_y0)

    f1 = df[outcome_col] * df[treatment_col] + predicted_y1 * (1 - df[treatment_col])
    f0 = df[outcome_col] * (1 - df[treatment_col]) + predicted_y0 * df[treatment_col]
    return np.mean(f1 - f0)


def ate_match(df, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    t_predictor = KNeighborsRegressor(1)
    cols_to_drop = [treatment_col, outcome_col, "PS"]
    t_predictor.fit(control.drop(cols_to_drop, axis=1), treated[outcome_col])
    c_predictor = KNeighborsRegressor(1)
    c_predictor.fit(control.drop(cols_to_drop, axis=1), control[outcome_col])
    f1 = df[outcome_col] * df[treatment_col] + \
        c_predictor.predict(df.drop(cols_to_drop, axis=1)) * (1 - df[treatment_col])
    f0 = df[outcome_col] * (1 - df[treatment_col]) + \
        t_predictor.predict(df.drop(cols_to_drop, axis=1)) * df[treatment_col]
    return np.mean(f1 - f0)


def ate_magic(df, treatment_col, outcome_col):
    """
    Added propensity score to the s-learner covariates. Our approach:
    https://shir663.files.wordpress.com/2016/12/my-meme.jpg
    :param df:
    :param treatment_col:
    :param outcome_col:
    :return:
    """
    cols_to_drop = [treatment_col, outcome_col]
    samples = df.drop(cols_to_drop, axis=1)

    predictor = Ridge(alpha=0.08, normalize=True, random_state=321)
    predictor.fit(samples, df[outcome_col])

    samples_y1 = samples.copy()
    samples_y1[treatment_col] = np.ones(len(samples_y1.index))
    predicted_y1 = predictor.predict(samples_y1)

    samples_y0 = samples.copy()
    samples_y0[treatment_col] = np.zeros(len(samples_y0.index))
    predicted_y0 = predictor.predict(samples_y0)

    f1 = df[outcome_col] * df[treatment_col] + predicted_y1 * (1 - df[treatment_col])
    f0 = df[outcome_col] * (1 - df[treatment_col]) + predicted_y0 * df[treatment_col]
    return np.mean(f1 - f0)


if __name__ == "__main__":
    data_path = os.path.join(PROJECT_ROOT, 'data')
    df_name = 'engineered_05.csv'
    outcome_column = 'totalScore'
    treatment_column = 'obesity_percentage'
    iv_column = 'pop_/n_re'

    ps_tolerance = 0.05
    df = get_engineered_dataframe(data_path, df_name, outcome_column,
                                  cols_to_keep=[treatment_column],
                                  cols_to_convert=[(treatment_column, 0.5)])
    df = calculate_propensity(df, treatment_column, outcome_column, LogisticRegression,
                              random_state=346, solver='lbfgs')
    df = trim_for_overlap(df, treatment_column, show_graphs=True, tolerance=ps_tolerance)
