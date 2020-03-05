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


def att_ipw(df: pd.DataFrame, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    sigma_t_y = sum(treated[outcome_col])
    sigma_t = len(treated.index)
    control = df[df[treatment_col] == 0]
    control = control.assign(PS_ODDS=control["PS"].apply(compute_odds))
    sigma_tinv_y = sum(control[outcome_col].multiply(control["PS_ODDS"]))
    sigma_tinv = sum(control["PS_ODDS"])
    return (sigma_t_y / sigma_t) - (sigma_tinv_y / sigma_tinv)


def att_s(df, treatment_col, outcome_col):
    factual_y1 = df[df[treatment_col] == 1][outcome_col]
    samples = df.drop(["PS", outcome_col], axis=1)
    labels = df[outcome_col]
    predictor = Ridge(alpha=0.08, normalize=True, random_state=321)
    predictor.fit(samples, labels)

    treated = samples[samples[treatment_col] == 1]
    treated_y0 = treated.copy()
    treated_y0[treatment_col] = np.zeros(len(treated.index))
    predicted_y0 = predictor.predict(treated_y0)
    return np.mean(factual_y1 - predicted_y0)


def att_t(df, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    c_predictor = Ridge(alpha=0.15, normalize=True, random_state=321)
    c_predictor.fit(control.drop([treatment_col, outcome_col, "PS"], axis=1), control[outcome_col])

    predicted_y0 = c_predictor.predict(treated.drop([treatment_col, outcome_col, "PS"], axis=1))
    return np.mean(treated[outcome_col] - predicted_y0)


def att_match(df, treatment_col, outcome_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    c_predictor = KNeighborsRegressor(1)
    c_predictor.fit(control.drop([treatment_col, outcome_col, "PS"], axis=1), control[outcome_col])
    return np.mean(treated[outcome_col] - c_predictor.predict(treated.drop([treatment_col, outcome_col, "PS"], axis=1)))


def att_magic(df, treatment_col, outcome_col):
    """
    Added propensity score to the t-learner covariates. Our approach:
    https://shir663.files.wordpress.com/2016/12/my-meme.jpg
    :param df:
    :return:
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    t_predictor = Ridge(alpha=0.15, normalize=True, random_state=321)
    t_predictor.fit(treated.drop([treatment_col, outcome_col], axis=1), treated[outcome_col])
    c_predictor = Ridge(alpha=0.15, normalize=True, random_state=321)
    c_predictor.fit(control.drop([treatment_col, outcome_col], axis=1), control[outcome_col])

    return np.mean(treated[outcome_col] -
                   ((t_predictor.predict(treated.drop([treatment_col, outcome_col], axis=1)) -
                     treated[outcome_col]) / treated['PS']) -
                   c_predictor.predict(treated.drop([treatment_col, outcome_col], axis=1)))


if __name__ == "__main__":
    outcome_column = 'totalScore'
    treatment_column = 'obesity_percentage'
    iv_column = 'pop_/n_re'
    df = get_engineered_dataframe(os.path.join(PROJECT_ROOT, 'data'), outcome_column)
    df = calculate_propensity(df, treatment_column, outcome_column, LogisticRegression, random_state=346, solver='lbfgs')
    df = trim_for_overlap(df, treatment_column, show_graphs=True)

    # xgboost_params = dict(max_depth=8, n_estimators=75, n_iter_no_change=5, random_state=212)
    # att_calculation_methods = [att_ipw, att_s, att_t, att_match, att_magic]
    # results_df = pd.DataFrame(np.zeros((len(att_calculation_methods), 3)), columns=["Type", "data1", "data2"])
    # results_df["Type"] = results_df["Type"].astype(int)
    # for ds_num in [1, 2]:
    #     print(f"Calculating ATT for dataset {ds_num}")
    #     dataset = read_dataset(ds_num)
    #     dataset = calculate_propensity(dataset, GradientBoostingClassifier, **xgboost_params)
    #     dataset = trim_for_overlap(dataset)
    #     for method_idx, method in enumerate(att_calculation_methods):
    #         results_df.loc[method_idx, "Type"] = method_idx + 1
    #         att_hat = method(dataset)
    #         results_df.loc[method_idx, f"data{ds_num}"] = att_hat
    #
    # print("\n\n")
    # print(results_df)
    # results_df.to_csv('ATT_results.csv', index=False)
    # print("\n\nBye!")
