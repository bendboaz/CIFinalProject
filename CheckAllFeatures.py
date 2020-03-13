import os
import re

import pandas as pd
from pandas import read_csv
import numpy as np
from scipy import stats

from DataProcessing.Preprocessing import get_big_dataframe
from DataProcessing.Features import get_engineered_dataframe
from Utils import PROJECT_ROOT
from Experiments import _run_experiment


def check_over_all_features():
    data_path = os.path.join(PROJECT_ROOT, 'data')
    big_df = get_big_dataframe(data_path)

    for n_clusters in [13, 14, 15]:
        print(_run_experiment(big_df, n_clusters, None, n_clusters, trimming_tolerance=0.05))


def get_confidence_interval(results):
    confidence = 0.05
    left_end = np.percentile(results.to_list(), confidence * 100 / 2, interpolation='higher')
    right_end = np.percentile(results.to_list(), (1 - confidence / 2) * 100, interpolation='higher')
    return f"{left_end:.3}, {right_end:.3}"


def aggregate_experiments(results):
    total_metrics = results.agg({col: ['mean', np.std, get_confidence_interval]
                                 for col in filter(lambda x: x != 'Description', results.columns)})
    return total_metrics


def aggregate_estimators(results):
    regex = r'exp\d*_(\w+)'
    results['est_name'] = results['Description'].apply(get_est_name, regex=regex)
    results_by_est = results.groupby(by='est_name')
    aggregated_by_est = {name: aggregate_experiments(result.drop(['est_name'], axis=1))
                         for name, result in results_by_est}
    return aggregated_by_est


def get_est_name(row, regex):
    return re.match(regex, row).group(1)


def compute_iv_sem_estimate(eng_df, label_col, treatment_col, iv_col):
    iv_cov_label = eng_df[iv_col].cov(eng_df[label_col])
    iv_cov_treatment = eng_df[iv_col].cov(eng_df[treatment_col])
    return iv_cov_label / iv_cov_treatment


def compute_iv_constant_estimate(eng_df, label_col, treatment_col, iv_col):
    (zero, with_iv), (one, without_iv) = eng_df.groupby(by=iv_col)
    assert zero == 0.0 and one == 1
    iv_on_label = with_iv[label_col].mean() - without_iv[label_col].mean()
    iv_on_treatment = with_iv[treatment_col].mean() - without_iv[treatment_col].mean()
    return iv_on_label / iv_on_treatment


if __name__ == "__main__":
    results_name = '10000x80_13-15.csv'
    result_path = os.path.join(PROJECT_ROOT, 'data', 'results', results_name)
    results = read_csv(result_path)
    results = results.rename({results.columns[0]: 'Description'}, axis='columns')
    results['IV-based_effect'] = results['IV->Y'] / results['IV->T']
    all_aggregations = aggregate_estimators(results)
    all_aggregations['total'] = aggregate_experiments(results.drop(['est_name'], axis=1))

    final_df = pd.concat(all_aggregations).unstack()
    print(final_df['T->Y'].transpose().to_latex())
    print(final_df['IV->T'].transpose().to_latex())
    print(final_df['IV-based_effect'].transpose().to_latex())

    data_path = os.path.join(PROJECT_ROOT, 'data')
    big_df = get_big_dataframe(data_path)

    label_col = 'totalScore'
    treatment_col = 'obesity_percentage'
    eng_df = get_engineered_dataframe(data_path, 'all_states', big_df, 'totalScore',
                                      cols_to_keep=[label_col, treatment_col], cols_to_convert=[(treatment_col, 0.5)])
    print(f"Correlation between IV and T:", eng_df['obesity_percentage'].corr(eng_df['pop_/n_re']))
    print(f"Correlation between IV and Y:", eng_df['totalScore'].corr(eng_df['pop_/n_re']))

    print("ATE computed on aggregated effects:", final_df['IV->Y']['mean']/final_df['IV->T']['mean'])
    iv_col = 'pop_/n_re'
    print(compute_iv_sem_estimate(eng_df, label_col, treatment_col, iv_col))
    print(compute_iv_constant_estimate(eng_df, label_col, treatment_col, iv_col))

