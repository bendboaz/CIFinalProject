import os
import re

import pandas as pd
from pandas import read_csv
import numpy as np
from scipy import stats

from DataProcessing.Preprocessing import get_big_dataframe
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


if __name__ == "__main__":
    results_name = '10000x80_13-15_binary.csv'
    result_path = os.path.join(PROJECT_ROOT, 'data', 'results', results_name)
    results = read_csv(result_path)
    results = results.rename({results.columns[0]: 'Description'}, axis='columns')
    all_aggregations = aggregate_estimators(results)
    all_aggregations['total'] = aggregate_experiments(results.drop(['est_name'], axis=1))

    final_df = pd.concat(all_aggregations).to_latex()
    print(final_df)

