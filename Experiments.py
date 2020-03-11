import os
from itertools import islice
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DataProcessing.Preprocessing import get_big_dataframe
from DataProcessing.Features import get_engineered_dataframe
from Utils import PROJECT_ROOT, trim_for_overlap, check_indicative_features
from EffectComputations import ate_ipw, ate_magic, ate_match, ate_s, calculate_propensity


LABEL_COLUMN = 'totalScore'
TREATMENT_COLUMN = 'obesity_percentage'  # TODO: Decide on a name.
IV_COL = 'pop_/n_re'

ALL_ESTIMATORS = [ate_ipw, ate_magic, ate_match, ate_s]
DESIRED_EFFECTS = {'IV->T': (IV_COL, TREATMENT_COLUMN),
                   'IV->Y': (IV_COL, LABEL_COLUMN),
                   'T->Y': (TREATMENT_COLUMN, LABEL_COLUMN)}


def _run_experiment(df, version_num, features, n_clusters, obesity_cutoff=0.5, trimming_tolerance=0.0):
    results_index = [f"exp{version_num}_{estimator.__name__}" for estimator in ALL_ESTIMATORS]

    results_df = pd.DataFrame(np.zeros((len(ALL_ESTIMATORS), len(DESIRED_EFFECTS))), index=results_index,
                              columns=DESIRED_EFFECTS, dtype=float)

    data_path = os.path.join(PROJECT_ROOT, 'data')
    eng_df = get_engineered_dataframe(data_path, f"engineered_exp{version_num}.csv", df, LABEL_COLUMN,
                                      cols_to_keep=[LABEL_COLUMN, TREATMENT_COLUMN],
                                      cols_to_convert=[(TREATMENT_COLUMN, obesity_cutoff)],
                                      keep_features=features,
                                      n_clusters=n_clusters, flag_must_create_new_file=True)

    # Check to see if the chosen features are indicative enough.
    classifier_score = check_indicative_features(eng_df, LABEL_COLUMN)
    if classifier_score < 0.7:
        # Features not indicative enough, skip experiment.
        return None

    df = calculate_propensity(eng_df, TREATMENT_COLUMN, LABEL_COLUMN, LogisticRegression,
                              random_state=346, solver='lbfgs')

    c_pre, t_pre = df[TREATMENT_COLUMN].value_counts().sort_index(ascending=True)
    df = trim_for_overlap(df, TREATMENT_COLUMN, tolerance=trimming_tolerance)
    c_post, t_post = df[TREATMENT_COLUMN].value_counts().sort_index(ascending=True)

    if c_post < (c_pre / 2) or t_post < (t_pre / 2):
        # Skipping experiment
        return None

    results = map(lambda estimator: list(map(lambda pair: estimator(df, pair[1][0], pair[1][1]),
                                             DESIRED_EFFECTS.items())),
                  ALL_ESTIMATORS)

    for est, result in zip(ALL_ESTIMATORS, results):
        results_df.loc[f"exp{version_num}_{est.__name__}"] = result

    return results_df


def generate_experiments(n_features_total, feature_set_size, min_clusters, max_clusters, idx2colname):
    all_feature_indices = np.arange(n_features_total)
    exp_counter = 1
    while True:
        feature_indices = np.random.permutation(all_feature_indices)[:feature_set_size]
        features = list(map(idx2colname.__getitem__, feature_indices))
        n_clusters = np.random.randint(min_clusters, max_clusters)
        yield {'version_num': exp_counter, 'features': features, 'n_clusters': n_clusters}
        exp_counter += 1


def run_experiments(n_experiments, feature_set_size, min_clusters, max_clusters):
    data_path = os.path.join(PROJECT_ROOT, 'data')
    big_df = get_big_dataframe(data_path)

    idx2colname = {idx: colname for idx, colname in enumerate(big_df.columns)}
    exp_generator = generate_experiments(len(big_df.columns), feature_set_size, min_clusters, max_clusters, idx2colname)
    experiment_configurations = {}
    results = None
    skipped_experiments = 0
    for params in tqdm(islice(exp_generator, n_experiments), desc='experiments', total=n_experiments, unit='exp'):
        experiment_configurations[params['version_num']] = {'features': params['features'],
                                                            'n_clusters': params['n_clusters']}
        result = _run_experiment(big_df, **params)
        if result is None:
            skipped_experiments += 1
            continue
        if results is None:
            results = result
        else:
            results = results.append(result)

        exp_name = f'{n_experiments}x{feature_set_size}_{min_clusters}-{max_clusters}.csv'
        results.to_csv(os.path.join(data_path, 'results', exp_name))
        with open(os.path.join(data_path, 'results', f"{exp_name[:-4]}.config"), 'w+') as f:
            f.write(json.dumps(experiment_configurations))

    print(f"Skipped {skipped_experiments} experiments!")
    return results


if __name__ == "__main__":
    run_experiments(int(1e4), feature_set_size=80, min_clusters=13, max_clusters=15)
