import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster.spectral import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from DataProcessing.Preprocessing import get_big_dataframe
from Utils import PROJECT_ROOT


def get_feature_clusters(df, label_column, idx2colname, n_clusters=13):
    if label_column in df.columns:
        df = df.drop([label_column], axis=1)
    clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=346345)
    cluster_argindices = clusterer.fit_predict(np.abs(df.corr()))
    cluster_indices = [np.where(cluster_argindices == cluster_idx)[0] for cluster_idx in range(0, n_clusters)]
    name_clusters = map(lambda x: list(map(idx2colname.__getitem__, x)), cluster_indices)
    return name_clusters, cluster_indices


def representative_features(df, label_column, necessary_features=None, n_clusters=13):
    if label_column in df.columns:
        df = df.drop([label_column], axis=1)
    if necessary_features is None:
        necessary_features = []
    features = necessary_features[:]
    corr_matrix = np.abs(df.corr().to_numpy())
    idx2colname = {idx: colname for idx, colname in enumerate(df.columns)}
    feature_clusters, cluster_indices = get_feature_clusters(df, label_column, idx2colname, n_clusters)
    for cluster, indices in zip(feature_clusters, cluster_indices):
        sign_mask = np.ones(len(df.columns))
        for feature_ind in range(len(df.columns)):
            if feature_ind not in indices:
                sign_mask[feature_ind] *= -1

        cluster_elements = corr_matrix[:, indices]
        scores = np.dot(sign_mask, cluster_elements)
        chosen_feature = np.argmax(scores)
        if idx2colname[indices[chosen_feature]] not in features:
            features.append(idx2colname[indices[chosen_feature]])

    return features


def convert_numeric_to_binary(df, features_list):
    for feature, percentile in features_list:
        df.loc[:, feature] = (df[feature] >= df[feature].quantile(percentile)).astype(int)
    return df


def add_density_feature(df, top_label, bottom_label):
    new_column = df[top_label] / df[bottom_label]
    new_name = f"{top_label:.4}/{bottom_label:.4}"
    df.loc[:, new_name] = new_column
    return df, new_name


def get_engineered_dataframe(data_path, dataset_name, big_df, label_column, label_percentile=0.5, cols_to_keep=None,
                             cols_to_convert=None, keep_features=None, n_clusters=13, flag_must_create_new_file=False):
    """
    :param data_path: Path to the 'data' directory.
    :param dataset_name: Name for the dataframe to be saved as.
    :param big_df: The dataframe containing all information (including labels, treatments and covariates).
    :param label_column: Name of the label column.
    :param label_percentile: Cutoff percentile for label column binarization.
    :param cols_to_keep: List of column names to keep while clustering.
    :param cols_to_convert: List of (colname, percentile) tuples.
    :param keep_features: If not None, only use the mentioned features instead of the entire dataframe.
    :param n_clusters: Number of clusters to use for dimensionality reduction.
    :param flag_must_create_new_file: flag if the func called from the experiments, we have to create new df
    :return:
    """
    save_path = os.path.join(data_path, 'processed', dataset_name)
    if os.path.isfile(save_path) and not flag_must_create_new_file:
        df = pd.read_csv(save_path, index_col='state')
        return df

    # print("Dataframe not found, generating it...")
    if cols_to_keep is None:
        cols_to_keep = []

    if cols_to_convert is None:
        cols_to_convert = []

    df = big_df
    if keep_features is not None:
        indices = list(set(set(keep_features) |
                           set(cols_to_keep) |
                           {'pop_sum', 'n_restaurants', label_column, 'state'}))
        df = df.loc[:, indices]

    df, density_col = add_density_feature(df, 'pop_sum', 'n_restaurants')
    df = df.set_index('state')
    cols_to_keep.append(density_col)
    cols_to_convert.append((density_col, 0.5))

    features = representative_features(df.drop([label_column], axis=1), label_column,
                                       necessary_features=cols_to_keep, n_clusters=n_clusters)
    # features.append(label_column)
    filtered_df = df.loc[:, features]
    filtered_df = convert_numeric_to_binary(filtered_df, cols_to_convert)

    scaler = MinMaxScaler()
    filtered_df[filtered_df.columns] = scaler.fit_transform(filtered_df[filtered_df.columns])

    filtered_df.to_csv(save_path)

    return filtered_df


if __name__ == "__main__":
    RANDOM_SEEDS = [4, 85, 1120, 3425345623, 99487]
    # label_column = 'totalScore'
    label_column = 'obesity_percentage'
    filtered_df = get_engineered_dataframe(os.path.join(PROJECT_ROOT, 'data'), label_column)

    X_train, X_test, y_train, y_test = train_test_split(filtered_df.drop([label_column], axis=1),
                                                        filtered_df[label_column],
                                                        train_size=0.8, random_state=1352345)

    errors = []

    for seed in RANDOM_SEEDS:
        # np.random.seed(seed)
        regressor = MLPClassifier(hidden_layer_sizes=tuple(), activation='logistic', max_iter=70000, alpha=0.5,
                                  random_state=seed)
        regressor.fit(X_train, y_train)
        iter_error = np.where(regressor.predict(X_test) != y_test)
        iter_error = list(map(X_test.index.__getitem__, iter_error))
        errors.append(iter_error)
        print("Regressor score:", regressor.score(X_test, y_test))

    print('errors:')
    print(errors)

