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


def representative_features(df, label_column, idx2colname, necessary_features=None):
    if label_column in df.columns:
        df = df.drop([label_column], axis=1)
    if necessary_features is None:
        necessary_features = []
    features = necessary_features[:]
    corr_matrix = np.abs(df.corr().to_numpy())
    feature_clusters, cluster_indices = get_feature_clusters(df, label_column, idx2colname, 13)
    for cluster, indices in zip(feature_clusters, cluster_indices):
        sign_mask = np.ones(len(df.columns))
        for feature_ind in range(len(df.columns)):
            if feature_ind not in indices:
                sign_mask[feature_ind] *= -1

        cluster_elements = corr_matrix[:, indices]
        scores = np.dot(sign_mask, cluster_elements)
        chosen_feature = np.argmax(scores)
        if chosen_feature not in necessary_features:
            features.append(df.columns[indices[chosen_feature]])

    return features


def convert_numeric_to_binary(df, features_list):
    relevant_cols = df.loc[:, features_list]
    medians = relevant_cols.median(axis=0)
    for median, feature in zip(medians, features_list):
        df.loc[:, feature] = (df[feature] >= median).astype(int)
    return df


def add_density_feature(df, top_label, bottom_label):
    new_column = df[top_label] / df[bottom_label]
    new_name = f"{top_label:.4}/{bottom_label:.4}"
    df.loc[:, new_name] = new_column
    return df, new_name


def get_engineered_dataframe(data_path, label_column, cols_to_keep=None):
    save_path = os.path.join(data_path, 'processed', 'engineered_v1.csv')
    if os.path.isfile(save_path):
        df = pd.read_csv(save_path, index_col='state')
        return df

    print("Dataframe not found, generating it...")
    if cols_to_keep is None:
        cols_to_keep = []

    df = get_big_dataframe(data_path)
    idx2colname = {idx: colname for idx, colname in enumerate(df.columns)}
    df, density_col = add_density_feature(df, 'pop_sum', 'n_restaurants')
    cols_to_keep.append(density_col)
    features = representative_features(df.drop([label_column], axis=1), label_column, idx2colname,
                                       necessary_features=cols_to_keep)
    features.append(label_column)
    filtered_df = df.loc[:, features]
    filtered_df = convert_numeric_to_binary(filtered_df, cols_to_keep + [label_column])

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
