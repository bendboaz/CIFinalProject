import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


PROJECT_ROOT = "C:\\Users\\bendb\\PycharmProjects\\FatHappy"


def overlap_graph(control, treated):
    plt.hist(control["PS"].tolist(), 20, range=(0, 1), label="control")
    plt.hist(treated["PS"].tolist(), 20, range=(0, 1), label="treated")
    plt.legend()
    plt.show()


def trim_for_overlap(df: pd.DataFrame, treatment_label, show_graphs=False, tolerance=0.0):
    control = df[df[treatment_label] == 0]
    treated = df[df[treatment_label] == 1]
    if show_graphs:
        overlap_graph(control, treated)

    min_common_ps = max(control["PS"].min(), treated["PS"].min())
    max_common_ps = min(control["PS"].max(), treated["PS"].max())
    # print(f"min ps = {min_common_ps}, max ps = {max_common_ps}")
    trimmed_dataset = df[(df["PS"] >= min_common_ps - tolerance) & (df["PS"] <= max_common_ps + tolerance)]
    trimmed_control = trimmed_dataset[trimmed_dataset[treatment_label] == 0]
    trimmed_treated = trimmed_dataset[trimmed_dataset[treatment_label] == 1]
    # print(f"Trimmed {len(control.index) - len(trimmed_control.index)} rows out of {len(control.index)} in control")
    # print(f"Trimmed {len(treated.index) - len(trimmed_treated.index)} rows out of {len(treated.index)} in treated")
    if show_graphs:
        overlap_graph(trimmed_control, trimmed_treated)
    return trimmed_dataset


def check_indicative_features(df, label_column):
    classifier = RandomForestRegressor(n_estimators=15, max_depth=4)
    X_train, X_test, y_train, y_test = train_test_split(df.drop([label_column], axis=1).to_numpy(),
                                                        df[label_column].to_numpy(),
                                                        train_size=0.8, random_state=321)
    classifier.fit(X_train, y_train)
    return classifier.score(X_test, y_test)
