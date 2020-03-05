import pandas as pd
import numpy as np
import os

from Utils import PROJECT_ROOT


def general_data_df(path):
    print("Generating monetary data frame...")
    columns_to_keep = ['state', 'state_ab', 'lat', 'lng', 'pop', 'male_pop', 'female_pop', 'rent_mean', 'rent_median', 'rent_stdev',
                       'rent_gt_10', 'rent_gt_20', 'rent_gt_30', 'rent_gt_40', 'rent_gt_50', 'hi_mean', 'hi_median',
                       'hi_stdev', 'family_mean', 'family_median', 'family_stdev', 'family_sample_weight',
                       'hc_mortgage_mean', 'hc_mortgage_median', 'hc_mortgage_stdev', 'hc_mean', 'hc_median',
                       'hc_stdev', 'home_equity_second_mortgage', 'second_mortgage', 'home_equity', 'debt',
                       'second_mortgage_cdf', 'home_equity_cdf', 'debt_cdf', 'hs_degree', 'hs_degree_male',
                       'hs_degree_female', 'male_age_mean', 'male_age_median', 'male_age_stdev', 'female_age_mean',
                       'female_age_median', 'female_age_stdev', 'married', 'married_snp', 'separated', 'divorced']
    aggregation_method = {'lat': ['min', 'max'], 'lng': ['min', 'max'], 'pop': 'sum', 'male_pop': 'sum',
                          'female_pop': 'sum', 'rent_mean': ['mean', np.median], 'rent_median': ['mean', np.median],
                          'rent_stdev': 'mean', 'rent_gt_10': ['mean', np.median], 'rent_gt_20': ['mean', np.median],
                          'rent_gt_30': ['mean', np.median], 'rent_gt_40': ['mean', np.median],
                          'rent_gt_50': ['mean', np.median], 'hi_mean': ['mean', np.median],
                          'hi_median': ['mean', np.median], 'hi_stdev': 'mean', 'family_mean': ['mean', np.median],
                          'family_median': ['mean', np.median], 'family_stdev': 'mean',
                          'hc_mortgage_mean': ['mean', np.median], 'hc_mortgage_median': ['mean', np.median],
                          'hc_mortgage_stdev': 'mean', 'hc_mean': ['mean', np.median], 'hc_median': ['mean', np.median],
                          'hc_stdev': 'mean', 'home_equity_second_mortgage': ['mean', np.median],
                          'second_mortgage': ['mean', np.median], 'home_equity': ['mean', np.median],
                          'debt': ['mean', np.median], 'second_mortgage_cdf': ['mean', np.median],
                          'home_equity_cdf': ['mean', np.median], 'debt_cdf': ['mean', np.median],
                          'hs_degree': ['mean', np.median], 'hs_degree_male': ['mean', np.median],
                          'hs_degree_female': ['mean', np.median], 'male_age_mean': ['mean', np.median],
                          'male_age_median': ['mean', np.median], 'male_age_stdev': 'mean',
                          'female_age_mean': ['mean', np.median], 'female_age_median': ['mean', np.median],
                          'female_age_stdev': 'mean', 'married': ['mean', np.median], 'separated': ['mean', np.median],
                          'divorced': ['mean', np.median]}
    df = pd.read_csv(path, usecols=columns_to_keep, error_bad_lines=False, engine='python')
    state2ab = dict()
    for _, row in df.iterrows():
        state2ab[row.state] = row.state_ab
    df = df.groupby(by='state', as_index=False).agg(aggregation_method)
    mi = df.columns
    ind = pd.Index([e[0] + "_" + e[1] for e in mi.tolist()])
    df.columns = ind
    df = df.rename(columns={'state_': 'state', 'state_ab_':'state_ab'})
    df = df.assign(state_ab=df['state'].apply(state2ab.__getitem__))
    # TODO: make the proportions correct
    return df


def get_physical_activities_df(path):
    print("Generating physical activity data frame...")
    df = pd.read_csv(path, usecols=['LocationDesc', 'Data_Value'], error_bad_lines=False)
    df = df.rename(columns={'LocationDesc': 'state', 'Data_Value': 'physically_active'})
    df = df[df['physically_active'] != '~']
    df = df.astype({'physically_active': 'float'})
    df['physically_active'] = df.physically_active.apply(lambda x: x / 100)
    return df


def get_obesity_df(path):
    print("Generating obesity data frame...")
    df = pd.read_csv(path, usecols=['LocationDesc', 'Data_Value'], error_bad_lines=False)
    df = df.rename(columns={'LocationDesc': 'state', 'Data_Value': 'obesity_percentage'})
    df = df[df['obesity_percentage'] != '~']
    df = df.astype({'obesity_percentage': 'float'})
    df['obesity_percentage'] = df['obesity_percentage'].apply(lambda x: x / 100)
    return df


def get_fastfood_df(path):
    print("Generating fast food data frame...")
    df = pd.read_csv(path)
    return df


def get_happiness_df(path):
    print("Generating happiness data frame...")
    df = pd.read_csv(path, error_bad_lines=False, usecols=['State', 'totalScore'])
    df = df.rename(columns={'State': 'state'})
    return df


def get_big_dataframe(data_path):
    final_path = os.path.join(data_path, 'processed', 'dataset_v1.pkl')
    if os.path.isfile(final_path):
        return pd.read_pickle(final_path)

    print("Dataframe not found, need to generate its components")

    raw_path = os.path.join(data_path, 'raw')
    general = general_data_df(os.path.join(raw_path, 'real_estate_db.csv')).set_index('state', drop=False)
    physical = get_physical_activities_df(os.path.join(raw_path, "active.csv")).set_index('state')
    obesity = get_obesity_df(os.path.join(raw_path, 'fat.csv')).set_index('state')
    fastfood = get_fastfood_df(os.path.join(raw_path, 'fast_food.csv')).set_index('state_ab')
    happiness = get_happiness_df(os.path.join(raw_path, 'happiness.csv')).set_index('state')
    big_df = general.join(physical, how='inner')
    big_df = big_df.join(obesity, how='inner')
    big_df = big_df.join(happiness, how='inner')
    big_df = big_df.set_index('state_ab').join(fastfood, how='inner')
    big_df = big_df.set_index('state')
    big_df.to_pickle(final_path)
    return big_df


if __name__ == "__main__":
    big_df = get_big_dataframe(os.path.join(PROJECT_ROOT, 'data'))
    big_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'dataset_v1.csv'))
    print(big_df)

