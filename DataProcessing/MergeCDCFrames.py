import os

import pandas as pd
from tqdm import tqdm

from Utils import PROJECT_ROOT

if __name__ == "__main__":
    cdc_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'cdc')
    partition = 'behavior'
    partition_path = os.path.join(cdc_path, partition)

    list_files = os.listdir(partition_path)
    file1 = list_files[0]
    new_df = pd.read_csv(os.path.join(partition_path, file1), usecols=['LocationDesc', 'Data_Value'])
    new_df = new_df.rename({'LocationDesc': 'state', 'Data_Value': file1[:-4]}, axis='columns').set_index('state')

    others = [pd.read_csv(os.path.join(partition_path, file), usecols=['LocationDesc', 'Data_Value'])
              for file in list_files[1:]]
    others = list(map(lambda pair: pair[0].rename({'LocationDesc': 'state', 'Data_Value': pair[1][:-4]}, axis='columns')
                      .dropna(axis=0)
                      .set_index('state'),
                      zip(others, list_files[1:])))
    print("Finished reading files")
    new_df = new_df.join(others, how='inner')

    new_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'raw', f'{partition}.csv'))
