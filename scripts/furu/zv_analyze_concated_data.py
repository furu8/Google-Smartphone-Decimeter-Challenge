# %%
import pandas as pd

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
%%time

basepath = '../../data/interim'

phone_names = (
    'Mi8',
    'Pixel4',
    'Pixel4Modded',
    'Pixel4XL',
    'Pixel4XLModded',
    'Pixel5',
    'SamsungS20Ultra'
)

data_dir =  (
    'train',
    'test'
)

for ddir in data_dir:
    concated_df = pd.DataFrame()
    for pname in phone_names:
        print(f'{ddir}, {pname}')
        df = pd.read_csv(f'{basepath}/{ddir}/groupbyed_{pname}.csv')
        concated_df = concated_df.append(df)

    concated_df = concated_df.reset_index(drop=True)
    display(concated_df)
    concated_df.to_csv(f'../../data/processed/{ddir}/{ddir}.csv', index=False)
# %%
# baselineとの行数確認
train_df = pd.read_csv('../../data/processed/train/train.csv')
baseline_train_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')

print(train_df.shape)
print(baseline_train_df.shape)

len(set(train_df['millisSinceGpsEpoch']) - set(baseline_train_df['millisSinceGpsEpoch']))

# %%
train_df.isnull().sum()

# %%
merged_df = pd.merge_asof(baseline_train_df[['millisSinceGpsEpoch', 'latDeg', 'lngDeg', 'phoneName', 'collectionName']].sort_values('millisSinceGpsEpoch'), 
            train_df.sort_values('millisSinceGpsEpoch'), 
            on='millisSinceGpsEpoch',
            by=['phoneName', 'collectionName'],
            suffixes=('_pred', ''),
            direction='nearest',
            tolerance=1000)

merged_df

# %%
merged_df.isnull().sum()
# %%
merged_df.to_csv('../../data/processed/train/train_merged_base.csv', index=False)

# %%
test_df = pd.read_csv('../../data/processed/test/test.csv')
baseline_test_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')

print(test_df.shape)
print(baseline_test_df.shape)
# %%
test_df.isnull().sum()

# %%
merged_df = pd.merge_asof(baseline_test_df[['millisSinceGpsEpoch', 'latDeg', 'lngDeg', 'phoneName', 'collectionName']].sort_values('millisSinceGpsEpoch'), 
            test_df.sort_values('millisSinceGpsEpoch'), 
            on='millisSinceGpsEpoch',
            by=['phoneName', 'collectionName'],
            suffixes=('_pred', ''),
            direction='nearest',
            tolerance=1000)

merged_df

# %%
merged_df.isnull().sum()
# %%
merged_df.to_csv('../../data/processed/test/test_merged_base.csv', index=False)
