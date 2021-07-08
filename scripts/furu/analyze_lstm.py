# %%
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
def calc_max_index(df, col):
    return df[col].value_counts().idxmax()

def act_groupby_value_counsts(df1, df2, col_list):
    for col in col_list:
        df = pd.DataFrame(df1.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col=col), columns=[col]).reset_index()
        df2[col] = df[col].values
    
    return df2

# %%[markdown]
# # Train

# %%
phone_name = input('スマホの名前指定: ')

train_dr_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_derived.csv')
train_fix_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_Fix_add_columns.csv')
train_orient_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_OrientationDeg_add_columns.csv')
train_raw_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_Raw_add_columns.csv')
train_status_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_Status_add_columns.csv')
train_acc_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_UncalAccel_add_columns.csv')
train_gyro_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_UncalGyro_add_columns.csv')
train_mag_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_UncalMag_add_columns.csv')
train_gt_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_gt.csv')


# shapeだけ表示
display(train_dr_df.shape)
display(train_fix_df.shape)
display(train_orient_df.shape)
display(train_raw_df.shape)
display(train_status_df.shape)
display(train_acc_df.shape)
display(train_gyro_df.shape)
display(train_mag_df.shape)
display(train_gt_df.shape)

# %%[markdown]
# ## derived

# %%
# derived empty
print(train_dr_df.empty)

# %%
train_dr_df
# %%
# Labelig signalType
lenc = LabelEncoder()
train_dr_df['signalType'] = lenc.fit_transform(train_dr_df['signalType'])

# %%
# gropuby mean
train_dr_df_mean = train_dr_df.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_dr_df_mean

# %%
# gropuby (collectionName, phoneName)
new_train_dr_df = pd.merge(train_dr_df_mean, 
                    train_dr_df.groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']],
                    on='millisSinceGpsEpoch')
new_train_dr_df

# %%[markdown]
# ## Fix

# %%
# fix empty
print(train_fix_df.empty)
# true

# %%
train_fix_df # 空

# %%[markdown]
# ## OrientationDeg

# %%
# orientation_deg empty
print(train_orient_df.empty)
# %%
train_orient_df # groupby必要なし

# %%
# new
new_train_orient_df = train_orient_df.copy()

# %%[markdown]
# ## Raw

# %%
# raw empty
print(train_raw_df.empty)

# %%
train_raw_df

# %%
# CodeType unique
print(train_raw_df['CodeType'].unique())

# %%
# Labeling CodeType
train_raw_df['CodeType'] = train_raw_df['CodeType'].fillna(0)
train_raw_df.loc[train_raw_df['CodeType']=='C', 'CodeType'] = 1
train_raw_df.loc[train_raw_df['CodeType']=='Q', 'CodeType'] = 2

# %%
# gropuby mean
train_raw_df_mean = train_raw_df.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_raw_df_mean

# %%
train_raw_df.info()

# %%
# 全欠損カラムを削除
train_raw_df_mean_dropped = train_raw_df_mean.dropna(axis=1, how='all')
train_raw_df_mean_dropped

# %%
train_raw_df_mean_dropped.info()

# %%
# 一部欠損を補完
train_raw_df_mean_dropped['AgcDb'] = train_raw_df_mean_dropped['AgcDb'].fillna(train_raw_df['AgcDb'].mean())
train_raw_df_mean_dropped

# %%
# gropuby (collectionName, phoneName)
new_train_raw_df = pd.merge(train_raw_df_mean_dropped, 
                    train_raw_df.groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']],
                    on='millisSinceGpsEpoch')
# new
new_train_raw_df

# %%[markdown]
# ## Status

# %%
# status empty
print(train_status_df.empty)

# %%
train_status_df

# %%
train_status_df_mean = train_status_df.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_status_df_mean

# %%
train_status_df.info()

# %%
# 全欠損カラムを削除
train_status_df_mean_dropped = train_status_df_mean.dropna(axis=1, how='all')
train_status_df_mean_dropped

# %%
train_status_df_mean_dropped.info()

# %%
# 一部欠損を補完
train_status_df_mean_dropped['CarrierFrequencyHz'] = train_status_df_mean_dropped['CarrierFrequencyHz'].fillna(train_status_df['CarrierFrequencyHz'].mean())
train_status_df_mean_dropped

# %%
# gropuby (collectionName, phoneName)
new_train_status_df = pd.merge(train_status_df_mean_dropped, 
                    train_status_df.groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']],
                    on='millisSinceGpsEpoch')
# new
new_train_status_df

# %%[markdown]
# ## UncalAccel

# %%
# accel empty
print(train_acc_df.empty)

# %%
train_acc_df # groupby必要なし

# %%
# new
new_train_acc_df = train_acc_df.copy()

# %%[markdown]
# ## UncalGyro

# %%
# gyro empty
print(train_gyro_df.empty)

# %%
train_gyro_df # groupby必要なし

# %%
# new
new_train_gyro_df = train_gyro_df.copy()

# %%[markdown]
# ## UncalMag

# %%
# gyro empty
print(train_mag_df.empty)

# %%
train_mag_df # groupby必要なし

# %%
# new
new_train_mag_df = train_mag_df.copy()


# %%
# category変数はもっとも多い値
train_gnss_df_mean = act_groupby_value_counsts(train_gnss_df_category, train_gnss_df_mean, gnss_list)
# %%
display(train_gnss_df_mean)

# %%
# phoneNameとcollectionNameを結合
merged_train_gnss_df = pd.merge_asof(train_gnss_df_mean, 
                            train_dr_df[['phoneName', 'collectionName', 'millisSinceGpsEpoch']], 
                            on='millisSinceGpsEpoch', 
                            direction='nearest')
merged_train_gnss_df

# %%[markdown]
# ## ground_truth
# %%
train_gt_df

# %%[markdown]
# ## 結合

# %%
# 結合メンバー
display(train_dr_df_mean)
display(merged_train_gnss_df)
display(train_gt_df[['latDeg', 'lngDeg']])

# %%
train_gnssgt_df = pd.merge_asof(merged_train_gnss_df, 
                        train_gt_df[['latDeg', 'lngDeg', 'millisSinceGpsEpoch', 'phoneName', 'collectionName']], 
                        on='millisSinceGpsEpoch',
                        by=['phoneName', 'collectionName'],
                        direction='nearest',
                        tolerance=1000
                        )
train_df = pd.merge_asof(train_gnssgt_df, train_dr_df_mean, 
                    on = 'millisSinceGpsEpoch',
                    by=['phoneName', 'collectionName'],
                    suffixes=('_gnss', '_derived'),
                    direction='nearest',
                    tolerance=10000 # Mi8は1000にすると欠損
                    )
train_df

# %%
# tolerance=1000にしたとき、derivedと結合する欠損するやつ
train_df.loc[train_df['Svid_derived'].isnull(), ['millisSinceGpsEpoch']]

# %%
train_df[3835:3840]

###########################################################################################
# %%[markdown]
# # Test
# ## derived
# %%
# signalType
lenc = LabelEncoder()
test_dr_df['signalType'] = lenc.fit_transform(test_dr_df['signalType'])
# %%
# groupy by for derived
derived_list = ['phoneName', 'collectionName', 'ConstellationType', 'Svid', 'signalType']

test_dr_df_4groupby = test_dr_df.drop(derived_list, axis=1)
test_dr_df_category = test_dr_df[['millisSinceGpsEpoch']+derived_list]

# %%
# gropuby mean
test_dr_df_mean = test_dr_df_4groupby.groupby('millisSinceGpsEpoch', as_index=False).mean()
test_dr_df_mean

# %%
# category変数はもっとも多い値
test_dr_df_mean = act_groupby_value_counsts(test_dr_df_category, test_dr_df_mean, derived_list)
# %%
test_dr_df_mean

# %%[markdown]
# ## gnss

# %%
# CodeType
test_gnss_df['CodeType'].unique()
# %%
test_gnss_df = test_gnss_df.fillna(0)
test_gnss_df.loc[test_gnss_df['CodeType']=='C', 'CodeType'] = 1
test_gnss_df.loc[test_gnss_df['CodeType']=='Q', 'CodeType'] = 2
test_gnss_df[['CodeType']]

# %%
test_gnss_df['HasEphemerisData'].unique()

# %%
# groupy by for gnss
gnss_list = ['Svid', 
            'AccumulatedDeltaRangeState',
            'MultipathIndicator',
            'ConstellationType',
            'ConstellationTypeStatus',
            'SvidStatus',
            'AzimuthDegrees',
            'ElevationDegrees',
            'UsedInFix',
            'HasAlmanacData',
            'HasEphemerisData',
            'CodeType'
            ]

test_gnss_df_4groupby = test_gnss_df.drop(gnss_list, axis=1)
test_gnss_df_category = test_gnss_df[['millisSinceGpsEpoch']+gnss_list]

# %%
# gropuby mean
test_gnss_df_mean = test_gnss_df_4groupby.groupby('millisSinceGpsEpoch', as_index=False).mean()
test_gnss_df_mean

# %%
# category変数はもっとも多い値
test_gnss_df_mean = act_groupby_value_counsts(test_gnss_df_category, test_gnss_df_mean, gnss_list)
# %%
display(test_gnss_df_mean)

# %%
# phoneNameとcollectionNameを結合
merged_test_gnss_df = pd.merge_asof(test_gnss_df_mean, 
                            test_dr_df[['phoneName', 'collectionName', 'millisSinceGpsEpoch']], 
                            on='millisSinceGpsEpoch', 
                            direction='nearest')
merged_test_gnss_df

# %%[markdown]
# ## ground_truth
# - testにはない

# %%[markdown]
# ## 結合

# %%
# 結合メンバー
display(test_dr_df_mean)
display(merged_test_gnss_df)

# %%

test_df = pd.merge_asof(merged_test_gnss_df, test_dr_df_mean, 
                    on = 'millisSinceGpsEpoch',
                    by=['phoneName', 'collectionName'],
                    suffixes=('_gnss', '_derived'),
                    direction='nearest',
                    tolerance=1000 # tolelance=100にすると欠損
                    )
test_df

# %%
# tolerance=100にしたとき、derivedと結合する欠損するやつ
test_df.loc[test_df['Svid_derived'].isnull(), ['millisSinceGpsEpoch']]

# %%
test_df[405:435]

############################################################################################
# %%[markdown]
# # 保存
# %%
name = str(dt.now().strftime('%Y-%m-%d_%H'))

train_df.to_csv(f'../../data/processed/confirm/train/{name}_{phone_name}.csv', index=False)
test_df.to_csv(f'../../data/processed/confirm/test/{name}_{phone_name}.csv', index=False)

# %%
