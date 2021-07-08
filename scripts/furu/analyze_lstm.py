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

# %%[markdown]
# ## ground_truth

# %%
# ground truth empty
print(train_gt_df.empty)
# %%
train_gt_df

# %%
# new
# 必要なground truthだけ抽出
new_train_gt_df = train_gt_df[['latDeg', 'lngDeg', 'millisSinceGpsEpoch', 'phoneName', 'collectionName']].copy()

# %%[markdown]
# ## merge

# %%
# 結合しないメンバー
train_fix_df

# %%
# 結合メンバー
display(new_train_dr_df)
display(new_train_orient_df)
display(new_train_raw_df)
display(new_train_status_df)
display(new_train_acc_df)
display(new_train_gyro_df)
display(new_train_mag_df)
display(new_train_gt_df)


# %%
def merge_df_dict(df_dict, merge_base_col='Raw'):
    base_df = df_dict[merge_base_col]
    df_dict.pop(merge_base_col)
    base_df['elapsedRealtimeNanos'] = 0 # suffixesのために、適当な値を入れる
    for key in df_dict.keys():
        if not df_dict[key] is None:
            base_df = pd.merge_asof(base_df, df_dict[key],
                            on='millisSinceGpsEpoch',
                            by=['phoneName', 'collectionName'],
                            suffixes=('', key),
                            direction='nearest',
                            tolerance=1000) # 1sec

    base_df = base_df.drop('elapsedRealtimeNanos', axis=1)

    return base_df

# %%
# 辞書のdf用意
df_dict = {}
df_dict['derived'] = new_train_dr_df.sort_values('millisSinceGpsEpoch')
df_dict['Fix'] = None
df_dict['OrientationDeg'] = new_train_orient_df.sort_values('millisSinceGpsEpoch')
df_dict['Raw'] = new_train_raw_df.sort_values('millisSinceGpsEpoch')
df_dict['Status'] = new_train_status_df.sort_values('millisSinceGpsEpoch')
df_dict['UncalAccel'] = new_train_acc_df.sort_values('millisSinceGpsEpoch')
df_dict['UncalGyro'] = new_train_gyro_df.sort_values('millisSinceGpsEpoch')
df_dict['UncalMag'] = new_train_mag_df.sort_values('millisSinceGpsEpoch')
df_dict['GroundTruth'] = new_train_gt_df.sort_values('millisSinceGpsEpoch')
df_dict

# %%
# millisSinceGpsEpochの型をint64にそろえる
for key in df_dict.keys():
    if not df_dict[key] is None:
        df_dict[key]['millisSinceGpsEpoch'] = df_dict[key]['millisSinceGpsEpoch'].astype(np.int64)
        print(key)
        print(df_dict[key][df_dict[key]['millisSinceGpsEpoch'].diff()<0].empty)

# %%
merged_train_df = merge_df_dict(df_dict)
merged_train_df

# %%
merged_train_df.info()

# %%
# データ確認
for key in df_dict.keys():
    print(key)
    display(df_dict[key])

# %%[markdown]
# # 保存
# %%
name = str(dt.now().strftime('%Y-%m-%d_%H'))

merged_train_df.to_csv(f'../../data/processed/confirm/train/{name}_{phone_name}.csv', index=False)