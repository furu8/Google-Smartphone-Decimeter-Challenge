# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
class DataJoiner:
    def __init__(self, phone_name, data_dir='train'):
        self.phone_name = phone_name
        self.data_dir = data_dir

        self.names = ['derived', 'Fix', 
                'OrientationDeg', 'Raw', 'Status', 
                'UncalAccel', 'UncalGyro', 'UncalMag', 'gt']
        if data_dir == 'test':
            self.names.remove('gt')

        self.df_info = pd.DataFrame(index=self.names)

    def load_df_dict(self):
        self.df_dict = {}
        for name in self.names:
            if name == 'derived' or name == 'gt':
                self.df_dict[name] = pd.read_csv(f'../../data/interim/{self.data_dir}/merged_{self.phone_name}_{name}.csv')
            else:
                self.df_dict[name] = pd.read_csv(f'../../data/interim/{self.data_dir}/merged_{self.phone_name}_{name}_add_columns.csv')
    
        return self.df_dict

    def encode_label(self, df, col):
        lenc = LabelEncoder()
        df[col] = lenc.fit_transform(df[col])
        return df

    def encode_CodeType(self):
        self.df_dict['Raw']['CodeType'] = self.df_dict['Raw']['CodeType'].fillna(0)
        for i, category in enumerate(self.df_dict['Raw']['CodeType'].unique()[1:]):
            self.df_dict['Raw'].loc[self.df_dict['Raw']['CodeType']==category, 'CodeType'] = i
    
    def groupby_mean(self, df):
        df_mean = df.groupby('millisSinceGpsEpoch', as_index=False).max()
        return df_mean

    def merge_df(self, df1, df2):
        return pd.merge(df1, df2, on='millisSinceGpsEpoch')

    def merge_df_dict(self, merge_base_col='Raw'):
        df_dict = self.df_dict.copy()
        
        base_df = df_dict[merge_base_col]
        print(base_df.info())
        df_dict.pop(merge_base_col)
        base_df['elapsedRealtimeNanos'] = 0 # suffixesのために、適当な値を入れる
        for key in df_dict.keys():
            if not df_dict[key].empty:
                print(df_dict[key].info())
                base_df = pd.merge_asof(base_df, df_dict[key],
                                on='millisSinceGpsEpoch',
                                by=['phoneName', 'collectionName'],
                                suffixes=('', key),
                                direction='nearest',
                                tolerance=1000) # 1sec

        base_df = base_df.drop('elapsedRealtimeNanos', axis=1)

        return base_df

    def drop_all_nan_col(self, key):
        self.df_dict[key] = self.df_dict[key].dropna(axis=1, how='all')

    def interpolate_mean(self, key, col):
        self.df_dict[key][col] = self.df_dict[key][col].fillna(self.df_dict[key][col].mean())

    def sort_df_dict(self):
        order_key = self.df_info[self.df_info['isorder']==False].index
        for key in order_key:
            self.df_dict[key] = self.df_dict[key].sort_values('millisSinceGpsEpoch')
    
    def set_mills_type(self):
        for key in self.df_info[self.df_info['isempty']==False].index:
            self.df_dict[key]['millisSinceGpsEpoch'] = self.df_dict[key]['millisSinceGpsEpoch'].astype(np.int64)

    def set_df_dict(self, df, key):
        self.df_dict[key] = df.copy()

    def check_millis_order(self, df_dict):
        diff_millis_list = [df_dict[key][df_dict[key]['millisSinceGpsEpoch'].diff()<0].empty for key in df_dict.keys()]
        self.df_info['isorder'] = diff_millis_list # Trueだと時間順
        print(self.df_info) 

    def check_empty_df(self, df_dict):
        empty_list = [df_dict[key].empty for key in df_dict.keys()]
        self.df_info['isempty'] = empty_list
        print(self.df_info)
        

# %%[markdown]
# # Train

# %%
"""
'Mi8',
'Pixel4',
'Pixel4Modded',
'Pixel4XL',
'Pixel4XLModded',
'Pixel5',
'SamsungS20Ultra'
"""

phone_name = 'Pixel4'
# %%
# load
pixel4 = DataJoiner(phone_name, data_dir='train')
org_df_dict = pixel4.load_df_dict()
org_df_dict
# %%
# shapeだけ表示
for org_df in org_df_dict.values():
    display(org_df.shape)

# %%
# df empty
pixel4.check_empty_df(org_df_dict)

# %%
# millisが正しい順か
pixel4.check_millis_order(org_df_dict)

# %%[markdown]
# ## derived
# %%
pixel4.df_dict['derived']
# %%
pixel4.df_dict['derived'].info()
# %%
# Labelig signalType
labeled_df = pixel4.encode_label(org_df_dict['derived'], 'signalType')
labeled_df

# %%
# gropuby mean
df_mean = pixel4.groupby_mean(labeled_df)
df_mean

# %%
df_mean.info()

# %%
# merge (collectionName, phoneName)
# new
new_dr_df = pixel4.merge_df(df_mean, org_df_dict['derived'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_dr_df

# %%
# set
pixel4.df_dict['derived'] = new_dr_df.copy()
pixel4.df_dict['derived']

# %%[markdown]
# ## Fix
# %%
pixel4.df_dict['Fix'] 

# %%
pixel4.df_dict['Fix'].info()

# %%[markdown]
# ## OrientationDeg

# %%
pixel4.df_dict['OrientationDeg']

# %%
pixel4.df_dict['OrientationDeg'].info()
# %%
# new
new_orient_df = pixel4.df_dict['OrientationDeg'].copy()
new_orient_df

# %%[markdown]
# ## Raw
# %%
pixel4.df_dict['Raw']

# %%
pixel4.df_dict['Raw'].info()

# %%
# 全欠損カラムを削除
pixel4.drop_all_nan_col('Raw')
pixel4.df_dict['Raw']

# %%
# 一部欠損を補完
pixel4.interpolate_mean('Raw', 'AgcDb')
pixel4.df_dict['Raw']
# %%
# CodeType unique
print(pixel4.df_dict['Raw']['CodeType'].unique())

# %%
# Labeling CodeType
pixel4.encode_CodeType()

# %%
pixel4.df_dict['Raw'].info()
# %%
# gropuby mean
df_mean = pixel4.groupby_mean(pixel4.df_dict['Raw'])
df_mean

# %%
df_mean.info()
# %%
# merge (collectionName, phoneName)
# new
new_raw_df = pixel4.merge_df(df_mean, org_df_dict['Raw'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_raw_df

# %%
# set
pixel4.df_dict['Raw'] = new_raw_df.copy()
pixel4.df_dict['Raw']

# %%[markdown]
# ## Status

# %%
pixel4.df_dict['Status']
# %%
pixel4.df_dict['Status'].info()

# %%
# 全欠損カラムを削除
pixel4.drop_all_nan_col('Status')
pixel4.df_dict['Status']

# %%
# 一部欠損を補完
pixel4.interpolate_mean('Status', 'CarrierFrequencyHz')
pixel4.df_dict['Status']

# %%
df_mean = pixel4.groupby_mean(pixel4.df_dict['Status'])
df_mean

# %%
df_mean.info()
# %%
# merge (collectionName, phoneName)
# new
new_status_df = pixel4.merge_df(df_mean, org_df_dict['Status'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_status_df

# %%
# set
pixel4.df_dict['Status'] = new_status_df.copy()
pixel4.df_dict['Status']

# %%[markdown]
# ## UncalAccel

# %%
pixel4.df_dict['UncalAccel']

# %%
pixel4.df_dict['UncalAccel'].info()

# %%
# new
new_acc_df = pixel4.df_dict['UncalAccel'].copy()
new_acc_df

# %%[markdown]
# ## UncalGyro

# %%
pixel4.df_dict['UncalGyro']
# %%
pixel4.df_dict['UncalGyro'].info()

# %%
# new
new_gyro_df = pixel4.df_dict['UncalGyro'].copy()
new_gyro_df

# %%[markdown]
# ## UncalMag

# %%
pixel4.df_dict['UncalMag']
# %%
pixel4.df_dict['UncalMag'].info()

# %%
# new
new_mag_df = pixel4.df_dict['UncalMag'].copy()
new_mag_df

# %%[markdown]
# ## ground_truth

# %%
pixel4.df_dict['gt']

# %%
# new
# 必要なground truthだけ抽出
new_gt_df = pixel4.df_dict['gt'][['latDeg', 'lngDeg', 'millisSinceGpsEpoch', 'phoneName', 'collectionName']].copy()
new_gt_df

# %%
# set
pixel4.df_dict['gt'] = new_gt_df.copy()
pixel4.df_dict['gt']

# %%[markdown]
# ## merge
# %%
# 結合しないメンバー
for key in pixel4.df_info[pixel4.df_info['isempty']==True].index:
    print(key)

# %%
# 結合メンバー
for key in pixel4.df_info[pixel4.df_info['isempty']==False].index:
    print(key)
   
# %%
# millisSinceGpsEpochの型をint64にそろえる
pixel4.set_mills_type()

# %%
# millisSinceGpsEpochを時間順にする
pixel4.sort_df_dict()

# %%
# shapeだけ表示
for df in pixel4.df_dict.values():
    display(df.shape)
# %%
merged_train_df = pixel4.merge_df_dict()
merged_train_df

# %%
merged_train_df.info()

# %%[markdown]
# # Test
# %%
# load
pixel4 = DataJoiner(phone_name, data_dir='test')
org_df_dict = pixel4.load_df_dict()
org_df_dict
# %%
# shapeだけ表示
for org_df in org_df_dict.values():
    display(org_df.shape)

# %%
# df empty
pixel4.check_empty_df(org_df_dict)

# %%
# millisが正しい順か
pixel4.check_millis_order(org_df_dict)

# %%[markdown]
# ## derived
# %%
pixel4.df_dict['derived']
# %%
pixel4.df_dict['derived'].info()
# %%
# Labelig signalType
labeled_df = pixel4.encode_label(org_df_dict['derived'], 'signalType')
labeled_df

# %%
# gropuby mean
df_mean = pixel4.groupby_mean(labeled_df)
df_mean

# %%
df_mean.info()

# %%
# merge (collectionName, phoneName)
# new
new_dr_df = pixel4.merge_df(df_mean, org_df_dict['derived'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_dr_df

# %%
# set
pixel4.df_dict['derived'] = new_dr_df.copy()
pixel4.df_dict['derived']

# %%[markdown]
# ## Fix
# %%
pixel4.df_dict['Fix'] 

# %%
pixel4.df_dict['Fix'].info()

# %%[markdown]
# ## OrientationDeg

# %%
pixel4.df_dict['OrientationDeg']

# %%
pixel4.df_dict['OrientationDeg'].info()
# %%
# new
new_orient_df = pixel4.df_dict['OrientationDeg'].copy()
new_orient_df

# %%[markdown]
# ## Raw
# %%
pixel4.df_dict['Raw']

# %%
pixel4.df_dict['Raw'].info()

# %%
# 全欠損カラムを削除
pixel4.drop_all_nan_col('Raw')
pixel4.df_dict['Raw']

# %%
# 一部欠損を補完
pixel4.interpolate_mean('Raw', 'AgcDb')
pixel4.df_dict['Raw']
# %%
# CodeType unique
print(pixel4.df_dict['Raw']['CodeType'].unique())

# %%
# Labeling CodeType
pixel4.encode_CodeType()

# %%
pixel4.df_dict['Raw'].info()
# %%
# gropuby mean
df_mean = pixel4.groupby_mean(pixel4.df_dict['Raw'])
df_mean

# %%
df_mean.info()

# %%
# merge (collectionName, phoneName)
# new
new_raw_df = pixel4.merge_df(df_mean, org_df_dict['Raw'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_raw_df

# %%
# set
pixel4.df_dict['Raw'] = new_raw_df.copy()
pixel4.df_dict['Raw']

# %%[markdown]
# ## Status

# %%
pixel4.df_dict['Status']
# %%
pixel4.df_dict['Status'].info()

# %%
# 全欠損カラムを削除
pixel4.drop_all_nan_col('Status')
pixel4.df_dict['Status']

# %%
# 一部欠損を補完
pixel4.interpolate_mean('Status', 'CarrierFrequencyHz')
pixel4.df_dict['Status']

# %%
df_mean = pixel4.groupby_mean(pixel4.df_dict['Status'])
df_mean
# %%
df_mean.info()
# %%
# merge (collectionName, phoneName)
# new
new_status_df = pixel4.merge_df(df_mean, org_df_dict['Status'].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']])
new_status_df

# %%
new_status_df.info()

# %%
# set
pixel4.df_dict['Status'] = new_status_df.copy()
pixel4.df_dict['Status']

# %%[markdown]
# ## UncalAccel

# %%
pixel4.df_dict['UncalAccel']

# %%
pixel4.df_dict['UncalAccel'].info()

# %%
# new
new_acc_df = pixel4.df_dict['UncalAccel'].copy()
new_acc_df

# %%[markdown]
# ## UncalGyro

# %%
pixel4.df_dict['UncalGyro']
# %%
pixel4.df_dict['UncalGyro'].info()

# %%
# new
new_gyro_df = pixel4.df_dict['UncalGyro'].copy()
new_gyro_df

# %%[markdown]
# ## UncalMag

# %%
pixel4.df_dict['UncalMag']
# %%
pixel4.df_dict['UncalMag'].info()

# %%
# new
new_mag_df = pixel4.df_dict['UncalMag'].copy()
new_mag_df

# %%[markdown]
# ## merge
# %%
# 結合しないメンバー
for key in pixel4.df_info[pixel4.df_info['isempty']==True].index:
    print(key)

# %%
# 結合メンバー
for key in pixel4.df_info[pixel4.df_info['isempty']==False].index:
    print(key)

# %%
# millisSinceGpsEpochの型をint64にそろえる
pixel4.set_mills_type()

# %%
# millisSinceGpsEpochを時間順にする
pixel4.sort_df_dict()

# %%
# shapeだけ表示
for df in pixel4.df_dict.values():
    display(df.shape)
# %%
merged_test_df = pixel4.merge_df_dict()
merged_test_df

# %%
merged_test_df.info()

# %%[markdown]
# # 保存
# %%
merged_train_df.to_csv(f'../../data/interim/confirm/train/{phone_name}.csv', index=False)
merged_test_df.to_csv(f'../../data/interim/confirm/test/{phone_name}.csv', index=False)
# %%
