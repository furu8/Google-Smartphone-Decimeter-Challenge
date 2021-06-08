# %%
import numpy as np
import pandas as pd
from pandas.core.reshape.merge import merge_asof
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns

#最大表示行数を設定
pd.set_option('display.max_rows', 500)

import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# # %%
# # baseline_locations_test

# base = pd.read_csv('../../data/raw/baseline_locations_test.csv')[['latDeg', 'lngDeg']]
# samp = pd.read_csv('../../data/submission/sample_submission.csv')[['phone', 'millisSinceGpsEpoch']]

# base_samp = pd.concat([samp, base], axis=1)

# # display(base_samp)
# # base_samp.to_csv('../../data/submission/sample_submission_baseline_locations_test.csv', index=False)

# # trainデータの件数
# base_train = pd.read_csv('../../data/raw/baseline_locations_train.csv')
# display(base_train.shape)
# display(base_train)

# %%[markdown]
# ## データセット作成

# %%
# 各種関数
def load_df(path_list):
    df = pd.DataFrame()
    paths = np.sort(np.array(path_list))
    for path in paths:
        onedf = pd.read_csv(path)
        print(len(onedf))
        df = pd.concat([df, onedf], axis=0)
    
    return df.reset_index(drop=True)
# %%
p4_gt_train_path = '../../data/raw/train/*/Pixel4/ground_truth.csv'
p4_dr_train_path = '../../data/raw/train/*/Pixel4/Pixel4_derived.csv'
p4_gt_train_path_list = gb.glob(p4_gt_train_path)
p4_dr_train_path_list = gb.glob(p4_dr_train_path)

print(p4_gt_train_path_list)
print(p4_dr_train_path_list)
# %%
# pixel4のground_truth読込(学習データ)
p4_gt_train_df = load_df(p4_gt_train_path_list)

display(p4_gt_train_df.shape)
display(p4_gt_train_df.head())
display(p4_gt_train_df.tail())

# %%
# pixel4のderived読込(学習データ)
p4_dr_train_df = load_df(p4_dr_train_path_list)

display(p4_dr_train_df.shape)
display(p4_dr_train_df.head())
display(p4_dr_train_df.tail())

# %%
p4_dr_test_path = '../../data/raw/test/*/Pixel4/Pixel4_derived.csv'
p4_dr_test_path_list = gb.glob(p4_dr_test_path)

print(p4_dr_test_path_list)
# %%
# pixel4のderived読込(テストデータ)
p4_dr_test_df = load_df(p4_dr_test_path_list)

display(p4_dr_test_df.shape)
display(p4_dr_test_df.head())
display(p4_dr_test_df.tail())

# %%
# 欠損確認
display(p4_gt_train_df.info())
display(p4_dr_train_df.info())
display(p4_dr_test_df.info())

# %%
# 統計値確認
display(p4_gt_train_df.describe())
display(p4_dr_train_df.describe())
display(p4_dr_test_df.describe())

# %%
# collectionNameごとにデータ数を確かめる
train_names = np.sort(p4_dr_train_df['collectionName'].unique())
test_names = np.sort(p4_dr_test_df['collectionName'].unique())

print(train_names)
print(test_names)

for tr_n, te_n in zip(train_names, test_names):
    """
    zipの関係で学習データをすべて可視化できていないが無視
    """
    tr_cnt = len(p4_dr_train_df[p4_dr_train_df['collectionName']==tr_n])
    te_cnt = len(p4_dr_test_df[p4_dr_test_df['collectionName']==te_n])
    print(tr_cnt, end=' ')
    print(te_cnt)

# %%
# collectionNameを可視化
nametr_df = p4_dr_train_df.copy()
namete_df = p4_dr_test_df.copy()
nametr_df['hue'] = 'train'
namete_df['hue'] = 'test'
name_df = pd.concat([nametr_df, namete_df])
plt.figure(figsize=(20,4))
sns.countplot(x='collectionName', data=name_df, hue=name_df['hue'])
plt.xticks(rotation=45)
plt.legend()
plt.show()  

# %%
# merge字の欠損値確認
checknull = pd.merge(p4_dr_train_df, p4_gt_train_df.iloc[:, 2:], 
                        on='millisSinceGpsEpoch', 
                        how='left')

onechecknull = checknull[checknull['collectionName']=='2020-05-21-US-MTV-1']
display(onechecknull)

# ground_truthがない時間（2020-05-21-US-MTV-1の中で）
gt_list = onechecknull.loc[onechecknull['latDeg'].isnull(), 'millisSinceGpsEpoch'].unique()
gt_list
# onechecknull.to_csv('../../data/interim/train/checknull_merge_2020-05-21-US-MTV-1.csv', index=False)
# %%
p4_train_df = pd.merge_asof(p4_dr_train_df.sort_values('millisSinceGpsEpoch'), 
                            p4_gt_train_df.iloc[:, 2:].sort_values('millisSinceGpsEpoch'), 
                            on='millisSinceGpsEpoch',
                            direction='nearest',
                            # tolerance=30
                            )
print(p4_dr_train_df.shape)
print(p4_train_df.shape)

onep4_train_df = p4_train_df[p4_train_df['collectionName']=='2020-05-21-US-MTV-1']
display(onep4_train_df)

display(p4_train_df.loc[(p4_train_df['collectionName']=='2020-05-21-US-MTV-1') & p4_train_df['latDeg'].isnull(), 'millisSinceGpsEpoch'].unique())
display(p4_train_df.loc[(p4_train_df['millisSinceGpsEpoch'].isin(gt_list))])
# onep4_train_df.to_csv('../../data/interim/train/checknull_mergeasof_2020-05-21-US-MTV-1.csv', index=False)
# %%
print(len(p4_dr_train_df['millisSinceGpsEpoch'].unique()))
print(len(p4_train_df['millisSinceGpsEpoch'].unique()))
p4_train_df.info()

# %%
# groupbyおまけ
mean_train_df = p4_dr_train_df.groupby('millisSinceGpsEpoch').mean()
mean_test_df = p4_dr_test_df.groupby('millisSinceGpsEpoch').mean()

display(mean_train_df.shape)
display(mean_test_df.shape)
display(mean_train_df)
display(mean_train_df)

"""
平均、最大、最小など、好みで中間データをここで吐き出しても良い
"""

# %%
print(mean_train_df.shape)
print(p4_gt_train_df.shape)

merge = pd.merge(mean_train_df, p4_gt_train_df.sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', how='left')
mergeasof = pd.merge_asof(mean_train_df, p4_gt_train_df.sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')

print(merge.shape)
print(mergeasof.shape)
# plt.scatter(p4_gt_train_df['latDeg'], p4_gt_train_df['lngDeg'])
# plt.scatter(mergeasof['latDeg'], mergeasof['lngDeg'])

# merge.to_csv('../../data/interim/train/checknull_merge_groupby.csv', index=False)
# mergeasof.to_csv('../../data/interim/train/checknull_mergeasof_groupby.csv', index=False)

# %%
# 保存
# p4_dr_train_df.to_csv('../../data/interim/train/all_Pixel4_derived.csv', index=False)
# p4_dr_test_df.to_csv('../../data/interim/test/all_Pixel4_derived.csv', index=False)
