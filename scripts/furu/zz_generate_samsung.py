# %%
import numpy as np
import pandas as pd

from IPython.core.display import display
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns

#最大表示行数を設定
pd.set_option('display.max_rows', 500)

import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# %%[markdown]
# ## データセット作成

# %%
# 各種関数
def load_df(dr_path_list, gt_path_list):
    df = pd.DataFrame()
    dr_paths = np.sort(np.array(dr_path_list))
    gt_paths = np.sort(np.array(gt_path_list))
    
    for dr_path, gt_path in zip(dr_paths, gt_paths):
        dr_onedf = pd.read_csv(dr_path)
        gt_onedf = pd.read_csv(gt_path)
        onedf = merge_df(dr_onedf, gt_onedf)
        df = pd.concat([df, onedf], axis=0)
    
    return df.reset_index(drop=True)

def merge_df(df1, df2):
    return  pd.merge_asof(df1, df2, 
                    on='millisSinceGpsEpoch', 
                    by=['collectionName', 'phoneName'], 
                    direction='nearest',
                    tolerance=100000)

# GNSSLogのテキストをdataframeにしよう
def gnss_log_to_dataframes(path):
    """
    https://www.kaggle.com/sohier/loading-gnss-logs
    """
    print('Loading ' + path, flush=True)
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            gnss_map[dataline[0]] = dataline[1:]
        elif not is_header:
            datas[dataline[0]].append(dataline[1:])

    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            results[k][col] = pd.to_numeric(results[k][col])

    return results

# %%[markdown]
# ### SamsusgS20Ultra

# %%
# パス
gt_train_path = '../../data/raw/train/*/SamsungS20Ultra/ground_truth.csv'
dr_train_path = '../../data/raw/train/*/SamsungS20Ultra/SamsungS20Ultra_derived.csv'
gt_train_path_list = gb.glob(gt_train_path)
dr_train_path_list = gb.glob(dr_train_path)

print(gt_train_path_list)
print(dr_train_path_list)

# %%
# データ
train_df = load_df(dr_train_path_list, gt_train_path_list)
display(train_df.shape)
display(train_df.head())
display(train_df.tail())
# %%
# 欠損確認
display(train_df.info())

# %%
# パス
dr_test_path = '../../data/raw/test/*/SamsungS20Ultra/SamsungS20Ultra_derived.csv'
dr_test_path_list = gb.glob(dr_test_path)

print(dr_test_path_list)
# %%
# データ
test_df = pd.DataFrame()
for dr_path in np.sort(np.array(dr_test_path_list)):
    onedf = pd.read_csv(dr_path)
    test_df = pd.concat([test_df, onedf], axis=0)

test_df = test_df.reset_index(drop=True)

display(test_df.shape)
display(test_df.head())
display(test_df.tail())

# %%
# 欠損確認
display(test_df.info())

# %%
# 統計値確認
display(train_df.describe())
display(test_df.describe())
# %%
# collectionNameを可視化
nametr_df = train_df.copy()
namete_df = test_df.copy()
nametr_df['hue'] = 'train'
namete_df['hue'] = 'test'
name_df = pd.concat([nametr_df, namete_df])
plt.figure(figsize=(20,4))
sns.countplot(x='collectionName', data=name_df, hue=name_df['hue'])
plt.xticks(rotation=45)
plt.legend()
plt.show()  

# %%
# groupby
mean_train_df = train_df.groupby('millisSinceGpsEpoch', as_index=False).mean()
mean_test_df = test_df.groupby('millisSinceGpsEpoch', as_index=False).mean()

display(mean_train_df.shape)
display(mean_test_df.shape)
display(mean_train_df)
display(mean_train_df)

"""
平均、最大、最小など、好みで中間データをここで吐き出しても良い
"""

# %%
# 保存
train_df.to_csv('../../data/interim/train/all_SamsungS20Ultra_derived.csv', index=False)
test_df.to_csv('../../data/interim/test/all_SamsungS20Ultra_derived.csv', index=False)
