# %%
from io import RawIOBase
import numpy as np
from numpy.matrixlib import defmatrix
import pandas as pd

from IPython.core.display import display
import glob as gb
from math import fmod
import matplotlib.pyplot as plt
import seaborn as sns

#最大表示行数を設定
pd.set_option('display.max_rows', 500)

import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# %%[markdown]
# # データセット作成

# %%
# 各種関数
def load_df(dr_path_list, gt_path_list):
    df = pd.DataFrame()
    dr_paths = np.sort(np.array(dr_path_list))
    gt_paths = np.sort(np.array(gt_path_list))
    
    for dr_path, gt_path in zip(dr_paths, gt_paths):
        dr_onedf = pd.read_csv(dr_path)
        gt_onedf = pd.read_csv(gt_path)
        onedf = pd.merge_asof(dr_onedf, gt_onedf,
                            on='millisSinceGpsEpoch', 
                            by=['collectionName', 'phoneName'], 
                            direction='nearest',
                            tolerance=100000)
        df = pd.concat([df, onedf], axis=0)
    
    return df.reset_index(drop=True)

def load_gnss_log(path_list):
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}

    paths = np.sort(np.array(path_list))
    for path in paths:
        dfs = gnss_log_to_dataframes(path, gnss_section_names)
        dfs = make_df(dfs)
        dfs = dfs[dfs['col_num']!=0]
        for df in dfs['dataframe'].values:
            pass

        # gyr_df, acc_df, mag_df, fix_df, sts_df, raw_df, otg_df = change_unix2gps(dfs)
        
        # df = pd.merge_asof(dfs['UncalGyro'], dfs['UncalAccel'])
        # df = pd.merge_asof(df, dfs['UncalMag'])
        # df = pd.merge_asof(df, dfs['Fix'])
        # df = pd.merge_asof(df, dfs['Status'])
        # df = pd.merge_asof(df, dfs['Raw'])
        # df = pd.merge_asof(df, dfs['OrientationDeg'])
        # display(df)
        # display(df.info())

def make_df(dfs):
    df = pd.DataFrame()
    df['col_num'] = check_emptydf(dfs)
    df['dataframe'] = change_df_unix2gps(dfs)
    return df

def check_emptydf(dfs):
    return [len(dfs[key].columns) for key in dfs.keys()]

def change_df_unix2gps(dfs):
    gyr_df = change_unix2gps(dfs['UncalGyro'], 'utcTimeMillis')
    acc_df = change_unix2gps(dfs['UncalAccel'], 'utcTimeMillis')
    mag_df = change_unix2gps(dfs['UncalMag'], 'utcTimeMillis')
    fix_df = change_unix2gps(dfs['Fix'], 'UnixTimeMillis')
    sts_df = change_unix2gps(dfs['Status'], 'UnixTimeMillis')
    raw_df = change_unix2gps(dfs['Raw'], 'utcTimeMillis')
    otg_df = change_unix2gps(dfs['OrientationDeg'], 'utcTimeMillis')

    return [gyr_df, acc_df, mag_df, fix_df, sts_df, raw_df, otg_df]

def change_unix2gps(df, timecol):
    df = df.rename(columns={timecol: 'millisSinceGpsEpoch'})
    
    if not df.empty:
        df['millisSinceGpsEpoch'] = df['millisSinceGpsEpoch'].apply(unix2gps)
    
    return df

# GNSSLogのテキストをdataframeにしよう
def gnss_log_to_dataframes(path, gnss_section_names):
    """
    https://www.kaggle.com/sohier/loading-gnss-logs
    """
    print('Loading ' + path, flush=True)
    
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

# utc->gps time
## あっとるんかわからんけどUTCとGPSを変換するらしいコードをパクるよ(PHP製)
def unix2gps(unix_time):
    """
    https://www.andrews.edu/~tzs/timeconv/timealgorithm.html
    """
    if fmod(unix_time, 1) != 0:
        unix_time -= 0.50
        isleap = 1
    else:
        isleap = 0
    gps_time = unix_time - 315964800000
    nleaps = countleaps(gps_time)
    gps_time = gps_time + nleaps + isleap
    return gps_time

def countleaps(gps_time):
    leaps = getleaps()
    lenleaps = len(leaps)
    nleaps = 0
    for i in range(lenleaps):
        if gps_time >= leaps[i] - i:
            nleaps += 1000
    return nleaps

def getleaps():
    leaps = [46828800000, 78364801000, 109900802000, 173059203000, 252028804000, 315187205000, 346723206000, 393984007000, 425520008000, 457056009000, 504489610000, 551750411000, 599184012000, 820108813000, 914803214000, 1025136015000, 1119744016000, 1167264017000]
    return leaps


# %%[markdown]
# ## Mi8
# ### _derived, ground_truth

# %%
# phone
phone_name = 'Mi8'
# %%
# train path
gt_train_path = f'../../data/raw/train/*/{phone_name}/ground_truth.csv'
dr_train_path = f'../../data/raw/train/*/{phone_name}/{phone_name}_derived.csv'
gt_train_path_list = gb.glob(gt_train_path)
dr_train_path_list = gb.glob(dr_train_path)

print(gt_train_path_list)
print(dr_train_path_list)

# %%
# train data
train_df = load_df(dr_train_path_list, gt_train_path_list)
display(train_df.shape)
display(train_df.head())
display(train_df.tail())
# %%
# 欠損確認
display(train_df.info())

# %%
# test path
dr_test_path = f'../../data/raw/test/*/{phone_name}/{phone_name}_derived.csv'
dr_test_path_list = gb.glob(dr_test_path)

print(dr_test_path_list)
# %%
# test data
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
train_df.to_csv(f'../../data/interim/train/all_{phone_name}_derived.csv', index=False)
test_df.to_csv(f'../../data/interim/test/all_{phone_name}_derived.csv', index=False)

# %%
# gnss log path
gnss_train_path = f'../../data/raw/train/*/{phone_name}/{phone_name}_GnssLog.txt'
gnss_test_path = f'../../data/raw/test/*/{phone_name}/{phone_name}_GnssLog.txt'

gnss_train_path_list = gb.glob(gnss_train_path)
gnss_test_path_list = gb.glob(gnss_train_path)

print(gnss_train_path_list)
print(gnss_test_path_list)

# %%[markdown]
# ### gnss log

# %%
# gnss log path
gnss_log_path = f'../../data/raw/train/*/{phone_name}/{phone_name}_GnssLog.txt'
gnss_log_path_list = gb.glob(gnss_log_path)

print(gnss_log_path_list)
# %%
# gnss log data
gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
df_dict = gnss_log_to_dataframes('../../data/raw/train/2020-07-17-US-MTV-1/Mi8/Mi8_GnssLog.txt', gnss_section_names)

display(df_dict.keys())
display(df_dict)
# %%
# UncalGyro
load_gnss_log(gnss_log_path_list)

# %%
# UncalAccel
df_dict['UncalAccel']
# %%
# Fix
df_dict['Fix']
# %%
# UncalMag
df_dict['UncalMag']

# %%
# Status
df_dict['Status']

# %%
# Raw
df_dict['Raw'].info()

# %%
# OrientationDeg
df_dict['OrientationDeg']

# %%
# 全欠損と結合
# カラムがあるFixは全欠損で結合される
null1_df = pd.merge_asof(df_dict['Status'], df_dict['Fix'], 
                        on='UnixTimeMillis',
                        direction='nearest',
                        tolerance=10000)

# カラムがないOritentationDegはエラー
# null2_df = pd.merge_asof(df_dict['UncalMag'], df_dict['OrientationDeg'], 
#                         on='utcTimeMillis',
#                         direction='nearest',
#                         tolerance=10000)

# %%
# Rawのmerger_asof
raw1_df = pd.merge_asof(df_dict['Raw'], df_dict['UncalMag'], 
                    on='utcTimeMillis',
                    direction='nearest',
                    tolerance=10000)
raw2_df = pd.merge_asof(df_dict['UncalMag'], df_dict['Raw'], 
                    on='utcTimeMillis',
                    direction='nearest',
                    tolerance=10000)
raw2_df

# %%
# Rawの時間系カラムのユニーク数
print(df_dict['Raw']['utcTimeMillis'].unique().shape)
print(df_dict['Raw']['TimeNanos'].unique().shape)
print(df_dict['Raw']['FullBiasNanos'].unique().shape)

# %%
# derivedの時間系カラムのユニーク数
train_df.loc[train_df['collectionName']=='2020-07-17-US-MTV-1', 'millisSinceGpsEpoch'].unique().shape

# %%
# 片方にしかない時間
(set(df_dict['Raw']['utcTimeMillis'].apply(unix2gps)) - set(train_df.loc[train_df['collectionName']=='2020-07-17-US-MTV-1', 'millisSinceGpsEpoch']))

# %%
gt = pd.read_csv('../../data/raw/train/2020-07-17-US-MTV-1/Mi8/ground_truth.csv')
dr = pd.read_csv('../../data/raw/train/2020-07-17-US-MTV-1/Mi8/Mi8_derived.csv')

set(gt['millisSinceGpsEpoch']) - set(dr['millisSinceGpsEpoch'])

# 1279059935000,1279060131000
# 1279059935000,1279060131000
