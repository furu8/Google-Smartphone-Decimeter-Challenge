# %%
import numpy as np
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
def load_df(path_list):
    df = pd.DataFrame()
    for path in np.sort(np.array(path_list)):
        onedf = pd.read_csv(path)
        df = pd.concat([df, onedf], axis=0)
    
    return df.reset_index(drop=True)

def load_gnss_log(path_list):
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    df = pd.DataFrame()

    paths = np.sort(np.array(path_list))
    for path in paths:
        df_dict = gnss_log_to_dataframes(path, gnss_section_names) # gnss_log.txtをdf化
        df_dict = change_dfs_unix2gps(df_dict) # 時間変換(utc->gps)
        df_dict = extract_gnss_log(df_dict) # 空のdf除去
        raw_df = df_dict['Raw']
        raw_df['elapsedRealtimeNanos'] = np.nan
        raw_df['collectionName'] = path.split('/')[5]
        raw_df['phoneName'] = path.split('/')[6]
        df_dict.pop('Raw')
        for key in df_dict.keys():
            raw_df = pd.merge_asof(raw_df, df_dict[key],
                                    on='millisSinceGpsEpoch',
                                    suffixes=('', key),
                                    direction='nearest',
                                    tolerance=1000) # 1sec
        
        df = pd.concat([df.reset_index(drop=True), 
                        raw_df.reset_index(drop=True)], axis=0)
        df = df.drop('elapsedRealtimeNanos', axis=1)

    return df.reset_index(drop=True)


def make_df(dfs):
    df = pd.DataFrame()
    df['col_num'] = check_emptydf(dfs)
    df['dataframe'] = change_df_unix2gps(dfs)
    return df

def check_emptydf(dfs):
    return [len(dfs[key].columns) for key in dfs.keys()]

def change_dfs_unix2gps(dfs):
    dfs['UncalGyro'] = change_df_unix2gps(dfs['UncalGyro'], 'utcTimeMillis')
    dfs['UncalAccel'] = change_df_unix2gps(dfs['UncalAccel'], 'utcTimeMillis')
    dfs['UncalMag'] = change_df_unix2gps(dfs['UncalMag'], 'utcTimeMillis')
    dfs['Fix'] = change_df_unix2gps(dfs['Fix'], 'UnixTimeMillis')
    dfs['Status'] = change_df_unix2gps(dfs['Status'], 'UnixTimeMillis')
    dfs['Raw'] = change_df_unix2gps(dfs['Raw'], 'utcTimeMillis')
    dfs['OrientationDeg'] = change_df_unix2gps(dfs['OrientationDeg'], 'utcTimeMillis')

    return dfs

def extract_gnss_log(gnss_log_df_dict):
    return {key: gnss_log_df for key, gnss_log_df in gnss_log_df_dict.items() if not gnss_log_df.empty}


def change_df_unix2gps(df, timecol):
    if not df.empty:
        df = df.rename(columns={timecol: 'millisSinceGpsEpoch'})
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
# ## Pixel4XL

# %%
# phone
phone_name = 'Pixel4XL'

# %%[markdown]
# ### gnss log

# %%
# train gnss log path
gnss_train_path = f'../../data/raw/train/*/{phone_name}/{phone_name}_GnssLog.txt'
gnss_test_path = f'../../data/raw/test/*/{phone_name}/{phone_name}_GnssLog.txt'

gnss_train_path_list = gb.glob(gnss_train_path)
gnss_test_path_list = gb.glob(gnss_test_path)

print(gnss_train_path_list)
print(gnss_test_path_list)

# %%
# train gnss log data
train_gnss_df = load_gnss_log(gnss_train_path_list)
for col in train_gnss_df.columns:
    print(col)

# %%
# 出力
display(train_gnss_df.shape)
display(train_gnss_df.head())
display(train_gnss_df.tail())

# %%
# 欠損
display(train_gnss_df.info())

# %%
# test gnss log data
test_gnss_df = load_gnss_log(gnss_test_path_list)
for col in test_gnss_df.columns:
    print(col)

# %%
display(test_gnss_df.shape)
display(test_gnss_df.head())
display(test_gnss_df.tail())

# %%
# 欠損
display(test_gnss_df.info())

# %%
# 統計値
display(train_gnss_df.describe())
display(test_gnss_df.describe())

#########################################################################################
# %%[markdown]
# ### デバッグ確認用
# %%
# train gnss log 確認用
gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
path = '../../data/raw/train/2020-05-21-US-MTV-2/Pixel4XL/Pixel4XL_GnssLog.txt'
df_dict = gnss_log_to_dataframes(path, gnss_section_names)

# %%
df_dict.keys()

# # %%
# # train gnss log 確認用
# gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
# path = '../../data/raw/train/2021-04-26-US-SVL-1/Mi8/Mi8_GnssLog.txt'
# df_dict = gnss_log_to_dataframes(path, gnss_section_names)

# f_dict = change_dfs_unix2gps(df_dict) # 時間変換(utc->gps)
# df_dict = extract_gnss_log(df_dict) # 空のdf除去
# raw_df = df_dict['Raw']
# df_dict.pop('Raw')
# for key in df_dict.keys():
#     raw_df = pd.merge_asof(raw_df, df_dict[key],
#                             on='millisSinceGpsEpoch',
#                             direction='nearest',
#                             tolerance=1000) # 1sec
# for col in raw_df.columns:
#     print(col)

"""
Pixel4XLは時間の順番がおかしい箇所がいっぱい
Statusを実行するとわかる
"""
# %%
# UncalAccel
# 447587レコード目で時間が３つ前と逆転してる
df_dict['UncalAccel'][df_dict['UncalAccel']['utcTimeMillis'].diff()<0]
df_dict['UncalAccel'].iloc[447583:447590]

# %%
# UncalMag
df_dict['UncalMag'][df_dict['UncalMag']['utcTimeMillis'].diff()<0]

# %%
# UncalGyro
df_dict['UncalGyro'][df_dict['UncalGyro']['utcTimeMillis'].diff()<0]

# %%
# Fix
df_dict['Fix'][df_dict['Fix']['UnixTimeMillis'].diff()<0]

# %%
# Status
# いっぱい
df_dict['Status'][df_dict['Status']['UnixTimeMillis'].diff()<0]

# %%
# Raw
df_dict['Raw'][df_dict['Raw']['utcTimeMillis'].diff()<0]

# %%
# OrientationDeg
# そもそもカラムなし
df_dict['OrientationDeg'][df_dict['OrientationDeg']['utcTimeMillis'].diff()<0] 

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
# Rawのmerge_asof
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

#################################################################################

# %%[markdown]
# ### _derived, ground_truth
# %%
# train path
gt_train_path = f'../../data/raw/train/*/{phone_name}/ground_truth.csv'
dr_train_path = f'../../data/raw/train/*/{phone_name}/{phone_name}_derived.csv'
gt_train_path_list = gb.glob(gt_train_path)
dr_train_path_list = gb.glob(dr_train_path)

print(dr_train_path_list)
print(gt_train_path_list)

# %%
# train data
dr_df = load_df(dr_train_path_list)
gt_df = load_df(gt_train_path_list)
display(dr_df.shape)
display(dr_df.head())
display(dr_df.tail())
display(gt_df.shape)
display(gt_df.head())
display(gt_df.tail())
# %%
# 欠損確認
display(dr_df.info())
display(gt_df.info())

# %%
# test path
dr_test_path = f'../../data/raw/test/*/{phone_name}/{phone_name}_derived.csv'
dr_test_path_list = gb.glob(dr_test_path)

print(dr_test_path_list)
# %%
# test data
test_df = load_df(dr_test_path_list)

display(test_df.shape)
display(test_df.head())
display(test_df.tail())

# %%
# 欠損確認
display(test_df.info())

# %%[markdown]
# ### merge_asof

# %%
# gnss_log 確認
train_gnss_df[train_gnss_df['millisSinceGpsEpoch'].diff()<0]
# %%
# train結合
train_df = pd.merge_asof(train_gnss_df, dr_df,
                        on='millisSinceGpsEpoch',
                        direction='nearest',
                        tolerance=100000)
train_df = pd.merge_asof(train_df, gt_df,
                        on='millisSinceGpsEpoch',
                        by=['collectionName', 'phoneName'],
                        direction='nearest',
                        tolerance=100000)
train_df
# %%
# test結合
test_df = pd.merge_asof(test_gnss_df, test_df,
                        on='millisSinceGpsEpoch',
                        direction='nearest',
                        tolerance=100000)

test_df

# %%
# 統計値確認
display(train_df.describe())
display(test_df.describe())

# %%
train_gnss_df[train_gnss_df['millisSinceGpsEpoch'].diff()<0][['FullBiasNanos', 'collectionName']]
train_gnss_df.iloc[303896:303905][['FullBiasNanos', 'collectionName']]

# %%
test_gnss_df[test_gnss_df['millisSinceGpsEpoch'].diff()<0][['FullBiasNanos', 'collectionName']]
# test_gnss_df.iloc[303896:303905][['FullBiasNanos', 'collectionName']]
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