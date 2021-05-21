
# %%
import os
import pandas as pd
import numpy as np
from IPython.core.display import display
from sklearn.decomposition import PCA
import umap as up
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from models import Util

#最大表示行数を設定
pd.set_option('display.max_rows', 500)

import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# # %%
# # baseline_locations_test

# base = pd.read_csv('../../data/raw/baseline_locations_test.csv')[['latDeg', 'lngDeg']]
# samp = pd.read_csv('../../data/submission/sample_submission.csv')[['phone', 'millisSinceGpsEpoch']]

# base_samp = pd.concat([samp, base], axis=1)

# base_samp.to_csv('../../data/submission/sample_submission_baseline_locations_test.csv', index=False)

# %%
# 読込
df_gt = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/ground_truth.csv')
df_pd = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv')

display(df_gt.shape)
display(df_pd.shape)

display(df_gt.head())
display(df_pd.head())

# %%
# 欠損値確認
display(df_gt.info())
display(df_pd.info())

# %%
# 基本統計値
display(df_gt.describe())
display(df_pd.describe())

# %%[markdown]
# ## 初期の可視化

# %%
# 緯度経度
plt.scatter(df_gt['latDeg'], df_gt['lngDeg'])
plt.show()

for col in df_gt.columns[5:]:
    print(col)
    plt.plot(df_gt[col])
    plt.show()

# %%
for col in df_pd.columns[6:]:
    print(col)
    plt.figure(figsize=(20,4))
    plt.plot(df_pd.loc[:1000, col])
    plt.show()

# %%
# millisSinceGpsEpochのヒストグラム
plt.hist(df_gt['millisSinceGpsEpoch'], bins=50)
plt.show()
plt.hist(df_pd['millisSinceGpsEpoch'], bins=50)
plt.show()

# %%
# データフレーム結合
df = pd.merge(df_pd, df_gt.iloc[:, 2:], on='millisSinceGpsEpoch')

print(df.columns)
print(df.shape)
df.head()

# %%
# 読込
df_pd_test = pd.read_csv('../../data/raw/test/2020-05-15-US-MTV-1/Pixel4/Pixel4_derived.csv')

display(df_pd_test.shape)
display(df_pd_test.head())

# %%
# 欠損値確認
display(df_pd_test.info())

# %%
# 基本統計値
display(df_pd_test.describe())

# %%[markdown]
# ## 初期の可視化

# %%
for col in df_pd_test.columns[6:]:
    print(col)
    plt.figure(figsize=(20,4))
    plt.plot(df_pd.loc[:1000, col])
    plt.show()

# %%
# millisSinceGpsEpochのヒストグラム
plt.hist(df_pd_test['millisSinceGpsEpoch'], bins=50)
plt.show()

# %%[markdown]
# ## 特徴量を追加

# %%
# millisSinceGpsEpochの差分
df['dif_mSGE'] = df['millisSinceGpsEpoch'].diff().fillna(0) / 1000000 # ミリ秒
df[['cum_mSGE']]= df['dif_mSGE'].cumsum()
df[['cum_mSGE']].tail(100)

# %%
for col in df_pd_test.columns[6:]:
    print(col)
    plt.figure(figsize=(20,4))
    plt.plot(df.loc[:1000, 'cum_mSGE'], df.loc[:1000, 'xSatPosM'])
    plt.show()

# %%
# cum_mSGEの周期性
plt.figure(figsize=(20,4))
plt.plot(df.loc[:300, 'cum_mSGE'])
plt.show()

# %%
import statsmodels.api as sm
#  自己相関のグラフ
fig = plt.figure(figsize=(20,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.loc[:300, 'cum_mSGE'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.loc[:300, 'cum_mSGE'], lags=40, ax=ax2)
plt.show()

seasonal_decompose_res = sm.tsa.seasonal_decompose(df.loc[:300, 'cum_mSGE'], freq=12)
seasonal_decompose_res.plot()
plt.show()
# %%
# cum_mSGEの周期性
mSGE = -1
for i, data in enumerate(df['cum_mSGE'].values):
    if i == 300:
        break
    if data == mSGE:
        continue
    else:
        print('index: ', i) # 周期性
        mSGE = data

"""メモ
rollingで27や28を指定するより、cum_mSGEでgroupbyしたほうが良いか？
"""

# %%
# correctedPrM = rawPrM + satClkBiasM - isrbM - ionoDelayM - tropoDelayMの確認

df['correctedPrM'] = df['rawPrM'] + df['satClkBiasM'] - df['isrbM'] - df['ionoDelayM'] - df['tropoDelayM']
df[['correctedPrM', 'latDeg', 'lngDeg']].head()

# %%
# ラベルエンコーディング
lenc = LabelEncoder()

lenc.fit(df['signalType'])
df['signalType_lenc'] = pd.DataFrame(lenc.transform(df['signalType']))

display(df[['signalType', 'signalType_lenc']].head(30))
display(df.info())
display(df.describe())

# %%[markdown]
# ## 特徴量追加後の可視化

# %%
plt.figure(figsize=(20,4))
plt.plot(df.loc[:300, 'correctedPrM'])
plt.show()

plt.figure(figsize=(20,4))
plt.plot(df.loc[:300, 'cum_mSGE'], df.loc[:300, 'correctedPrM'])
plt.show()
