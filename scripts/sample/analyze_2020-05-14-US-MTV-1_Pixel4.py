
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
import statsmodels.api as sm

from models import Util

#最大表示行数を設定
pd.set_option('display.max_rows', 500)

import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# %%[markdown]
# ## データ確認
# %%
# Pixel4読込
df_gt_p4 = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/ground_truth.csv')
df_dr_p4 = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv')

display(df_gt_p4.shape)
display(df_dr_p4.shape)

display(df_gt_p4.head())
display(df_dr_p4.head())

# %%
# Pixel4XL読込
df_gt_p4XL = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4XLModded/ground_truth.csv')
df_dr_p4XL = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4XLModded/Pixel4XLModded_derived.csv')

display(df_gt_p4XL.shape)
display(df_dr_p4XL.shape)
display(df_gt_p4XL.head())
display(df_dr_p4XL.head())

"""メモ
Moddedは改造された的な意味だが、何を意味してある言葉なのか不明
"""

# %%
# Pixel4テスト読込
df_dr_p4_test = pd.read_csv('../../data/raw/test/2020-05-15-US-MTV-1/Pixel4/Pixel4_derived.csv')
# Pixel4XLテスト読込
df_dr_p4XL_test = pd.read_csv('../../data/raw/test/2020-05-15-US-MTV-1/Pixel4XL/Pixel4XL_derived.csv')

display(df_dr_p4_test.shape)
display(df_dr_p4_test.shape)
display(df_dr_p4_test.head())
display(df_dr_p4_test.head())

# %%
# 欠損値確認
display(df_gt_p4.info())
display(df_dr_p4.info())
display(df_gt_p4XL.info())
display(df_dr_p4XL.info())
display(df_dr_p4_test.info())
display(df_dr_p4XL_test.info())

# %%
# 基本統計値
display(df_gt_p4.describe())
display(df_dr_p4.describe())
display(df_dr_p4_test.describe())
display(df_gt_p4XL.describe())
display(df_dr_p4XL.describe())
display(df_dr_p4XL_test.describe())

# %%[markdown]
# ## 初期の可視化
# ### 緯度経度

# %%
# 緯度経度
plt.scatter(df_gt_p4['latDeg'], df_gt_p4['lngDeg'], label='p4')
plt.scatter(df_gt_p4XL['latDeg'], df_gt_p4XL['lngDeg'], s=1, label='p4XL')
plt.show()

"""メモ
ほぼ同じ位置につけているとみて良さそうか？
"""

# %%[markdown]
# ### ひたすら折れ線グラフ
# %%
# 可視化関連の関数適当に
def plot_col(df, columns, istime=False):
    for col in columns:
        print(col)
        plt.figure(figsize=(20,4))
        if istime:
            plt.plot(df['cum_mSGE'], df[col])
        else:
            plt.plot(df[col])
        plt.show()

def plot_col_dfs(df1, df2, columns, istime=False):
    for col in columns:
        print(col)
        plt.figure(figsize=(20,4))
        if istime:
            plt.plot(df1['cum_mSGE'], df1[col])
            plt.plot(df2['cum_mSGE'], df2[col])
        else:
            plt.plot(df1[col])
            plt.plot(df2[col], alpha=0.5)
        plt.show()

def plot_hist(df, col):
    plt.hist(df[col], bins=50)
    plt.show()

# %%
# pixel4のground_truth
plot_col(df_gt_p4, df_gt_p4.columns[5:])
"""
わりと直感的にわかる
"""

# %%
# pixel4のderived
plot_col(df_dr_p4, df_dr_p4.columns[6:])

"""メモ
単純な可視化はほぼ見えない
"""

# %%
# pixel4XLのground_truth
plot_col(df_gt_p4XL, df_gt_p4XL.columns[5:])

"""
pixel4とほぼ同じに見える
"""

# %%
# pixel4XLのderived
plot_col(df_dr_p4XL, df_dr_p4XL.columns[6:])

"""メモ
単純な可視化はほぼ見えない
"""

# %%
# pixel4のtest
plot_col(df_dr_p4_test, df_dr_p4_test.columns[5:])

"""メモ
単純な可視化はほぼ見えない
"""

# %%
# pixel4のtest
plot_col(df_dr_p4XL_test, df_dr_p4XL_test.columns[5:])

"""メモ
単純な可視化はほぼ見えない
"""

# %%
# Pixel4とPixel4XLのground_trueを比較
plot_col_dfs(df_gt_p4, df_gt_p4XL, df_gt_p4.columns[6:])

"""メモ
hDopなど、一致していない部分もある
"""

# %%
# Pixel4とPixel4XLのderivedを比較
plot_col_dfs(df_dr_p4, df_dr_p4XL, df_dr_p4.columns[5:])

"""メモ
相変わらず、死ぬほど見にくい。
データの長さが異なり、無印側が長い。
satClkBiasMとか、satClkDriftMpsなどの値が結構違う。
"""

# %%
# テストのPixel4とPixel4XLのderivedを比較
plot_col_dfs(df_dr_p4_test, df_dr_p4XL_test, df_dr_p4_test.columns[5:])

"""メモ
見にくさは同様。
データの長さは学習データより顕著で、XL側が長い（横軸をindexではなく、時間にすれば変わりそう）。
satClkBiasMとか、satClkDriftMpsなどの値が結構違う。
"""

# %%[markdown]
# ### 時間（millisSinceGpsEpoch）に着目

# %%
# Pixel4とPixel4XLのground_trueのヒストグラム
plot_hist(df_gt_p4, 'millisSinceGpsEpoch')
plot_hist(df_gt_p4XL, 'millisSinceGpsEpoch')

"""メモ
gtはmillisSinceGpsEpochがユニークなので、均一っぽい
"""

# %%
# Pixel4とPixel4XLのderivedのヒストグラム
plot_hist(df_dr_p4, 'millisSinceGpsEpoch')
plot_hist(df_dr_p4XL, 'millisSinceGpsEpoch')

"""メモ
よくよく見たら違う頻度かな、というくらい
"""
# %%
# テストのPixel4とPixel4XLのderivedのヒストグラム
plot_hist(df_dr_p4_test, 'millisSinceGpsEpoch')
plot_hist(df_dr_p4XL_test, 'millisSinceGpsEpoch')
"""メモ
Pixel4は10近辺多くなったりしてる。
Pixel4XLの方がデータ数多いため、縦軸の値が大きい。
"""

# %%
# データフレーム結合
df_p4 = pd.merge(df_dr_p4, df_gt_p4.iloc[:, 2:], on='millisSinceGpsEpoch')
df_p4XL = pd.merge(df_dr_p4XL, df_gt_p4XL.iloc[:, 2:], on='millisSinceGpsEpoch')

print(df_p4.columns)
print(df_p4XL.columns)
print(df_p4.shape)
print(df_p4XL.shape)
display(df_p4.head())
display(df_p4XL.head())

"""メモ
df_p4XLにmillisSinceGpsEpochが一致する時間が存在しない
"""
# %%
# 欠損値確認
display(df_p4.info())
display(df_p4XL.info())

# %%
# 基本統計値
display(df_p4.describe())

# %%[markdown]
# ## 特徴量を追加
# - 以降はdf_p4のみ

# %%
# millisSinceGpsEpochの差分
df_p4['dif_mSGE'] = df_p4['millisSinceGpsEpoch'].diff().fillna(0) / 1000000 # ミリ秒
df_p4[['cum_mSGE']]= df_p4['dif_mSGE'].cumsum()
df_p4[['cum_mSGE']].tail(100)

# %%
# 横軸index
plot_col(df_p4.iloc[:1000, :], df_p4.columns[7:-3])

# %%
# 横軸時間
plot_col(df_p4.iloc[:1000, :], df_p4.columns[7:-3], istime=True)

# %%[markdown]
# ## cum_mSGEの周期性を見てみる
# - 周期性の分析はまじめにやるとコードが膨大になりそうなので、別途書く

# %%
# cum_mSGEの可視化（再掲）
plt.figure(figsize=(20,4))
plt.plot(df_p4.loc[:300, 'cum_mSGE'])
plt.show()

# %%
# 自己相関のグラフ
fig = plt.figure(figsize=(20,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_p4['cum_mSGE'], lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_p4['cum_mSGE'], lags=50, ax=ax2)
plt.show()

# %%
# 周期性、トレンド、ホワイトノイズ
seasonal_decompose_res = sm.tsa.seasonal_decompose(df_p4.loc[:300, 'cum_mSGE'], freq=27)
seasonal_decompose_res.plot()
plt.show()
# %%
# cum_mSGEの周期性
mSGE = -1
for i, data in enumerate(df_p4['cum_mSGE'].values):
    if i == 300:
        break
    if data == mSGE:
        continue
    else:
        print(f'index: {i}, {data}') # 周期性
        mSGE = data

"""メモ
rollingで27や28を指定するより、cum_mSGEでgroupbyしたほうが良いか？
"""

# %%
# correctedPrM = rawPrM + satClkBiasM - isrbM - ionoDelayM - tropoDelayMの確認

df_p4['correctedPrM'] = df_p4['rawPrM'] + df_p4['satClkBiasM'] - df_p4['isrbM'] - df_p4['ionoDelayM'] - df_p4['tropoDelayM']
df_p4[['correctedPrM', 'latDeg', 'lngDeg']].head()

# %%
# ラベルエンコーディング
lenc = LabelEncoder()

lenc.fit(df_p4['signalType'])
df_p4['signalType_lenc'] = pd.DataFrame(lenc.transform(df_p4['signalType']))

display(df_p4[['signalType', 'signalType_lenc']].head(30))
display(df_p4.info())
display(df_p4.describe())

# %%[markdown]
# ## 特徴量追加後の可視化

# %%
# そのまま
plt.figure(figsize=(20,4))
plt.plot(df_p4.loc[:300, 'correctedPrM'])
plt.show()

# cum_mSGEごと
plt.figure(figsize=(20,4))
plt.plot(df_p4.loc[:300, 'cum_mSGE'], df_p4.loc[:300, 'correctedPrM'])
plt.show()

# %%
