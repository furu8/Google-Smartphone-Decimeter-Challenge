
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
    plt.scatter(df_pd[col])
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
# %%[markdown]
# ## 特徴量を追加

# %%
# 家族人数
df['Family'] = df['SibSp'] + df['Parch']

# %%
# ラベルエンコーディング
lenc = LabelEncoder()

lenc.fit(df['Sex'])
df['Sex'] = pd.DataFrame(lenc.transform(df['Sex']))

lenc.fit(df['Embarked'])
df['Embarked'] = pd.DataFrame(lenc.transform(df['Embarked']))

df['Cabin'] = df['Cabin'].apply(lambda x:str(x)[0])
lenc.fit(df['Cabin'])
df['Cabin'] = pd.DataFrame(lenc.transform(df['Cabin']))

display(df)
display(df.info())
display(df.describe())

# %%[markdown]
# ## 特徴量追加後の可視化

# %%
# 死亡者と生存者の違い
df_s = df[df['data']=='train']
cols = ['Family', 'Cabin']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df_s, hue=df_s['Survived'])
    plt.legend( loc='upper right')
    # plt.show()
    plt.savefig(f'../figures/diffSurvived_{col}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)

# %%
# trainとtestの違い
cols = ['Family', 'Cabin']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df, hue=df['data'])
    plt.legend( loc='upper right')
    # plt.show()
    plt.savefig(f'../figures/diffTrainTest_{col}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)

# %%[markdown]
# ## 保存
# %%
# データ保存
df.to_csv('../data/processed/all.csv')

# %%
# 特徴量
df = df.drop(['PassengerId', 'data', 'Survived'], axis=1)
df
# %%
# 特徴量保存
Util.dump(df.columns, '../config/features/all.pkl')
# %%