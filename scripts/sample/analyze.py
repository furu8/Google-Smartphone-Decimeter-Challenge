
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

# %%
# 読込
df_pd = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv')
df_gt = pd.read_csv('../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/ground_truth.csv')

display(df_pd.head())
display(df_gt.head())

# %%[markdown]
# ## 初期の可視化

# %%
# # 関数
# def plot_bar(df1, df2, col, xlabels):
#     left = df1[col].value_counts(sort=False) / len(df1)
#     right = df2[col].value_counts(sort=False) / len(df2)

#     # print(left.index)
#     # print(right.index)
#     # print(xlabels)
#     # labels = np.union1d(left.index.values, right.index.values)

#     fig, ax = plt.subplots(1, 2, figsize=(6, 4))
#     ax[0].bar(left.index, left, color='red')     # 左
#     ax[1].bar(right.index, right, color='green') # 右

#     ax[0].set_xticks(left.index)
#     ax[0].set_xticklabels(xlabels)
#     ax[1].set_xticks(left.index)
#     ax[1].set_xticklabels(xlabels)
#     ax[0].set_ylim(0,1)
#     ax[1].set_ylim(0,1)
#     plt.show()

# # %%
# # 死亡者と生存者の違い
# df_s0 = df[df['Survived']==0] # 死亡者
# df_s1 = df[df['Survived']==1] # 生存者
# plot_bar(df_s0, df_s1, 'Pclass', ['1', '2', '3'])
# plot_bar(df_s0, df_s1, 'Sex', ['female', 'male'])
# plot_bar(df_s0, df_s1, 'SibSp', ['0', '1', '2', '3', '4', '5', '8'])
# plot_bar(df_s0, df_s1, 'Parch', ['0', '1', '2', '3', '4', '5', '6'])
# plot_bar(df_s0, df_s1, 'Embarked', ['C', 'Q', 'S'])

# # %%
# # trainとtestの違い
# df_s0 = df[df['Survived']!=-1] # train
# df_s1 = df[df['Survived']==-1] # test
# plot_bar(df_s0, df_s1, 'Pclass', ['1', '2', '3'])
# plot_bar(df_s0, df_s1, 'Sex', ['female', 'male'])
# plot_bar(df_s0, df_s1, 'SibSp', ['0', '1', '2', '3', '4', '5', '8'])
# plot_bar(df_s0, df_s1, 'Parch', ['0', '1', '2', '3', '4', '5', '6'])
# plot_bar(df_s0, df_s1, 'Embarked', ['C', 'Q', 'S'])

# %%
# 死亡者と生存者の違い
df_s = df[df['data']=='train']
cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df_s, hue=df_s['Survived'])
    plt.legend( loc='upper right')
    # plt.show()
    plt.savefig(f'../figures/diffSurvived_{col}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)

# %%
# trainとtestの違い
cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df, hue=df['data'])
    plt.legend( loc='upper right')
    # plt.show()
    plt.savefig(f'../figures/diffTrainTest_{col}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)
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