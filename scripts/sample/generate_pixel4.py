# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns

# 最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 200)
# 最大表示行数の指定（ここでは50行を指定）
pd.set_option('display.max_rows', 100)

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
display(p4_dr_train_df.info())
display(p4_dr_test_df.info())

"""メモ
Pixel4に関しては学習データ、テストデータともに欠損なし。
"""

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
# 統計値確認
display(train_df.describe())
display(test_df.describe())

# %%
# 結合
df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)
df['Survived'] = df['Survived'].fillna(-1)

display(df)
display(df.info())
display(df.describe())

# %%
# 欠損補完
df['Age'] = df['Age'].fillna(df['Age'].mean()) # 29.881137667304014
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().idxmax()) # S
df['Fare'] = df['Fare'].fillna(df['Fare'].mean()) # 33.295479281345564
 
# カラム削除
df = df.drop(['Name', 'Ticket'], axis=1)

# trainとtest
df.loc[df['Survived']!=-1, 'data'] = 'train'
df.loc[df['Survived']==-1, 'data'] = 'test'

display(df)
display(df.info())
display(df.describe())

# %%
# 保存
df.to_csv('../data/interim/all.csv', index=False)
# %%
