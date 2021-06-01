# %%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display
import glob as gb

# %%[markdown]
# ## データセット作成

# %%
# 各種関数
def load_df(path_list):
    df = pd.DataFrame()
    for path in path_list:
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
# pixel4のground_truth読込
p4_gt_train_df = load_df(p4_gt_train_path_list)

display(p4_gt_train_df.shape)
display(p4_gt_train_df.head())

# %%
# pixel4のderived読込
p4_dr_train_df = load_df(p4_dr_train_path_list)

display(p4_dr_train_df.shape)
display(p4_dr_train_df.head())

# %%
# 欠損確認
display(train_df.info())
display(test_df.info())

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
