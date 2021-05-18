# %%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

# %%[markdown]
# ## データセット作成

# %%
# 読込
train_df = pd.read_csv('../data/raw/train.csv')
test_df = pd.read_csv('../data/raw/test.csv')

display(train_df.head())
display(test_df.head())

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
