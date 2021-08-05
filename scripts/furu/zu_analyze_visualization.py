# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap as up

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
%%time
basepath = '../../data/processed'
train_df = pd.read_csv(f'{basepath}/train/train_merged_base.csv')
test_df = pd.read_csv(f'{basepath}/test/test_merged_base.csv')
display(train_df)
display(test_df)
# %%
train_df[['latDeg', 'lngDeg']]
# %%
display(train_df[['latDeg', 'lngDeg']].info())
display(test_df[['latDeg', 'lngDeg']].info())

# %%
# 全緯度経度みたい
def plot_lat_lng(df):
    print(len(df))
    plt.figure(figsize=(8,6))
    plt.scatter(df['lngDeg'], df['latDeg'], s=0.5)
    plt.show()

# %%
plot_lat_lng(train_df)
plot_lat_lng(test_df)

# %%
target_df = train_df.drop(['millisSinceGpsEpoch', 'collectionName', 'phoneName', 'latDeg', 'lngDeg'], axis=1).copy()
target_df

# %%
target_df = target_df.dropna(axis=1)
target_df
# %%
scaler = StandardScaler()
train_df_std = pd.DataFrame(scaler.fit_transform(target_df), columns=target_df.columns) 
train_df_std
# %%
pca = PCA()
pca.fit(train_df_std)
pca_df = pd.DataFrame(pca.transform(train_df_std), columns=["PC{}".format(x + 1) for x in range(len(train_df_std.columns))])
pca_df = pd.concat([pca_df, train_df[['collectionName']]], axis=1)
pca_df['collectionName'] = LabelEncoder().fit_transform(train_df['collectionName'])
pca_df[['collectionName']]
# %%
plt.scatter(pca_df['PC1'], pca_df['PC3'], alpha=0.5, c=list(pca_df.loc[:, 'collectionName']))
plt.show()

# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(pca_df.drop('collectionName', axis=1).values)
kmeans.labels_

# %%
kmeansed_df =  pd.concat([pca_df, pd.DataFrame(kmeans.labels_, columns=['label'])], axis=1)
plt.scatter(kmeansed_df['PC1'], kmeansed_df['PC2'], alpha=0.5, c=list(kmeansed_df.loc[:, 'label']))
plt.show()

# %%
train_df['label'] = kmeans.labels_
plt.scatter(train_df['lngDeg'], train_df['latDeg'], alpha=0.5, c=list(train_df.loc[:, 'label']))
plt.show()
# %%
baseline_train_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')
baseline_test_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
display(baseline_train_df)
display(baseline_test_df)
# %%
plot_lat_lng(baseline_train_df)
plot_lat_lng(baseline_test_df)
# %%
cols = df.columns
for col in cols:
    print(col)

# %%
df_list = [df[[col]] for col in cols if not df.loc[df[col]!=0, col].empty]

# %%
concated_df = pd.DataFrame()
for onedf in df_list:
    concated_df = pd.concat([concated_df, onedf], axis=1)

# %%
concated_df
# %%
%%time
# 可視化しきれん
for i, col in enumerate(concated_df.columns[1:]):
    print(i, col)
    plt.figure(figsize=(20,4))
    plt.plot(concated_df[col])
    plt.show()

# %%
columns = [
    'xSatPosM', 'ySatPosM', 'zSatPosM',
    'xSatVelMps', 'ySatVelMps', 'zSatVelMps', 'rawPrM',
    'rollDeg', 'pitchDeg', 'yawDeg'
]

for i, col in enumerate(columns):
    print(i, col)
    plt.figure(figsize=(20,4))
    plt.plot(concated_df[col])
    plt.show()
