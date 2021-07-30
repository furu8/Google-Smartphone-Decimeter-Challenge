# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, display_pdf
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns
# %%
pd.set_option('display.max_rows', 500)
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
p4xl_gt_train_path = '../../data/raw/train/*/Pixel4XL/ground_truth.csv'
p4xl_dr_train_path = '../../data/raw/train/*/Pixel4XL/Pixel4XL_derived.csv'
p4xl_gt_train_path_list = gb.glob(p4xl_gt_train_path)
p4xl_dr_train_path_list = gb.glob(p4xl_dr_train_path)

print(p4xl_gt_train_path_list)
print(p4xl_dr_train_path_list)
# %%
# pixel4のground_truth読込(学習データ)
p4xl_gt_train_df = load_df(p4xl_gt_train_path_list)

display(p4xl_gt_train_df.shape)
display(p4xl_gt_train_df.head())
display(p4xl_gt_train_df.tail())
# %%
# pixel4のderived読込(学習データ)
p4xl_dr_train_df = load_df(p4xl_dr_train_path_list)

display(p4xl_dr_train_df.shape)
display(p4xl_dr_train_df.head())
display(p4xl_dr_train_df.tail())
# %%
p4xl_dr_test_path = '../../data/raw/test/*/Pixel4XL/Pixel4XL_derived.csv'
p4xl_dr_test_path_list = gb.glob(p4xl_dr_test_path)

print(p4xl_dr_test_path_list)
# %%
p4xl_dr_test_df = load_df(p4xl_dr_test_path_list)

display(p4xl_dr_test_df.shape)
display(p4xl_dr_test_df.head())
display(p4xl_dr_test_df.tail())
# %%
# 欠損確認
display(p4xl_dr_train_df.info())
display(p4xl_dr_test_df.info())
display(p4xl_gt_train_df.info())
# %%
# 統計値確認
display(p4xl_dr_train_df.describe())
display(p4xl_dr_test_df.describe())
# %%
# collectionNameごとにデータ数を確かめる
train_names = np.sort(p4xl_dr_train_df['collectionName'].unique())
test_names = np.sort(p4xl_dr_test_df['collectionName'].unique())

print(train_names)
print(test_names)

for tr_n, te_n in zip(train_names, test_names):
    """
    zipの関係で学習データをすべて可視化できていないが無視
    """
    tr_cnt = len(p4xl_dr_train_df[p4xl_dr_train_df['collectionName']==tr_n])
    te_cnt = len(p4xl_dr_test_df[p4xl_dr_test_df['collectionName']==te_n])
    print(tr_cnt, end=' ')
    print(te_cnt)
# %%
# collectionNameを可視化
nametr_df = p4xl_dr_train_df.copy()
namete_df = p4xl_dr_test_df.copy()
nametr_df['hue'] = 'train'
namete_df['hue'] = 'test'
name_df = pd.concat([nametr_df, namete_df])
plt.figure(figsize=(20,4))
sns.countplot(x='collectionName', data=name_df, hue=name_df['hue'])
plt.xticks(rotation=45)
plt.legend()
plt.show()
# %%
# groupbyおまけ
mean_train_df = p4xl_dr_train_df.groupby('millisSinceGpsEpoch').mean()
mean_test_df = p4xl_dr_test_df.groupby('millisSinceGpsEpoch').mean()

display(mean_train_df.shape)
display(mean_test_df.shape)
display(mean_train_df)
display(mean_train_df)
# %%
# とりあえず何も考えずにmergeしてみる
display(p4xl_dr_train_df.merge(p4xl_gt_train_df))
display(p4xl_dr_test_df.merge(p4xl_gt_train_df))
# %%
# merge_asofを試してみる
display(pd.merge_asof(p4xl_dr_train_df, p4xl_gt_train_df, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=5))
# %%
# 全部読み込む
all_train_df_path = "../../data/raw/train/*/*/*_derived.csv"
all_gt_df_path = "../../data/raw/train/*/*/ground_truth.csv"
all_train_df_list = gb.glob(all_train_df_path)
all_gt_df_list = gb.glob(all_gt_df_path)
all_train_df = load_df(all_train_df_list)
all_gt_df = load_df(all_gt_df_list)
# %%
all_test_df_path = "../../data/raw/test/*/*/*_derived.csv"
all_test_df_list = gb.glob(all_test_df_path)
all_test_df = load_df(all_test_df_list)
# %%
all_test_df.shape
# %%
display(all_test_df.head())
display(all_test_df.tail())
# %%
all_gt_df.shape
# %%
all_train_df.shape
# %%
display(all_train_df.head())
display(all_train_df.tail())
# %%
display(all_gt_df.head())
display(all_gt_df.tail())
display(all_gt_df.describe())
# %%
all_train_df.sort_values("millisSinceGpsEpoch")
# %%
all_gt_df.sort_values("millisSinceGpsEpoch")
# %%
display(pd.merge_asof(all_train_df.sort_values("millisSinceGpsEpoch"), all_gt_df.sort_values("millisSinceGpsEpoch"), on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest"))
# %%
plt.hist(all_train_df['millisSinceGpsEpoch'], bins=50)
plt.show()
# %%
plt.hist(pd.merge_asof(all_train_df.sort_values("millisSinceGpsEpoch"), all_gt_df.sort_values("millisSinceGpsEpoch"), on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest")["millisSinceGpsEpoch"], bins=50)
plt.show()
# %%
display(pd.merge_asof(all_train_df.sort_values("millisSinceGpsEpoch"), all_gt_df.sort_values("millisSinceGpsEpoch"), on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest").info())
# %%
display(all_train_df.sort_values("millisSinceGpsEpoch"))
# %%
p4_tr = pd.read_csv("../../data/raw/train/2020-05-21-US-MTV-1/Pixel4/Pixel4_derived.csv")
# %%
p4_tr.head(20)
# %%
p4_tr_sorted = p4_tr.sort_values("millisSinceGpsEpoch")
p4_tr_sorted.head(20)
# %%
display(p4_tr.info())
display(p4_tr_sorted.info())
# %%
!pip install natsort
# %%
from natsort import index_natsorted
all_train_df.sort_values(
    by="millisSinceGpsEpoch",
    key=lambda x: np.argsort(index_natsorted(all_train_df["millisSinceGpsEpoch"]))
).head(100)
# %%
p4_tr_sorted['millisSinceGpsEpoch'].plot()
# %%
p4_tr['constellationType'].plot()
# %%
p4_tr_sorted['receivedSvTimeInGpsNanos'].plot()
# %%
p4_tr['xSatPosM'].plot()
# %%
all_train_df["phoneName"].unique()
# %%
all_train_df.groupby("phoneName").describe()
# %%
