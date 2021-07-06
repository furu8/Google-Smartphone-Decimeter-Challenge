# %%
from numpy.core.numeric import outer
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdp

from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)


# %%
class Outlier:
    def __init__(self, phone_name) -> None:
        self.phone_name = phone_name
        self.df = None
        self.plot_df = None

    def load_df(self, base_path, dirname, _filename):
        self.df = pd.read_csv(f'{base_path}/{dirname}/merged_{self.phone_name}_{_filename}.csv')
        self.plot_df = pd.read_csv(f'{base_path}/{dirname}/merged_{self.phone_name}_{_filename}.csv')

    def detect_outlier(self, col):
        df = self.df.select_dtypes(include='number').copy() # int, floatだけ抽出
        df[f'{col}_mean'] = df[col].mean()
        df[f'{col}_std'] = df[col].std()
        th = df[f'{col}_mean'] + df[f'{col}_std'] * 6
        return df[df[col]>th]

    def plot_hist(self, col):
        plt.figure(figsize=(4,3))
        self.plot_df[col].hist()
        plt.show()

    def plot_onedate_hist(self, col, date):
        plot_df = self.plot_df[self.plot_df['collectionName']==date]
        plt.figure(figsize=(4,3))
        plot_df[col].hist()
        plt.show()

    def plot_onedate_line(self, col, date):
        plot_df = self.plot_df[self.plot_df['collectionName']==date]
        plt.figure(figsize=(20,4))
        plot_df[col].plot()
        plt.show()

# %%
%%time
phone_name = input('スマホの名前指定: ')
dir_name = input('ディレクトリの名前指定（train/test）: ')
file_name = input('ファイルの名前指定（derived/gt等）: ')

basepath = '../../data/interim'
outlier = Outlier(phone_name)
outlier.load_df(basepath, dir_name, file_name)
outlier.df
# %%
outlier.df.describe()

# %%
outlier.df.info()

# %%
# 外れ値
outlier_col_list = []
outlier_date_dict = {}

for col in outlier.df.select_dtypes(include='number').columns:
    outlier_df = outlier.detect_outlier(col)

    outlier_date_df = pd.merge_asof(outlier_df, outlier.df[['millisSinceGpsEpoch', 'collectionName']], 
            on='millisSinceGpsEpoch')[['collectionName', col]]

    # dfが空じゃなかったら
    if not outlier_df.empty:
        outlier_col_list.append(col)
        outlier_date_dict[col] = (outlier_date_df['collectionName'].unique())

    print(col)
    display(outlier_date_df)

# %%
# 外れ値とみなしたカラムだけの基本統計量
outlier.df.describe()[outlier_col_list]

# %%
# 外れ値とみなしたカラムだけの日付
outlier_date_dict

# %%
%%time
# 外れ値だけでヒストグラム可視化
for col, date_list in outlier_date_dict.items():
    for date in date_list:
        print(col, date)
        outlier.plot_onedate_hist(col, date)

# %%
%%time
# 外れ値だけでヒストグラム可視化（全体）
print(outlier_col_list)
for col in outlier_col_list:
    print(col)
    outlier.plot_hist(col)

# %%
%%time
# 外れ値だけで折れ線可視化
print(outlier_date_dict)
for col, date_list in outlier_date_dict.items():
    for date in date_list:
        print(col, date)
        outlier.plot_onedate_line(col, date=date)
# %%
# profile = pdp.ProfileReport(train_dr_df)
# profile.to_file(outputfile=f'train_dr_df_{phone_name}.html')