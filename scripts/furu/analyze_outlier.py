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
    def __init__(self, df) -> None:
        self.df = df

    def detect_outlier(self, col):
        df = self.df.select_dtypes(include='number').copy() # int, floatだけ抽出
        df[f'{col}_mean'] = df[col].mean()
        df[f'{col}_std'] = df[col].std()
        th = df[f'{col}_mean'] + df[f'{col}_std'] * 3
        return df[df[col]>th]

    def plot_hist(self, col):
        plt.figure(figsize=(4,3))
        self.df[col].hist()
        plt.show()

    def plot_onedate_hist(self, col, date):
        plot_df = self.df[self.df['collectionName']==date]
        plt.figure(figsize=(4,3))
        plot_df[col].hist()
        plt.show()

    def plot_onedate_line(self, col, date):
        plot_df = self.df[self.df['collectionName']==date]
        plt.figure(figsize=(20,4))
        plot_df[col].plot()
        plt.show()

# %%
def prepare_imu_data(dataset_name, cname, pname):
    '''Prepare IMU Dataset (For Train: IMU+GT+BL; For Test: IMU+BL)
    Input：
        1. data_dir: data_dir
        2. dataset_name: dataset name（'train'/'test'）
        3. cname: CollectionName
        4. pname: phoneName
        5. bl_df: baseline's dataframe
    Output：df_all
    '''
    # load GNSS log
    acc_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalAccel_add_columns.csv')
    gyr_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalGyro_add_columns.csv')
    mag_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalMag_add_columns.csv')
    ort_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_OrientationDeg_add_columns.csv')

    # acc_df = acc_df[acc_df['collectionName']==cname].reset_index(drop=True)
    # gyr_df = gyr_df[gyr_df['collectionName']==cname].reset_index(drop=True)
    # mag_df = mag_df[mag_df['collectionName']==cname].reset_index(drop=True)
    # ort_df = ort_df[ort_df['collectionName']==cname].reset_index(drop=True)

    acc_df['millisSinceGpsEpoch'] = acc_df['millisSinceGpsEpoch'].astype(np.int64)
    gyr_df['millisSinceGpsEpoch'] = gyr_df['millisSinceGpsEpoch'].astype(np.int64)
    mag_df['millisSinceGpsEpoch'] = mag_df['millisSinceGpsEpoch'].astype(np.int64)
    ort_df['millisSinceGpsEpoch'] = ort_df['millisSinceGpsEpoch'].astype(np.int64)

    # display(acc_df)
    # display(gyr_df)
    # display(mag_df)
    # display(ort_df)

    # merge sub-datasets
    # accel + gyro
    imu_df = pd.merge_asof(acc_df.sort_values('millisSinceGpsEpoch'),
                           gyr_df.drop('elapsedRealtimeNanos', axis=1).sort_values('millisSinceGpsEpoch'),
                           on = 'millisSinceGpsEpoch',
                           by=['collectionName', 'phoneName'],
                           direction='nearest')
    # (accel + gyro) + mag
    imu_df = pd.merge_asof(imu_df.sort_values('millisSinceGpsEpoch'),
                           mag_df.drop('elapsedRealtimeNanos', axis=1).sort_values('millisSinceGpsEpoch'),
                           on = 'millisSinceGpsEpoch',
                           by=['collectionName', 'phoneName'],
                           direction='nearest')
    # ((accel + gyro) + mag) + OrientationDeg
    imu_df = pd.merge_asof(imu_df.sort_values('millisSinceGpsEpoch'),
                           ort_df.drop('elapsedRealtimeNanos', axis=1).sort_values('millisSinceGpsEpoch'),
                           on = 'millisSinceGpsEpoch',
                           by=['collectionName', 'phoneName'],
                           direction='nearest')

    return imu_df
# %%
%%time
dir_name = 'train'
collection_name = '2021-04-22-US-SJC-1'
phone_name = 'SamsungS20Ultra'

imu_df = prepare_imu_data(dir_name, collection_name, phone_name)
imu_df

# %%
outlier = Outlier(imu_df)

# %%
outlier.df.describe()

# %%
outlier.df.info()

# %%
%%time
# 外れ値
outlier_col_list = []
outlier_date_dict = {}

for col in outlier.df.select_dtypes(include='number').columns:
    outlier_df = outlier.detect_outlier(col)
    print(col)
    display(outlier_df)
    # dfが空じゃなかったら
    if not outlier_df.empty:
        outlier_date_df = pd.merge_asof(outlier_df, outlier.df[['millisSinceGpsEpoch', 'collectionName']].sort_values('millisSinceGpsEpoch'), 
            on='millisSinceGpsEpoch')[['collectionName', col]]
        outlier_col_list.append(col)
        outlier_date_dict[col] = (outlier_date_df['collectionName'].unique())       

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
# 全体
for col in imu_df.columns:
    print(col)
    plt.hist(imu_df[col])
    plt.show()
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
# kalman

# %%
# profile = pdp.ProfileReport(train_dr_df)
# profile.to_file(outputfile=f'train_dr_df_{phone_name}.html')