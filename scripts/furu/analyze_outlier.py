# %%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdp
import simdkalman
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
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

    def check_outlier(self):
        # 外れ値
        outlier_col_list = []
        outlier_date_dict = {}

        for col in self.df.select_dtypes(include='number').columns:
            outlier_df = self.detect_outlier(col)
            print(col)
            # display(outlier_df)
            # dfが空じゃなかったら
            if not outlier_df.empty:
                outlier_date_df = pd.merge_asof(outlier_df, 
                                                self.df[['millisSinceGpsEpoch', 'collectionName']].sort_values('millisSinceGpsEpoch'), 
                                                on='millisSinceGpsEpoch')[['collectionName', col]]
                outlier_col_list.append(col)
                outlier_date_dict[col] = (outlier_date_df['collectionName'].unique())       

                display(outlier_date_df)

        return outlier_col_list, outlier_date_dict

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
    try:
        ort_df['millisSinceGpsEpoch'] = ort_df['millisSinceGpsEpoch'].astype(np.int64)
        ort_flag = True
    except:
        ort_flag = False

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
    if ort_flag:
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
# phone_name = 'SamsungS20Ultra'
phone_name = 'Mi8'

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
outlier_col_list, outlier_date_dict = outlier.check_outlier()

# %%
# 外れ値とみなしたカラムだけの基本統計量
outlier.df.describe()[outlier_col_list]

# %%
# 外れ値とみなしたカラムだけの日付
outlier_date_dict

# %%
# BiasZMps2補間
imu_df.loc[imu_df['BiasZMps2']>-0.002394, ['BiasZMps2']] = imu_df['BiasZMps2'].mean()
imu_df[imu_df['collectionName']=='2021-04-15-US-MTV-1']

# %%
outlier_date_dict.pop('BiasZMps2')
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
# # kalman shapeで怒られる
# def kalman():
#     T = 1.0
#     state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], [0, 1, 0, T, 0, 0.5 * T ** 2], [0, 0, 1, 0, T, 0],
#                                 [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
#     process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
#     observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#     observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9

#     print(state_transition.shape)
#     print(process_noise.shape)
#     print(observation_model.shape)
#     print(observation_noise.shape)

#     kf = simdkalman.KalmanFilter(
#             state_transition = state_transition,
#             process_noise = process_noise,
#             observation_model = observation_model,
#             observation_noise = observation_noise)
#     return kf

# def apply_kf_smoothing(df, kf_, keys):
#     unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
#     for collection, phone in tqdm(unique_paths):
#         cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
#         data = df[cond][keys].to_numpy()
#         data = data.reshape(1, len(data), len(keys))
#         smoothed = kf_.smooth(data)
#         for i, key in enumerate(keys):
#             df.loc[cond, key] = smoothed.states.mean[0, :, i]
        
#     return df

# # %%
# keys = list(outlier_date_dict.keys())
# print(len(keys))
# kf = kalman()
# apply_kf_smoothing(imu_df, kf, keys)

# %%
# adaptive gauss
def apply_gauss_smoothing(arg_df, params, keys):
    df = arg_df.copy()
    SZ_1 = params['sz_1']
    SZ_2 = params['sz_2']
    SZ_CRIT = params['sz_crit']    
    
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][keys].copy()
        display(data)
        for key in keys:
            gaussianed1 = gaussian_filter1d(data[key], np.sqrt(SZ_1))
            gaussianed2 = gaussian_filter1d(data[key], np.sqrt(SZ_2))

            dif = data.values[1:,0] - data.values[:-1,0] # 先頭の次1つから最後 - 先頭から最後の前1つ
    
            crit = np.append(np.abs(gaussian_filter1d(dif, np.sqrt(SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(dif), np.sqrt(SZ_CRIT)))),[0])

            df.loc[cond, key] = gaussianed1 * crit + gaussianed2 * (1.0 - crit)  
                       
    return df

# %%
keys = list(outlier_date_dict.keys())
aga_params = {'sz_1' : 0.85, 'sz_2' : 5.65, 'sz_crit' : 1.5}
smoothed_imu_df = apply_gauss_smoothing(imu_df, aga_params, keys)
smoothed_imu_df

# %%
imu_df == smoothed_imu_df
# %%
display(imu_df)
display(smoothed_imu_df)

# %%
plt.figure(figsize=(16,4))
plt.plot(imu_df['UncalAccelXMps2'], label='imu')
plt.plot(smoothed_imu_df['UncalAccelXMps2'], label='smooth')
plt.legend()
plt.show()

# %%
%%time
outlier_smoothed = Outlier(smoothed_imu_df)

# 外れ値
outlier_smoothed_col_list, outlier_smoothed_date_dict = outlier_smoothed.check_outlier()
outlier_smoothed_date_dict

# %%
%%time
# 外れ値だけでヒストグラム可視化
for col, date_list in outlier_smoothed_date_dict.items():
    for date in date_list:
        print(col, date)
        outlier_smoothed.plot_onedate_hist(col, date)

# %%
%%time
# 外れ値だけでヒストグラム可視化（全体）
print(outlier_smoothed_col_list)
for col in outlier_smoothed_col_list:
    print(col)
    outlier_smoothed.plot_hist(col)

# %%
print(outlier_col_list)
fig, ax = plt.subplots(len(outlier_col_list), 2, figsize=(16,32))
for i, col in enumerate(outlier_col_list):
    print(col)
    ax[i][0].hist(imu_df[col])
    ax[i][1].hist(smoothed_imu_df[col])

plt.tight_layout()
plt.show()

# %%
for col in smoothed_imu_df.columns:
    imu_df[col] = smoothed_imu_df[col].values
imu_df

# %%
# 全体
collection_names =  [
    '2021-04-22-US-SJC-1',
    '2021-04-29-US-SJC-2',
    '2021-04-28-US-SJC-1',
    '2021-04-29-US-SJC-2',
    '2021-04-22-US-SJC-1',
    '2021-04-28-US-SJC-1',
    '2021-04-29-US-MTV-1',
    '2021-04-29-US-MTV-1',
    '2021-04-29-US-MTV-1',
    '2021-01-05-US-SVL-1',
    '2021-01-05-US-SVL-1',
    '2021-04-15-US-MTV-1',
    '2020-05-14-US-MTV-2',
    '2020-09-04-US-SF-1]',
    '2021-04-28-US-MTV-1',
    '2021-03-10-US-SVL-1',
    '2021-01-05-US-SVL-2',
    '2021-01-04-US-RWC-2',
    '2021-03-10-US-SVL-1',
    '2021-01-04-US-RWC-1',
    '2021-01-04-US-RWC-2'
]

phone_names = [
    'SamsungS20Ultra',
    'Pixel4',
    'Pixel4Modded',
    'Pixel4XL',
    'Pixel4XLModded',
    'Pixel5',
    'Mi8'
]

bl_trn_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')

# collectionNameとphoneNameの総組み合わせ
cn2pn_df = bl_trn_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df

# %%
%%time
# 全体
for pn in phone_names:
    print(pn)
    imu_df = prepare_imu_data('train', 'cname', pn)
    # 外れ値
    outlier = Outlier(imu_df)
    outlier_col_list, outlier_date_dict = outlier.check_outlier()
    print(outlier_date_dict)

# %%
# 全体
for cn in collection_names:
    pns = cn2pn_df.loc[cn2pn_df['collectionName'] == cn, 'phoneName'].values
    for pn in pns:
        print('Prepare Training Dataset：', cn + '_' + pn)
        imu_df = prepare_imu_data('train', cn, pn)

        # 外れ値
        outlier = Outlier(imu_df)
        outlier_col_list, outlier_date_dict = outlier.check_outlier()

        print('_'*20)

# %%
# profile = pdp.ProfileReport(train_dr_df)
# profile.to_file(outputfile=f'train_dr_df_{phone_name}.html')