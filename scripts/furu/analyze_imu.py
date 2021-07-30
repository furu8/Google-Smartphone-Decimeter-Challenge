# %%
from IPython.core.display import display
import numpy as np
from cv2 import Rodrigues
import pandas as pd
from pathlib import Path
from pandas.core.algorithms import mode, value_counts
import pyproj
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import re
import glob as gb
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore", category=Warning)

# 最大表示行数を設定
pd.set_option('display.max_rows', 1000)
# 最大表示列数の指定
pd.set_option('display.max_columns', 1000)


# %%
# prepare_imu_data関連

# pitch:y
# yaw:z
# roll:x
def an2v(y_delta, z_delta, x_delta):
    '''
    Euler Angles ->Rotation Matrix -> Rotation Vector

    Input：
        1. y_delta          (float): the angle with rotateing around y-axis.
        2. z_delta         (float): the angle with rotateing around z-axis. 
        3. x_delta         (float): the angle with rotateing around x-axis. 
    Output：
        rx/ry/rz             (float): the rotation vector with rotateing 
    
    Code Ref.: https://www.zacobria.com/universal-robots-knowledge-base-tech-support-forum-hints-tips/python-code-example-of-converting-rpyeuler-angles-to-rotation-vectorangle-axis-for-universal-robots/
    (Note：In Code Ref: pitch=y,yaw=z,roll=x. But Google is pitch=x,yaw=z,roll=y)
    '''
    # yaw: z
    Rz_Matrix = np.matrix([
    [np.cos(z_delta), -np.sin(z_delta), 0],
    [np.sin(z_delta), np.cos(z_delta), 0],
    [0, 0, 1]
    ])

    # pitch: y
    Ry_Matrix = np.matrix([
    [np.cos(y_delta), 0, np.sin(y_delta)],
    [0, 1, 0],
    [-np.sin(y_delta), 0, np.cos(y_delta)]
    ])
    
    # roll: x
    Rx_Matrix = np.matrix([
    [1, 0, 0],
    [0, np.cos(x_delta), -np.sin(x_delta)],
    [0, np.sin(x_delta), np.cos(x_delta)]
    ])

    R = Rz_Matrix * Ry_Matrix * Rx_Matrix

    theta = np.arccos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * np.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    return rx, ry, rz

def prepare_imu_data(dataset_name, cname, pname, bl_df):
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

    acc_df = acc_df[acc_df['collectionName']==cname].reset_index(drop=True)
    gyr_df = gyr_df[gyr_df['collectionName']==cname].reset_index(drop=True)
    mag_df = mag_df[mag_df['collectionName']==cname].reset_index(drop=True)
    ort_df = ort_df[ort_df['collectionName']==cname].reset_index(drop=True)

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
    # display(imu_df)

    if dataset_name == 'train':
        # read GT dataset
        gt_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_gt.csv', 
                usecols = ['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg'])

        gt_df = gt_df[gt_df['collectionName']==cname].reset_index(drop=True)

        # merge GT dataset
        imu_df = pd.merge_asof(gt_df.sort_values('millisSinceGpsEpoch'),
                               imu_df.drop(['elapsedRealtimeNanos'], axis=1).sort_values('millisSinceGpsEpoch'),
                               on = 'millisSinceGpsEpoch',
                               by=['collectionName', 'phoneName'],
                               direction='nearest')
    elif dataset_name == 'test':
        # merge smaple_df
        imu_df = pd.merge_asof(sample_df.sort_values('millisSinceGpsEpoch'),
                           imu_df.drop(['elapsedRealtimeNanos'], axis=1).sort_values('millisSinceGpsEpoch'),
                           on = 'millisSinceGpsEpoch',
                        #    by=['collectionName', 'phoneName'],
                           direction='nearest')
    # display(imu_df)

    # OrientationDeg -> Rotation Vector
    rxs = []
    rys = []
    rzs = []
    for i in range(len(imu_df)):
        y_delta = imu_df['rollDeg'].iloc[i]
        z_delta = imu_df['yawDeg'].iloc[i]
        x_delta = imu_df['pitchDeg'].iloc[i]
        rx, ry, rz = an2v(y_delta, z_delta, x_delta)
        rxs.append(rx)
        rys.append(ry)
        rzs.append(rz)

    imu_df['ahrsX'] = rxs
    imu_df['ahrsY'] = rys
    imu_df['ahrsZ'] = rzs

    # display(imu_df)

    # calibrate sensors' reading
    for axis in ['X', 'Y', 'Z']:
        imu_df['Accel{}Mps2'.format(axis)] = imu_df['UncalAccel{}Mps2'.format(axis)] - imu_df['Bias{}Mps2'.format(axis)]
        imu_df['Gyro{}RadPerSec'.format(axis)] = imu_df['UncalGyro{}RadPerSec'.format(axis)] - imu_df['Drift{}RadPerSec'.format(axis)]
        imu_df['Mag{}MicroT'.format(axis)] = imu_df['UncalMag{}MicroT'.format(axis)] - imu_df['Bias{}MicroT'.format(axis)]

        # clearn bias features
        imu_df.drop(['Bias{}Mps2'.format(axis), 'Drift{}RadPerSec'.format(axis), 'Bias{}MicroT'.format(axis)], axis = 1, inplace = True) 

    # display(imu_df)

    if dataset_name == 'train':
        # merge Baseline dataset：imu_df + bl_df = (GT + IMU) + Baseline
        df_all = pd.merge(imu_df.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'}),
                      bl_df.drop(['phone'], axis=1).rename(columns={'latDeg':'latDeg_bl','lngDeg':'lngDeg_bl'}),
                      on = ['collectionName', 'phoneName', 'millisSinceGpsEpoch'])
    elif dataset_name == 'test':
        df_all = pd.merge(imu_df,
              bl_df[(bl_df['collectionName']==cname) & (bl_df['phoneName']==pname)].drop(['phone'], axis=1).rename(columns={'latDeg':'latDeg_bl','lngDeg':'lngDeg_bl'}),
              on = ['millisSinceGpsEpoch'])
        df_all.drop(['phone'], axis=1, inplace=True)
        
    return df_all

# %%
# get_xyz関連

def WGS84_to_ECEF(lat, lon, alt):
    # convert to radians
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    # f is the flattening factor
    finv = 298.257223563
    f = 1 / finv   
    # e is the eccentricity
    e2 = 1 - (1 - f) * (1 - f)    
    # N is the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt)        * np.sin(rad_lat)
    return x, y, z

transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},)

def ECEF_to_WGS84(x,y,z):
    lon, lat, alt = transformer.transform(x,y,z,radians=False)
    return lon, lat, alt

def get_xyz(df_all, dataset_name):
    # baseline: lat/lngDeg -> x/y/z
    df_all['Xbl'], df_all['Ybl'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_bl'], x['lngDeg_bl'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xaga_mean_predict_phone_mean'], df_all['Yaga_mean_predict_phone_mean'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_aga_mean_predict_phone_mean'], x['lngDeg_aga_mean_predict_phone_mean'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xaga_phone_mean_mean_predict'], df_all['Yaga_phone_mean_mean_predict'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_aga_phone_mean_mean_predict'], x['lngDeg_aga_phone_mean_mean_predict'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xaga_phone_mean'], df_all['Yaga_phone_mean'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_aga_phone_mean'], x['lngDeg_aga_phone_mean'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xaga_mean_predict'], df_all['Yaga_mean_predict'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_aga_mean_predict'], x['lngDeg_aga_mean_predict'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xkalman_mean_predict_phone_mean'], df_all['Ykalman_mean_predict_phone_mean'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_kalman_mean_predict_phone_mean'], x['lngDeg_kalman_mean_predict_phone_mean'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xkalman_mean_predict'], df_all['Ykalman_mean_predict'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_kalman_mean_predict'], x['lngDeg_kalman_mean_predict'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xkalman_phone_mean_mean_predict'], df_all['Ykalman_phone_mean_mean_predict'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_kalman_phone_mean_mean_predict'], x['lngDeg_kalman_phone_mean_mean_predict'], x['heightAboveWgs84EllipsoidM']), axis=1))
    df_all['Xkalman_phone_mean'], df_all['Ykalman_phone_mean'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_kalman_phone_mean'], x['lngDeg_kalman_phone_mean'], x['heightAboveWgs84EllipsoidM']), axis=1))
    
    if dataset_name == 'train':
        # gt: lat/lngDeg -> x/y/z
        df_all['Xgt'], df_all['Ygt'], df_all['Zgt'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x['latDeg_gt'], x['lngDeg_gt'], x['heightAboveWgs84EllipsoidM']), axis=1))
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_gt','lngDeg_gt', 'latDeg_bl', 'lngDeg_bl', 
            'latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean',
            'latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict',
            'latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict',
            'latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean',
            'latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean',
            'latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict',
            'latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict',
            'latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean',
        ]]
        df_all.drop(['latDeg_gt','lngDeg_gt', 'latDeg_bl', 'lngDeg_bl',
            'latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean',
            'latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict',
            'latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict',
            'latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean',
            'latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean',
            'latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict',
            'latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict',
            'latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean',
        ], axis = 1, inplace = True)
    elif dataset_name == 'test':
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_bl', 'lngDeg_bl',
            'latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean',
            'latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict',
            'latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict',
            'latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean',
            'latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean',
            'latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict',
            'latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict',
            'latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean',
        ]]
        df_all.drop(['latDeg_bl', 'lngDeg_bl', 'latDeg','lngDeg',
            'latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean',
            'latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict',
            'latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict',
            'latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean',
            'latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean',
            'latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict',
            'latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict',
            'latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean',
        ], axis = 1, inplace = True)     
        
    return lat_lng_df, df_all

# %%
# prepare_df_train関連

def prepare_df_train(df_all_train, window_size):
    '''prepare training dataset with all aixses'''
    tgt_df = df_all_train.copy()
    total_len = len(tgt_df) 
    moving_times = total_len - window_size + 1

    tgt_df.rename(columns = {'yawDeg':'yawZDeg', 'rollDeg':'rollYDeg', 'pitchDeg':'pitchXDeg'}, inplace = True)
    
    # 'Xgt', 'Ygt', 'Zgt'を除外
    feature_cols = np.array([f for f in tgt_df.columns if f not in ['Xgt', 'Ygt', 'Zgt']])
    
    # Historical Feature names
    hist_feats = []
    for time_flag in range(1, window_size + 1):
        for fn in feature_cols:
            hist_feats.append(fn + '_' + str(time_flag))
    # print('pitchXDeg_30' in hist_feats)

    # Window Sliding
    # t1 t2 t3 t4 t5 -> t6
    # t2 t3 t4 t5 t6 -> t7

    # Add historical data 
    df_train = pd.DataFrame()

    features = [tgt_df[feature_cols].iloc[start_idx : start_idx+window_size,:].values.flatten() for start_idx in range(moving_times)]
        
    # gtは別に処理
    x = tgt_df['Xgt'][29:total_len].values
    y = tgt_df['Ygt'][29:total_len].values
    z = tgt_df['Zgt'][29:total_len].values

    print(np.array(features).shape, np.array(x).shape)
    
    df_train = pd.DataFrame(features, columns = hist_feats)
    df_train['Xgt'] = x
    df_train['Ygt'] = y
    df_train['Zgt'] = z

    # display(df_train)

    # clean single-value feature: collectionName_[1-5]\phoneName_[1-5]
    tmp_feats = []
    for fn in df_train.columns:
        if (fn.startswith('collectionName_') == False) and (fn.startswith('phoneName_') == False):
            tmp_feats.append(fn)
    df_train = df_train[tmp_feats]

    # clean time feature
    tmp_drop_feats = []
    for f in df_train.columns:
        if (f.startswith('millisSinceGpsEpoch') == True) or (f.startswith('timeSinceFirstFixSeconds') == True) or (f.startswith('utcTimeMillis') == True):
            tmp_drop_feats.append(f)
    df_train.drop(tmp_drop_feats, axis = 1, inplace = True)
    
    return df_train

# %%
def remove_other_axis_feats(df_all, tgt_axis):
    '''unrelated-aixs features and uncalibrated features'''
    # Clean unrelated-aixs features
    all_imu_feats = ['UncalAccelXMps2', 'UncalAccelYMps2', 'UncalAccelZMps2',
                     'UncalGyroXRadPerSec', 'UncalGyroYRadPerSec', 'UncalGyroZRadPerSec',
                     'UncalMagXMicroT', 'UncalMagYMicroT', 'UncalMagZMicroT',
                     'ahrsX', 'ahrsY', 'ahrsZ',
                     'AccelXMps2', 'AccelYMps2', 'AccelZMps2',
                     'GyroXRadPerSec', 'GyroZRadPerSec', 'GyroYRadPerSec',
                     'MagXMicroT', 'MagYMicroT', 'MagZMicroT',
                     'yawZDeg', 'rollYDeg', 'pitchXDeg',
                     'Xbl', 'Ybl', 'Zbl',
                    'Xaga_mean_predict_phone_mean', 'Xaga_mean_predict_phone_mean',
                    'Xaga_phone_mean_mean_predict', 'Xaga_phone_mean_mean_predict',
                    'Xaga_mean_predict', 'Xaga_mean_predict',
                    'Xaga_phone_mean', 'Xaga_phone_mean',
                    'Xkalman_mean_predict_phone_mean', 'Xkalman_mean_predict_phone_mean',
                    'Xkalman_phone_mean_mean_predict', 'Xkalman_phone_mean_mean_predict',
                    'Xkalman_mean_predict', 'Xkalman_mean_predict',
                    'Xkalman_phone_mean', 'Xkalman_phone_mean',
                    ]
    tgt_imu_feats = []
    for axis in ['X', 'Y', 'Z']:
        if axis != tgt_axis:
            for f in all_imu_feats:
                if f.find(axis) >= 0:
                    tgt_imu_feats.append(f) # tgt_axis以外の軸を取得（'_'なし）
    # print(tgt_imu_feats)
            
    tmp_drop_feats = []
    for f in df_all.columns:
        if f.split('_')[0] in tgt_imu_feats:
            tmp_drop_feats.append(f) # tgt_aixs以外の軸を取得（'_'あり）
    # print(tmp_drop_feats)　

    tgt_df = df_all.drop(tmp_drop_feats, axis = 1) # tgt_aixs以外の軸を削除
    
    # Clean uncalibrated features
    uncal_feats = [f for f in tgt_df.columns if f.startswith('Uncal') == True] # Uncalとつくやつ削除
    tgt_df = tgt_df.drop(uncal_feats, axis = 1)
    
    return tgt_df

def add_stat_feats(data, tgt_axis):
    for f in ['yawZDeg', 'rollYDeg', 'pitchXDeg']:
        if f.find(tgt_axis) >= 0:
            ori_feat = f
            break
            
    # heightAboveWgs84EllipsoidMとtgt_axisが含まれる特徴量
    cont_feats = ['heightAboveWgs84EllipsoidM', 'ahrs{}'.format(tgt_axis),
           'Accel{}Mps2'.format(tgt_axis), 'Gyro{}RadPerSec'.format(tgt_axis), 'Mag{}MicroT'.format(tgt_axis),
            '{}bl'.format(tgt_axis)] + [ori_feat]
    # print(cont_feats)
    
    # window_size内から統計的特徴量作成
    # display(data[[f + f'_{i}' for i in range(1,window_size)]].mean(axis=1))
    for f in cont_feats:
        data[f + '_' + str(window_size) + '_mean'] = data[[f + f'_{i}' for i in range(1,window_size)]].mean(axis=1)
        data[f + '_' + str(window_size) + '_std'] = data[[f + f'_{i}' for i in range(1,window_size)]].std(axis=1)
        data[f + '_' + str(window_size) + '_max'] = data[[f + f'_{i}' for i in range(1,window_size)]].max(axis=1)
        data[f + '_' + str(window_size) + '_min'] = data[[f + f'_{i}' for i in range(1,window_size)]].min(axis=1)
        data[f + '_' + str(window_size) + '_median'] = data[[f + f'_{i}' for i in range(1,window_size)]].median(axis=1)
        data[f + '_' + str(window_size) + '_skew'] = data[[f + f'_{i}' for i in range(1,window_size)]].skew(axis=1)
        data[f + '_' + str(window_size) + '_kurt'] = data[[f + f'_{i}' for i in range(1,window_size)]].kurt(axis=1)
        data[f + '_max_min'] = data[f + '_30_max'] - data[f + '_30_min']
    
    return data

# # RMSを求める関数
# def calc_rms(signal):
#     # print(type(signal))
#     signal_2 = signal * signal        # 二乗
#     signal_sum = np.sum(signal_2)     # 総和
#     signal_sqrt = np.sqrt(signal_sum) # 平方根
#     rms = np.mean(signal_sqrt)        # 平均値

    # return rms

# %%
# prepare_df_test関連
def prepare_df_test(df_all_test, window_size):
    '''prepare testing dataset with all aixses'''
    tgt_df = df_all_test.copy()
    total_len = len(tgt_df) 
    moving_times = total_len - window_size + 1
    
    tgt_df.rename(columns = {'yawDeg':'yawZDeg', 'rollDeg':'rollYDeg', 'pitchDeg':'pitchXDeg'}, inplace = True)

    feature_cols = [f for f in list(tgt_df) if f not in ['Xgt', 'Ygt', 'Zgt']] 
    
    hist_feats = []
    for time_flag in range(1, window_size + 1):
        for fn in feature_cols:
            hist_feats.append(fn + '_' + str(time_flag))

    # t1 t2 t3 t4 t5 -> t6
    # t2 t3 t4 t5 t6 -> t7
    df_test = pd.DataFrame()
    
    features = [tgt_df[feature_cols].iloc[start_idx : start_idx+window_size,:].values.flatten() for start_idx in range(moving_times)]

    df_test = pd.DataFrame(features, columns = hist_feats)

    tmp_feats = []
    for fn in list(df_test):
        if (fn.startswith('collectionName_') == False) and (fn.startswith('phoneName_') == False):
            tmp_feats.append(fn)
    df_test = df_test[tmp_feats]

    tmp_drop_feats = []
    for f in list(df_test):
        if (f.startswith('millisSinceGpsEpoch') == True) or (f.startswith('timeSinceFirstFixSeconds') == True) or (f.startswith('utcTimeMillis') == True) or (f.startswith('elapsedRealtimeNanos') == True):
            tmp_drop_feats.append(f)
    df_test.drop(tmp_drop_feats, axis = 1, inplace = True)
    
    return df_test
# %%
"""code
https://www.kaggle.com/alvinai9603/predict-next-point-with-the-imu-data
"""

bl_trn_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')
bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
sample_df = pd.read_csv('../../data/submission/sample_submission.csv')

# %%
print('Baseline Train shape:', bl_trn_df.shape)
print('Baseline Test shape:', bl_tst_df.shape)
print('Test shape:', sample_df.shape)

# %%
train_pahts = gb.glob('../../data/interim/*_train.csv')

latlng_dict = {}
for train_path in train_pahts:
    key = re.split('/|_train', train_path)[4]
    print(key)
    latlng_dict[key] = pd.read_csv(train_path)

# %%
# rename
for key in latlng_dict.keys():
    print(key)
    latlng_dict[key] = latlng_dict[key].rename(columns={'latDeg':f'latDeg_{key}', 'lngDeg':f'lngDeg_{key}'})
    # display(latlng_dict[key])

# %%
# merge
for key, value in latlng_dict.items():
    print(key)
    # display(value)
    bl_trn_df = pd.merge_asof(
                    bl_trn_df.sort_values('millisSinceGpsEpoch'),
                    value[[f'latDeg_{key}', f'lngDeg_{key}', 'millisSinceGpsEpoch', 'collectionName', 'phoneName']].sort_values('millisSinceGpsEpoch'),
                    on='millisSinceGpsEpoch',
                    by=['collectionName', 'phoneName'],
                    direction='nearest'
    )
bl_trn_df

# %%
bl_trn_df.to_csv('../../data/interim/train/predicted_many_lat_lng_deg.csv', index=False)

# %%
bl_trn_df = pd.read_csv('../../data/interim/train/predicted_many_lat_lng_deg.csv')
bl_trn_df
# %%
# collectionNameとphoneNameの総組み合わせ
cn2pn_df = bl_trn_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df

# %%
# 特徴量作成
collection_names =  [
    '2021-04-22-US-SJC-1',
    '2021-04-26-US-SVL-1',
    '2021-04-28-US-SJC-1',
    '2021-04-29-US-SJC-2',
    '2021-04-15-US-MTV-1',
    '2021-04-28-US-MTV-1',
    '2021-04-29-US-MTV-1',
    '2021-03-10-US-SVL-1',
]
window_size = 30


# %%
%%time
# train
axis_dict = {}
x_df_trains, y_df_trains, z_df_trains = [], [], []
lat_lng_df_trains = []
for tgt_cn in tqdm(collection_names):
    pns = cn2pn_df.loc[cn2pn_df['collectionName'] == tgt_cn, 'phoneName'].values
    # print(tgt_cn, pns)
    for tgt_pn in pns:
        print('Prepare Training Dataset：', tgt_cn + '_' + tgt_pn)  
        df_all_train = prepare_imu_data('train', tgt_cn, tgt_pn, bl_trn_df)
        # df_all_train = prepare_imu_data('train', tgt_cn, tgt_pn, aga_pred_phone_trn_df)
        # display(df_all_train)
        df_all_train = df_all_train.drop_duplicates(subset='millisSinceGpsEpoch').reset_index(drop=True)
        lat_lng_df_train, df_all_train = get_xyz(df_all_train, 'train')
        df_train = prepare_df_train(df_all_train,  window_size) # 所有轴的数据
        print(len(bl_trn_df[(bl_trn_df['collectionName']==tgt_cn) & (bl_trn_df['phoneName']==tgt_pn)]), len(df_all_train), len(df_train))

        for axis in ['X', 'Y', 'Z']:
            axis_df = remove_other_axis_feats(df_train, axis)
            axis_df = add_stat_feats(axis_df, axis)
            axis_dict[axis] = axis_df.copy()
            axis_dict[axis]['collectionName'] = tgt_cn
            axis_dict[axis]['phoneName'] = tgt_pn


        x_df_trains.append(axis_dict['X'])
        y_df_trains.append(axis_dict['Y'])
        z_df_trains.append(axis_dict['Z'])
        lat_lng_df_trains.append(lat_lng_df_train)
        print('_'*20)

x_df_train = pd.concat(x_df_trains, axis = 0)
y_df_train = pd.concat(y_df_trains, axis = 0)
z_df_train = pd.concat(z_df_trains, axis = 0)
lat_lng_df_train = pd.concat(lat_lng_df_trains, axis = 0)

# %%
x_df_train = x_df_train.reset_index(drop=True)
y_df_train = y_df_train.reset_index(drop=True)
z_df_train = z_df_train.reset_index(drop=True)
lat_lng_df_train = lat_lng_df_train.reset_index(drop=True)

print('X; Final Dataset shape：', x_df_train.shape)
print('Y; Final Dataset shape：', y_df_train.shape)
print('Z; Final Dataset shape：', z_df_train.shape)

# %%
print(x_df_train.isnull().sum())
# %%
print(y_df_train.isnull().sum())
# %%
print(z_df_train.isnull().sum())

# %%
x_df_train

# %%
# 保存
x_df_train.to_csv('../../data/processed/train/imu_x_many_lat_lng_deg.csv', index=False)
y_df_train.to_csv('../../data/processed/train/imu_y_many_lat_lng_deg.csv', index=False)
z_df_train.to_csv('../../data/processed/train/imu_z_many_lat_lng_deg.csv', index=False)

# %%
for col in x_df_train.columns:
    j = x_df_train[x_df_train[col].isnull()==True]
    idx = x_df_train[x_df_train[col].isnull()==True].index
    if not j.empty:
        print(col)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(x_df_train.loc[idx, [col, fillcol]])
        display(x_df_train.loc[idx+1, [col, fillcol]])
        display(x_df_train.loc[idx+2, [col, fillcol]])

        # x_df_train.loc[idx, col] = x_df_train.loc[idx+1, fillcol].values
        # display(x_df_train.loc[idx, [col, fillcol]])

# %%
for col in y_df_train.columns:
    j = y_df_train[y_df_train[col].isnull()==True]
    idx = y_df_train[y_df_train[col].isnull()==True].index
    if not j.empty:
        print(col)
        display(j)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(y_df_train.loc[idx, [col, fillcol]])
        display(y_df_train.loc[idx+1, [col, fillcol]])
        display(y_df_train.loc[idx+2, [col, fillcol]])

        # y_df_train.loc[idx, col] = y_df_train.loc[idx+1, fillcol].values
        # display(y_df_train.loc[idx, [col, fillcol]])

# %%
for col in z_df_train.columns:
    j = z_df_train[z_df_train[col].isnull()==True]
    idx = z_df_train[z_df_train[col].isnull()==True].index
    if not j.empty:
        print(col)
        display(j)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(z_df_train.loc[idx, [col, fillcol]])
        display(z_df_train.loc[idx+1, [col, fillcol]])
        display(z_df_train.loc[idx+2, [col, fillcol]])

        # z_df_train.loc[idx, col] = z_df_train.loc[idx+1, fillcol].values
        # display(z_df_train.loc[idx, [col, fillcol]])


# %%
test_pahts = gb.glob('../../data/interim/*_test.csv')

latlng_dict = {}
for test_path in test_pahts:
    key = re.split('/|_test', test_path)[4]
    print(key)
    latlng_dict[key] = pd.read_csv(test_path)

# %%
# rename
for key in latlng_dict.keys():
    print(key)
    latlng_dict[key] = latlng_dict[key].rename(columns={'latDeg':f'latDeg_{key}', 'lngDeg':f'lngDeg_{key}'})
    # display(latlng_dict[key])

# %%
# merge
for key, value in latlng_dict.items():
    print(key)
    # display(value)
    bl_tst_df = pd.merge_asof(
                    bl_tst_df.sort_values('millisSinceGpsEpoch'),
                    value[[f'latDeg_{key}', f'lngDeg_{key}', 'millisSinceGpsEpoch', 'collectionName', 'phoneName']].sort_values('millisSinceGpsEpoch'),
                    on='millisSinceGpsEpoch',
                    by=['collectionName', 'phoneName'],
                    direction='nearest'
    )
bl_tst_df

# %%
bl_tst_df.to_csv('../../data/interim/test/predicted_many_lat_lng_deg.csv', index=False)

# %%
bl_tst_df = pd.read_csv('../../data/interim/test/predicted_many_lat_lng_deg.csv')
bl_tst_df
# %%
# collectionNameとphoneNameの総組み合わせ
cn2pn_df = bl_tst_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df

# %%
collection_names = [
    '2021-04-02-US-SJC-1',
    '2021-04-22-US-SJC-2',
    '2021-04-29-US-SJC-3',
    '2021-03-16-US-MTV-2',
    '2021-04-08-US-MTV-1',
    '2021-04-21-US-MTV-1',
    '2021-04-28-US-MTV-2',
    '2021-04-29-US-MTV-2',
    '2021-04-26-US-SVL-2',
    '2021-03-16-US-RWC-2',
    '2021-03-25-US-PAO-1'
]
"""3月以降他
'2021-03-16-US-RWC-2'
'2021-03-25-US-PAO-1' # テストにしかないID
"""
# %%
%%time
# test
axis_dict = {}
x_df_tests, y_df_tests, z_df_tests = [], [], []
lat_lng_df_tests = []
for tgt_cn in tqdm(collection_names):
    pns = cn2pn_df.loc[cn2pn_df['collectionName'] == tgt_cn, 'phoneName'].values
    for tgt_pn in pns:
        print('\nPrepare Testing Dataset：', tgt_cn + '_' + tgt_pn)  
        df_all_test = prepare_imu_data('test', tgt_cn, tgt_pn, bl_tst_df)
        lat_lng_df_test, df_all_test = get_xyz(df_all_test, 'test')
        df_all_test = df_all_test.drop_duplicates(subset='millisSinceGpsEpoch').reset_index(drop=True)
        df_test = prepare_df_test(df_all_test,  window_size) # 所有轴的数据
        print(len(bl_tst_df[(bl_tst_df['collectionName']==tgt_cn) & (bl_tst_df['phoneName']==tgt_pn)]), len(df_all_test), len(df_test))
        # display(bl_tst_df.loc[(bl_tst_df['collectionName']==tgt_cn) & (bl_tst_df['phoneName']==tgt_pn), ['millisSinceGpsEpoch']])
        # display(df_all_test[['millisSinceGpsEpoch']])
        # display(df_test[['millisSinceGpsEpoch']])

        for axis in ['X', 'Y', 'Z']:
            axis_df = remove_other_axis_feats(df_test, axis)
            axis_df = add_stat_feats(axis_df, axis)
            axis_dict[axis] = axis_df.copy()
            axis_dict[axis]['collectionName'] = tgt_cn
            axis_dict[axis]['phoneName'] = tgt_pn

        x_df_tests.append(axis_dict['X'])
        y_df_tests.append(axis_dict['Y'])
        z_df_tests.append(axis_dict['Z'])
        lat_lng_df_tests.append(lat_lng_df_test)
        print('_'*20)

x_df_test = pd.concat(x_df_tests, axis = 0)
y_df_test = pd.concat(y_df_tests, axis = 0)
z_df_test = pd.concat(z_df_tests, axis = 0)
lat_lng_df_test = pd.concat(lat_lng_df_tests, axis = 0)

# %%
x_df_test = x_df_test.reset_index(drop=True)
y_df_test = y_df_test.reset_index(drop=True)
z_df_test = z_df_test.reset_index(drop=True)
lat_lng_df_test = lat_lng_df_test.reset_index(drop=True)

print('X; Final Dataset shape：', x_df_test.shape)
print('Y; Final Dataset shape：', y_df_test.shape)
print('Z; Final Dataset shape：', z_df_test.shape)

# %%
print(x_df_test.isnull().sum())
# %%
print(y_df_test.isnull().sum())
# %%
print(z_df_test.isnull().sum())

# %%
x_df_test
# %%
# 保存
x_df_test.to_csv('../../data/processed/test/imu_x_many_lat_lng_deg.csv', index=False)
y_df_test.to_csv('../../data/processed/test/imu_y_many_lat_lng_deg.csv', index=False)
z_df_test.to_csv('../../data/processed/test/imu_z_many_lat_lng_deg.csv', index=False)

# %%
for col in x_df_test.columns:
    j = x_df_test[x_df_test[col].isnull()==True]
    idx = x_df_test[x_df_test[col].isnull()==True].index
    if not j.empty:
        print(col)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(x_df_test.loc[idx, [col, fillcol]])
        display(x_df_test.loc[idx+1, [col, fillcol]])
        display(x_df_test.loc[idx+2, [col, fillcol]])

        x_df_test.loc[idx, col] = x_df_test.loc[idx+1, fillcol].values
        display(x_df_test.loc[idx, [col, fillcol]])

# %%
for col in y_df_test.columns:
    j = y_df_test[y_df_test[col].isnull()==True]
    idx = y_df_test[y_df_test[col].isnull()==True].index
    if not j.empty:
        print(col)
        display(j)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(y_df_test.loc[idx, [col, fillcol]])
        display(y_df_test.loc[idx+1, [col, fillcol]])
        display(y_df_test.loc[idx+2, [col, fillcol]])

        y_df_test.loc[idx, col] = y_df_test.loc[idx+1, fillcol].values
        display(y_df_test.loc[idx, [col, fillcol]])

# %%
for col in z_df_test.columns:
    j = z_df_test[z_df_test[col].isnull()==True]
    idx = z_df_test[z_df_test[col].isnull()==True].index
    if not j.empty:
        print(col)
        display(j)
        fillcol = col.split('_')[0] + '_29'
        print(fillcol)
        display(j)
        display(j[[col]])
        display(j[[fillcol]])
        display(z_df_test.loc[idx, [col, fillcol]])
        display(z_df_test.loc[idx+1, [col, fillcol]])
        display(z_df_test.loc[idx+2, [col, fillcol]])

        z_df_test.loc[idx, col] = z_df_test.loc[idx+1, fillcol].values
        display(z_df_test.loc[idx, [col, fillcol]])
# %%
x_df_test = x_df_test.reset_index(drop=True)

# %%
x_df_test[x_df_test['GyroXRadPerSec_30'].isnull()==True]

# %%
x_df_test['GyroXRadPerSec_30']

# %%
print(x_df_test['GyroXRadPerSec_30'].mean())
print((x_df_test.loc[27502, 'GyroXRadPerSec_30'] + x_df_test.loc[27504, 'GyroXRadPerSec_30']) / 2)
0.006540	

# %%
x_df_test[(x_df_test['collectionName']=='2021-04-08-US-MTV-1') & (x_df_test['phoneName']=='Pixel5')].reset_index(drop=True)
# %%
