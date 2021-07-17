# %%
import numpy as np
from cv2 import Rodrigues
import pandas as pd
from pathlib import Path
import pyproj
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=Warning)

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)
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

# test
a_df = prepare_imu_data('train', '2021-04-22-US-SJC-1', 'SamsungS20Ultra', bl_trn_df)
# a_df
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
    df_all['Xbl'], df_all['Ybl'], df_all['Zbl'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x.latDeg_bl, x.lngDeg_bl, x.heightAboveWgs84EllipsoidM), axis=1))
    
    if dataset_name == 'train':
        # gt: lat/lngDeg -> x/y/z
        df_all['Xgt'], df_all['Ygt'], df_all['Zgt'] = zip(*df_all.apply(lambda x: WGS84_to_ECEF(x.latDeg_gt, x.lngDeg_gt, x.heightAboveWgs84EllipsoidM), axis=1))
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_gt','lngDeg_gt', 'latDeg_bl', 'lngDeg_bl']]
        df_all.drop(['latDeg_gt','lngDeg_gt', 'latDeg_bl', 'lngDeg_bl'], axis = 1, inplace = True)
    elif dataset_name == 'test':
        # copy lat/lngDeg
        lat_lng_df = df_all[['latDeg_bl', 'lngDeg_bl']]
        df_all.drop(['latDeg_bl', 'lngDeg_bl', 'latDeg','lngDeg',], axis = 1, inplace = True)     
        
    return lat_lng_df, df_all

# test
b_df, c_df = get_xyz(a_df, 'train')
display(b_df)
display(c_df)

# %%
%%time
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

# test
d_df = prepare_df_train(c_df, 30)
d_df

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
# training関連

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
                     'Xbl', 'Ybl', 'Zbl']
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
    
    return data

def training(df_train, df_test, tgt_axis):
    '''For the given axis target to train the model. Also, it has validation and prediciton.'''
    df_train = remove_other_axis_feats(df_train, tgt_axis) # 必要な特徴量だけ抽出（一軸）
    df_train = add_stat_feats(df_train, tgt_axis)
    df_test = remove_other_axis_feats(df_test, tgt_axis)
    df_test = add_stat_feats(df_test, tgt_axis)
    
    feature_names = [f for f in df_train.columns if f not in ['Xgt', 'Ygt', 'Zgt']] # gt除外
    target = '{}gt'.format(tgt_axis)

    kfold = KFold(n_splits=folds, shuffle=True, random_state=params['seed'])

    pred_valid = np.zeros((len(df_train),)) 
    pred_test = np.zeros((len(df_test),)) 
    scores = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        model = lgb.LGBMRegressor(**params)
        lgb_model = model.fit(X_train, 
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric=params['metric'],
                              early_stopping_rounds=params['early_stopping_rounds'])

        pred_valid[val_idx] = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration_)
        pred_test += lgb_model.predict(df_test[feature_names], num_iteration = lgb_model.best_iteration_)

        scores.append(lgb_model.best_score_['valid']['l2'])
    
    pred_test = pred_test /  kfold.n_splits
    
    if verbose_flag == True:
        print("Each Fold's MSE：{}, Average MSE：{:.4f}".format([np.round(v,2) for v in scores], np.mean(scores)))
        print("-"*60)
    
    return df_train, df_test, pred_valid, pred_test

# test
df_train, df_test, pred_valid, pred_test = training(d_df, df_test, 'X')
display(df_train)
display(df_test)
display(pred_valid)
display(pred_test)

# %%
# From：https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(a**0.5)
    dist = 6_367_000 * c
    return dist

# %%
# 可視化関連
def visualize_trafic(df, center, zoom=15):
    fig = px.scatter_mapbox(df,
                            
                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",
                            
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",
                            
                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()
    
def visualize_collection(df):
    target_df = df
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    
    visualize_trafic(target_df, center)

# %%
# LightGBM
params = {
    'metric':'mse',
    'objective':'regression',
    'seed':2021,
    'boosting_type':'gbdt',
    'early_stopping_rounds':10,
    'subsample':0.7,
    'feature_fraction':0.7,
    'bagging_fraction': 0.7,
    'reg_lambda': 10
}
window_size = 30
verbose_flag = True
folds = 5
# %%
# collectionNameとphoneNameの総組み合わせ
cn2pn_df = bl_trn_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df
# %%
%%time
# Example: I use SJC's dataset for training 
tgt_cns = ['2021-04-22-US-SJC-1', '2021-04-28-US-SJC-1', '2021-04-29-US-SJC-2']
df_trains = []
lat_lng_df_trains = []

for tgt_cn in tqdm(tgt_cns):
    pns = cn2pn_df.loc[cn2pn_df['collectionName'] == tgt_cn, 'phoneName'].values
    
    for tgt_pn in pns:
        print('Prepare Training Dataset：', tgt_cn + '_' + tgt_pn)  
        df_all_train = prepare_imu_data('train', tgt_cn, tgt_pn, bl_trn_df)
        lat_lng_df_train, df_all_train = get_xyz(df_all_train, 'train')
        df_train = prepare_df_train(df_all_train,  window_size) # 所有轴的数据

        df_trains.append(df_train)
        lat_lng_df_trains.append(lat_lng_df_train)
        print('_'*20)
        
df_train = pd.concat(df_trains, axis = 0)
lat_lng_df_train = pd.concat(lat_lng_df_trains, axis = 0)
print('Final Dataset shape：', df_train.shape)
# %%
%%time
# Example: I choose one of SJC collection from the test dataset as my test dataset, you can choose what as you like
cname_test = '2021-04-29-US-SJC-3'
pname_test = 'SamsungS20Ultra'
df_all_test = prepare_imu_data('test', cname_test, pname_test, bl_tst_df)
lat_lng_df_test, df_all_test = get_xyz(df_all_test, 'test')
df_test = prepare_df_test(df_all_test,  window_size)
print('df_test:', df_test.shape)
print('df_test.columns:', df_test.columns)

# %%
df_train
# %%
df_train_x, df_test_x, pred_valid_x, pred_test_x = training(df_train, df_test, 'X')
df_train_y, df_test_y, pred_valid_y, pred_test_y = training(df_train, df_test, 'Y')
df_train_z, df_test_z, pred_valid_z, pred_test_z = training(df_train, df_test, 'Z')
# %%
val_compare_df = pd.DataFrame({'Xgt':df_train_x['Xgt'].values, 'Xpred':pred_valid_x,
                               'Ygt':df_train_y['Ygt'].values, 'Ypred':pred_valid_y,
                                'Zgt':df_train_z['Zgt'].values, 'Zpred':pred_valid_z})

# %%
val_compare_df[['Xgt', 'Xpred']].plot(figsize=(16,8))
# %%
val_compare_df[['Ygt', 'Ypred']].plot(figsize=(16,8))
# %%
val_compare_df[['Zgt', 'Zpred']].plot(figsize=(16,8))

# %%
# xyz -> lng, lat
lng_gt, lat_gt, _ = ECEF_to_WGS84(val_compare_df['Xgt'].values,val_compare_df['Ygt'].values,val_compare_df['Zgt'].values)
lng_pred, lat_pred, _ = ECEF_to_WGS84(val_compare_df['Xpred'].values,val_compare_df['Ypred'].values,val_compare_df['Zpred'].values)
lng_test_pred, lat_test_pred, _ = ECEF_to_WGS84(pred_test_x, pred_test_y, pred_test_z)

    
val_compare_df['latDeg_gt'] = lat_gt
val_compare_df['lngDeg_gt'] = lng_gt
val_compare_df['latDeg_pred'] = lat_pred
val_compare_df['lngDeg_pred'] = lng_pred
test_pred_df = pd.DataFrame({'latDeg':lat_test_pred, 'lngDeg':lng_test_pred})
test_pred_df

# %%
# Baseline vs. GT
lat_lng_df_train['dist'] = calc_haversine(lat_lng_df_train.latDeg_gt, lat_lng_df_train.lngDeg_gt, 
                                lat_lng_df_train.latDeg_bl, lat_lng_df_train.lngDeg_bl)
print('dist_50:',np.percentile(lat_lng_df_train['dist'],50) )
print('dist_95:',np.percentile(lat_lng_df_train['dist'],95) )
print('avg_dist_50_95:',(np.percentile(lat_lng_df_train['dist'],50) + np.percentile(lat_lng_df_train['dist'],95))/2)
print('avg_dist:', lat_lng_df_train['dist'].mean())

# %%
# IMU Prediction vs. GT
val_compare_df['dist'] = calc_haversine(val_compare_df.latDeg_gt, val_compare_df.lngDeg_gt, 
                                val_compare_df.latDeg_pred, val_compare_df.lngDeg_pred)
# IMU预测vsGT（多collection）
print('dist_50:',np.percentile(val_compare_df['dist'],50) )
print('dist_95:',np.percentile(val_compare_df['dist'],95) )
print('avg_dist_50_95:',(np.percentile(val_compare_df['dist'],50) + np.percentile(val_compare_df['dist'],95))/2)
print('avg_dist:', val_compare_df['dist'].mean())

# %%
# Visualization: Train dataset
cname = '2021-04-29-US-SJC-2'
pname = 'SamsungS20Ultra'
# IMU Prediciton
tmp0 = val_compare_df.copy()
tmp0.rename(columns={'latDeg_pred':'latDeg', 'lngDeg_pred':'lngDeg'}, inplace=True)
tmp0['phoneName'] = [cname + '_' + pname + '_imu_pred' for i in range(len(tmp0))]
# GT
tmp1 = val_compare_df.copy()
tmp1.rename(columns={'latDeg_gt':'latDeg', 'lngDeg_gt':'lngDeg'}, inplace=True)
tmp1['phoneName'] = [cname + '_' + pname + '_gt' for i in range(len(tmp1))]
# Baseline
tmp2 = lat_lng_df_train.copy()
tmp2.rename(columns={'latDeg_bl':'latDeg', 'lngDeg_bl':'lngDeg'}, inplace=True)
tmp2['phoneName'] = [cname + '_' + pname + '_bl_pred' for i in range(len(tmp2))]

tmp = pd.concat([tmp0, tmp1, tmp2])
visualize_collection(tmp)

# %%
# Visualization:: Test dataset
cname = '2021-04-29-US-SJC-3'
pname = 'SamsungS20Ultra'
tmp3 = test_pred_df.copy()
tmp3['phoneName'] = cname_test + '_' + pname_test + '_imu_pred' 

tmp4 = bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cname_test + '_' + pname_test].index[window_size:],3:5].copy()
tmp4['phoneName'] = cname_test + '_' + pname_test + '_bl_pred' 

tmp5 = pd.concat([tmp3, tmp4])
visualize_collection(tmp5)

# %%
# subに代入
bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cname_test + '_' + pname_test].index[window_size:],3] = test_pred_df['latDeg'].values
bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cname_test + '_' + pname_test].index[window_size:],4] = test_pred_df['lngDeg'].values

# %%
# save
bl_tst_df = bl_tst_df[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']]
bl_tst_df.to_csv(f'../../data/submission/imu_test_only_{cname_test}-{pname_test}.csv', index=False)